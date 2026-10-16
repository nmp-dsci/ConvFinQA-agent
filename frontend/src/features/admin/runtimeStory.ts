/**
 * The runtime comparison, as data shaping.
 *
 * The page this backs makes one claim — move the runtime to the Claude Agent
 * SDK single-session agent — and the claim has to be *derived*, not typed. So
 * everything that decides what the page says lives here as pure functions over
 * the `/eval/campaigns` payload, and the components below only render what
 * these return. Three rules hold throughout:
 *
 *  1. `null` in means absent out. A stage with no run is `present: false` and
 *     prints "not yet run"; it never becomes a zero, a dash on a zero baseline,
 *     or a bar of no height.
 *  2. The recommendation is a function of the gate (`promoted` AND a positive
 *     paired delta), never a literal. If the gate flips, the banner flips.
 *  3. The only literals about *accuracy* permitted here are the paper's human
 *     figures, and they are not written here either — they are read out of
 *     `features/system/paper.ts`, where every number already carries the table
 *     it came from and a test fails the build if one does not.
 */

import type {
  CampaignExperiment,
  CampaignSummary,
  ChampionAgent,
  ChampionPoint,
  Challengers,
  JudgeSummary,
  RuntimeArm,
  RuntimeComparison,
  RuntimeGate,
  RuntimeSlice,
  SdkModelComparison,
  TestPass,
} from './api';
import { NO_VALUE } from '../landing/format';
import { BASELINES, PAPER } from '../system/paper';

/** The significance level both arms' gates are judged at. */
export const ALPHA = 0.05;

/** The four subagent metrics, in pipeline order. */
export const PANEL_STAGES = ['triage', 'preprocess', 'retriever', 'calculator'] as const;

// ---------------------------------------------------------------------------
// The published ceiling
// ---------------------------------------------------------------------------

function ratio(value: string | null | undefined): number | null {
  const n = value === null || value === undefined ? NaN : Number.parseFloat(value);
  return Number.isFinite(n) ? n / 100 : null;
}

/**
 * The paper's human-expert row, lifted from the benchmark table rather than
 * retyped.
 *
 * This is the one thing on the page that is not a measurement of this system,
 * so it travels with its citation and is labelled as published everywhere it is
 * drawn. `paper.ts` is the single place the figure exists, and `paper.test.ts`
 * already requires it to name the table it came from.
 */
const CEILING = BASELINES.find((row) => row.ceiling);

export const PAPER_HUMAN = {
  label: CEILING?.system ?? 'Human expert',
  /** Execution accuracy — the same quantity both arms are scored on. */
  exe: ratio(CEILING?.exe),
  /** Program accuracy — the quantity neither arm comes close to. */
  prog: ratio(CEILING?.prog),
  evaluatedOn: CEILING?.evaluatedOn ?? '',
  citation: `${PAPER.authors}, ${PAPER.venue}${CEILING ? ` — ${CEILING.source}` : ''}`,
} as const;

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------

export type RuntimeKey = 'pipeline' | 'agent_sdk';

export interface ArmDescription {
  key: RuntimeKey;
  /** What the arm *is*, in the fewest words that stay true. */
  title: string;
  /** The architecture, spelled out — this is not a measurement. */
  architecture: string;
  aliasNote: string;
}

/**
 * What each arm is, as prose. Deliberately architectural and not numeric: the
 * shape of a runtime is not something the tracking store records, and it is the
 * one thing a reader cannot infer from the figures beside it.
 */
export const ARMS: Record<RuntimeKey, ArmDescription> = {
  pipeline: {
    key: 'pipeline',
    title: 'four DeepSeek agents',
    architecture:
      'triage → preprocess → retriever → calculator, in that fixed order, one prompt and one model call each.',
    aliasNote: 'pipeline_champion — the alias the pipeline campaigns move',
  },
  agent_sdk: {
    key: 'agent_sdk',
    title: 'one Claude session',
    architecture:
      'one Claude Agent SDK session per conversation, the six calculator functions as its only tools, one prompt; it reports the same four stages so the same panel applies.',
    aliasNote:
      'champion — the universal champion since 2026-10-15; the pipeline keeps pipeline_champion',
  },
};

/** True when this arm has been run at all. */
export function armPresent(arm: RuntimeArm | null | undefined): boolean {
  return Boolean(arm && (arm.accuracy !== null || arm.version));
}

// ---------------------------------------------------------------------------
// The verdict
// ---------------------------------------------------------------------------

export type Recommendation = 'adopt-agent-sdk' | 'stay-on-pipeline' | 'not-yet-run';

export interface RuntimeVerdict {
  recommendation: Recommendation;
  /** The one line the page leads with. */
  headline: string;
  /** Why, in the terms the gate decides on. */
  because: string;
  promoted: boolean | null;
  deltaPp: number | null;
  pValue: number | null;
  /** the cluster bootstrap interval on Δ, as a *fraction* (−0.069 = −6.9pp), unlike deltaPp */
  ciLo: number | null;
  ciHi: number | null;
  fixed: number | null;
  broken: number | null;
  baselineVersion: string | null;
  candidateVersion: string | null;
  gateId: string | null;
  gateQuestions: number | null;
  gateReports: number | null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * The recommendation, derived.
 *
 * `promoted` alone is not enough and neither is the delta alone: a gate row can
 * carry a positive delta it refused to promote (net positive but not
 * significant, which is most of this project's record), and a promoted row with
 * a non-positive delta would be a bug worth surfacing rather than reading as a
 * recommendation. Both, or the page recommends nothing.
 */
export function runtimeVerdict(
  comparison: RuntimeComparison | null | undefined,
  split: Record<string, unknown> | undefined,
): RuntimeVerdict {
  const gate: RuntimeGate | null = comparison?.gate ?? null;
  const ci = gate?.ci ?? [null, null];
  const deltaPp = numberOrNull(gate?.delta_pp);
  const promoted = gate?.promoted ?? null;
  const pipelineVersion = comparison?.pipeline?.version ?? null;
  const candidateVersion = gate?.candidate_version ?? comparison?.agent_sdk?.version ?? null;

  const measured = deltaPp !== null && promoted !== null;
  const recommendation: Recommendation = !measured
    ? 'not-yet-run'
    : promoted && deltaPp > 0
      ? 'adopt-agent-sdk'
      : 'stay-on-pipeline';

  const headline =
    recommendation === 'adopt-agent-sdk'
      ? `Move the runtime to the Claude Agent SDK single-session agent${
          candidateVersion ? ` (${candidateVersion})` : ''
        }.`
      : recommendation === 'stay-on-pipeline'
        ? `Keep the four-agent pipeline${
            pipelineVersion ? ` (${pipelineVersion})` : ''
          } — the single-session arm did not clear the gate.`
        : 'No cross-runtime gate has been run yet, so there is no recommendation to make.';

  const because =
    recommendation === 'not-yet-run'
      ? 'Run both arms over the same gate split and gate them against each other.'
      : recommendation === 'adopt-agent-sdk'
        ? 'The single-session arm is net positive on the shared gate questions and clears the one-sided cluster-corrected McNemar test.'
        : 'The paired comparison did not clear the promotion rule, so the incumbent stands.';

  return {
    recommendation,
    headline,
    because,
    promoted,
    deltaPp,
    pValue: numberOrNull(gate?.p_value),
    ciLo: numberOrNull(ci?.[0]),
    ciHi: numberOrNull(ci?.[1]),
    fixed: numberOrNull(gate?.fixed),
    broken: numberOrNull(gate?.broken),
    baselineVersion: pipelineVersion,
    candidateVersion,
    gateId: (gate?.gate_id as string | undefined) ?? null,
    gateQuestions: numberOrNull(split?.gate_questions),
    gateReports: numberOrNull(split?.gate_reports),
  };
}

// ---------------------------------------------------------------------------
// Where the difference is
// ---------------------------------------------------------------------------

export type SliceEffect = 'significant' | 'no-effect' | 'not-significant';

export interface SliceRow {
  key: 'program' | 'number';
  label: string;
  what: string;
  n: number | null;
  baseline: number | null;
  candidate: number | null;
  deltaPp: number | null;
  fixed: number | null;
  broken: number | null;
  pValue: number | null;
  effect: SliceEffect;
  /** The verdict in words, because "0.00pp" alone reads as a missing number. */
  effectLabel: string;
}

const SLICE_LABELS: Record<'program' | 'number', { label: string; what: string }> = {
  program: {
    label: 'program turns',
    what: 'a computation: plan the sub-questions, retrieve the operands, run the DSL',
  },
  number: {
    label: 'number turns',
    what: 'a lookup: one value out of the table or the prose',
  },
};

/**
 * Classify one slice.
 *
 * A delta that rounds to 0.00pp with an equal number of flips either way is not
 * a small effect, it is no effect, and saying so is the whole point of splitting
 * the gate: an aggregate +8.88pp that is +13.03pp on one slice and exactly zero
 * on the other describes neither slice. "Not significant" is kept distinct from
 * "no effect" because a −2.87pp slice that failed the test is a different claim
 * from a slice where nothing moved.
 */
export function sliceEffect(deltaPp: number | null, pValue: number | null): SliceEffect {
  if (deltaPp === null) return 'not-significant';
  if (pValue !== null && pValue < ALPHA && deltaPp > 0) return 'significant';
  // The gate reports deltas rounded to four decimal places of a percentage
  // point, so "exactly zero" is a hair either side of it.
  if (Math.abs(deltaPp) < 0.005) return 'no-effect';
  return 'not-significant';
}

const EFFECT_LABEL: Record<SliceEffect, string> = {
  significant: 'significant',
  'no-effect': 'no effect',
  'not-significant': 'not significant',
};

/**
 * The gate's per-turn-type split, program first.
 *
 * Program first because that slice carries the entire aggregate gain, and a
 * reader who stops after one row should have read the row that matters.
 */
export function sliceRows(gate: RuntimeGate | null | undefined): SliceRow[] {
  const by = gate?.by_turn_type ?? null;
  if (!by) return [];
  const out: SliceRow[] = [];
  for (const key of ['program', 'number'] as const) {
    const slice: RuntimeSlice | undefined = by[key];
    if (!slice) continue;
    const deltaPp = numberOrNull(slice.delta_pp);
    const pValue = numberOrNull(slice.cluster_p_one_sided);
    const effect = sliceEffect(deltaPp, pValue);
    out.push({
      key,
      label: SLICE_LABELS[key].label,
      what: SLICE_LABELS[key].what,
      n: numberOrNull(slice.n),
      baseline: numberOrNull(slice.baseline_accuracy),
      candidate: numberOrNull(slice.candidate_accuracy),
      deltaPp,
      fixed: numberOrNull(slice.fixed),
      broken: numberOrNull(slice.broken),
      pValue,
      effect,
      effectLabel: EFFECT_LABEL[effect],
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// The progression
// ---------------------------------------------------------------------------

/** One row of the judge table: a version scored on a split. */
export interface JudgeRow {
  version: string;
  split: 'calibrate' | 'test';
  isChampion: boolean;
  n: number;
  nWrong: number;
  runtimeAccuracy: number;
  coverage: number;
  highBandAccuracy: number | null;
  nHighWrong: number;
  nHigh: number;
  unseenErrorUpper: number | null;
  failureCapture: number | null;
  nCaught: number;
  meetsTarget: boolean;
  auroc: number | null;
}

/**
 * The judge table, one row per (version, split), calibrate before test, the
 * champion's test row being the headline. Optimise is deliberately absent:
 * it is balanced 50/50 and its coverage means nothing at production prevalence.
 */
export function judgeRows(summary: JudgeSummary | null | undefined): JudgeRow[] {
  if (!summary?.versions?.length) return [];
  const rows: JudgeRow[] = [];
  for (const entry of summary.versions) {
    for (const split of ['calibrate', 'test'] as const) {
      const m = entry.splits?.[split];
      if (!m) continue;
      rows.push({
        version: entry.version,
        split,
        isChampion: entry.version === summary.champion,
        n: m.n,
        nWrong: m.n_wrong,
        runtimeAccuracy: m.accuracy,
        coverage: m.coverage,
        highBandAccuracy: m.high_band_accuracy,
        nHighWrong: m.n_high_wrong,
        nHigh: m.n_high,
        unseenErrorUpper: m.high_band_error_upper95,
        failureCapture: m.failure_capture,
        nCaught: m.n_failures_caught,
        meetsTarget: m.meets_target,
        auroc: m.auroc,
      });
    }
  }
  return rows;
}

/** The champion judge's test-split row, or null before one exists. */
export function judgeHeadline(summary: JudgeSummary | null | undefined): JudgeRow | null {
  return judgeRows(summary).find((r) => r.isChampion && r.split === 'test') ?? null;
}

export type StageKey =
  | 'pipeline_raw'
  | 'pipeline_optimised'
  | 'sdk_distilled'
  | 'sdk_optimised'
  | 'sdk_model_swap'
  | 'challenger';

/**
 * What a bar's colour says, and the word beside it (status is never colour
 * alone): the agent serving today in the accent; a pass — promoted, beats,
 * matches — in ok; a rejection or a challenger that falls short hatched in
 * red; anything that was only ever a starting point neutral.
 */
export type PointTone = 'current' | 'ok' | 'fail' | 'neutral';

export type PointOutcome =
  | 'champion'
  | 'promoted'
  | 'passed'
  | 'beats'
  | 'matches'
  | 'rejected'
  | 'falls_short'
  | 'baseline';

export interface ProgressionPoint {
  key: StageKey;
  /** Unique per point: a challenger stage repeats, one per agent. */
  id: string;
  /** The stage as the story tells it. */
  stage: string;
  runtime: RuntimeKey;
  version: string | null;
  /** sdk: the model id; pipeline: the model its four agents run on, when recorded. */
  model: string | null;
  /** sdk only. */
  effort: string | null;
  /** The day the pass ran (YYYY-MM-DD), from `test_timeline`; null when not on record. */
  date: string | null;
  accuracy: number | null;
  present: boolean;
  /** What happened at this stage, or why it is absent. */
  note: string;
  /** Only meaningful for a gated stage; null where nothing was gated. */
  promoted: boolean | null;
  /** What the record says about it; null for an absent stage. */
  outcome: PointOutcome | null;
  tone: PointTone;
}

/** Everything newer than the five-stage story: the record the timeline is drawn from. */
export interface ProgressionRecord {
  challengers?: Challengers | null;
  timeline?: TestPass[];
  championAgent?: ChampionAgent | null;
}

const PASSING: ReadonlySet<PointOutcome> = new Set(['promoted', 'passed', 'beats', 'matches']);
const FAILING: ReadonlySet<PointOutcome> = new Set(['rejected', 'falls_short']);

function toneOf(outcome: PointOutcome | null, current: boolean): PointTone {
  if (current) return 'current';
  if (outcome && FAILING.has(outcome)) return 'fail';
  if (outcome && PASSING.has(outcome)) return 'ok';
  return 'neutral';
}

/** `falls_short` → `falls short`; the word under the bar beside its glyph. */
export function outcomeWord(point: Pick<ProgressionPoint, 'outcome' | 'tone'>): string {
  if (point.tone === 'current') return '● champion';
  switch (point.outcome) {
    case 'promoted':
    case 'passed':
    case 'beats':
    case 'matches':
      return `● ${point.outcome}`;
    case 'rejected':
      return '✕ rejected';
    case 'falls_short':
      return '✕ falls short';
    case 'champion':
      return '○ champion then';
    case 'baseline':
      return '○ baseline';
    default:
      return '○ not gated';
  }
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/**
 * `2026-09-05` → `5 Sep 2026`, read as written: the date is the run name's own
 * stamp, and parsing it through `Date` would move it a day in some timezones.
 */
export function formatDay(date: string | null | undefined, withYear = true): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date ?? '');
  if (!m) return NO_VALUE;
  const month = MONTHS[Number(m[2]) - 1];
  if (!month) return NO_VALUE;
  return `${Number(m[3])} ${month}${withYear ? ` ${m[1]}` : ''}`;
}

/** The short name a run of this model carries; the pipeline's four agents share one. */
export function agentShortName(point: Pick<ProgressionPoint, 'runtime' | 'model'>): string {
  if (!point.model) return point.runtime === 'pipeline' ? 'four agents' : 'model not recorded';
  if (point.runtime === 'pipeline') return `${modelShortName(point.model).split('-')[0]} ×4`;
  return modelShortName(point.model);
}

function isCurrent(
  champion: ChampionAgent | null | undefined,
  version: string | null,
  model: string | null,
  effort: string | null,
): boolean {
  if (!champion || !version) return false;
  return champion.version === version && champion.model === model && champion.effort === effort;
}

/** An SDK experiment that is a baseline read-out rather than an optimisation. */
export function isBaselineExperiment(exp: CampaignExperiment): boolean {
  return (exp.target_class || exp.target_agent) === 'baseline';
}

/**
 * Every stage of the story, in the order it happened.
 *
 * The pipeline points come from `champion_track` — first entry is the raw
 * multi-agent system, last is what the campaigns optimised it into — the SDK
 * points from the comparison arm and the SDK campaign's own experiments. Then
 * the challenger agents: with `record.challengers`, one point per agent that
 * ran on another model or effort than the champion's (s19, s21), each paired
 * against the champion's committed runs and carrying its verdict; without it,
 * the single model-swap stage the record held before s21 (the same prompt on a
 * second model). A stage with nothing behind it is returned `present: false`
 * with the reason, so the chart can draw an empty slot rather than silently
 * having fewer columns. Dates come from `record.timeline`, matched by the run
 * each point measured; the champion is drawn as such only when it is the agent
 * `record.championAgent` names — version, model and effort.
 */
export function progression(
  track: ChampionPoint[] | undefined,
  comparison: RuntimeComparison | null | undefined,
  sdkExperiments: CampaignExperiment[] | undefined,
  modelComparison?: SdkModelComparison | null,
  record: ProgressionRecord = {},
): ProgressionPoint[] {
  const points = (track ?? []).filter((p) => numberOrNull(p.accuracy) !== null);
  const raw = points[0] ?? null;
  const last = points.length > 1 ? points[points.length - 1] : null;
  const pipelineArm = comparison?.pipeline ?? null;
  // Fallback for the case where the champion has never moved: the track holds
  // one point, but the comparison arm still names the version that was gated
  // against the SDK. Only used when it is genuinely a different version.
  const optimised =
    last ??
    (pipelineArm && pipelineArm.version && pipelineArm.version !== raw?.version
      ? { version: pipelineArm.version, accuracy: pipelineArm.accuracy }
      : null);

  const sdk = comparison?.agent_sdk ?? null;
  const attempts = (sdkExperiments ?? []).filter((e) => !isBaselineExperiment(e));
  const attempt = attempts.length ? attempts[attempts.length - 1] : null;
  // The first model that is not the reference: a scoring pass of the same
  // prompt, so its accuracy sits on the same axis as the distilled point.
  const swap = modelRows(modelComparison).find((r) => !r.isReference) ?? null;

  const timeline = (record.timeline ?? []).filter((r) => r.complete);
  const champion = record.championAgent ?? null;
  const byRun = new Map(timeline.map((r) => [r.run_name, r]));
  const pipelinePass = (version: string | null | undefined) =>
    version
      ? (timeline.find(
          (r) => r.runtime === 'pipeline' && r.version === version && r.champion_run,
        ) ??
        timeline.find((r) => r.runtime === 'pipeline' && r.version === version) ??
        null)
      : null;
  const rawPass = pipelinePass(raw?.version);
  const optPass = pipelinePass(optimised?.version);
  const sdkPass = sdk?.run_name ? (byRun.get(sdk.run_name) ?? null) : null;
  const attemptPass = attempt
    ? (timeline.find(
        (r) =>
          r.runtime === 'agent_sdk' &&
          r.version === attempt.candidate_version &&
          (r.outcome === 'rejected' || r.outcome === 'promoted' || r.outcome === 'passed'),
      ) ?? null)
    : null;
  const sdkModel = sdk?.model ?? sdkPass?.model ?? null;
  const sdkEffort = sdk?.effort ?? sdkPass?.effort ?? null;

  const stage = (
    base: Omit<ProgressionPoint, 'id' | 'tone'> & {
      id?: string;
      current?: boolean;
    },
  ): ProgressionPoint => {
    const { current, ...rest } = base;
    return {
      ...rest,
      id: base.id ?? base.key,
      tone: base.present ? toneOf(base.outcome, Boolean(current)) : 'neutral',
    };
  };

  const fixed: ProgressionPoint[] = [
    stage({
      key: 'pipeline_raw',
      stage: 'multi-agent, raw',
      runtime: 'pipeline',
      version: raw?.version ?? null,
      model: rawPass?.model ?? null,
      effort: null,
      date: rawPass?.date ?? null,
      accuracy: numberOrNull(raw?.accuracy),
      present: Boolean(raw),
      note: raw
        ? 'four prompts as first written, before any campaign'
        : 'no pipeline run on the gate split yet',
      promoted: null,
      outcome: raw ? 'baseline' : null,
    }),
    stage({
      key: 'pipeline_optimised',
      stage: 'multi-agent, optimised',
      runtime: 'pipeline',
      version: optimised?.version ?? null,
      model: optPass?.model ?? null,
      effort: null,
      date: optPass?.date ?? null,
      accuracy: numberOrNull(optimised?.accuracy),
      present: Boolean(optimised && numberOrNull(optimised.accuracy) !== null),
      note: optimised
        ? 'the champion the campaigns arrived at, one subagent per experiment'
        : 'the campaigns have not moved the champion yet',
      promoted: null,
      // It became the pipeline's champion by passing its gate (c01-e03).
      outcome: optimised ? (last ? 'promoted' : 'baseline') : null,
    }),
    stage({
      key: 'sdk_distilled',
      stage: 'SDK, distilled',
      runtime: 'agent_sdk',
      version: sdk?.version ?? null,
      model: sdkModel,
      effort: sdkEffort,
      date: sdkPass?.date ?? null,
      accuracy: numberOrNull(sdk?.accuracy),
      present: armPresent(sdk),
      note: sdk
        ? 'the four tuned prompts distilled into one, run as a single session'
        : 'the single-session arm has not been run',
      promoted: null,
      outcome: armPresent(sdk) ? (comparison?.gate?.promoted ? 'promoted' : 'baseline') : null,
      current: isCurrent(champion, sdk?.version ?? null, sdkModel, sdkEffort),
    }),
    stage({
      key: 'sdk_optimised',
      // Short enough for one chart slot at 13px.
      stage: 'SDK, loop attempt',
      runtime: 'agent_sdk',
      version: attempt?.candidate_version ?? null,
      model: attemptPass?.model ?? null,
      effort: attemptPass?.effort ?? null,
      date: attemptPass?.date ?? null,
      accuracy: numberOrNull(attempt?.accuracy_candidate),
      present: Boolean(attempt && numberOrNull(attempt.accuracy_candidate) !== null),
      note: attempt
        ? attempt.promoted
          ? 'the loop improved on the distilled prompt'
          : 'the loop tried and was rejected by the gate — the distilled prompt stands'
        : 'no optimisation experiment has been gated on this arm',
      promoted: attempt ? attempt.promoted : null,
      outcome: attempt ? (attempt.promoted ? 'promoted' : 'rejected') : null,
      current: isCurrent(
        champion,
        attempt?.candidate_version ?? null,
        attemptPass?.model ?? null,
        attemptPass?.effort ?? null,
      ),
    }),
  ];

  const pairs = record.challengers?.pairs ?? null;
  if (pairs === null) {
    // The record before s21: one model swap, read off the model comparison.
    const swapPass = swap?.runName ? (byRun.get(swap.runName) ?? null) : null;
    return [
      ...fixed,
      stage({
        key: 'sdk_model_swap',
        stage: swap ? `SDK on ${swap.shortName}` : 'SDK, model swap',
        runtime: 'agent_sdk',
        version: swap ? `${sdk?.version ?? 'sdk'} · ${swap.shortName}` : null,
        model: swap?.model ?? null,
        effort: swap?.effort ?? null,
        date: swapPass?.date ?? null,
        accuracy: swap?.accuracy ?? null,
        present: Boolean(swap && swap.accuracy !== null),
        note: swap
          ? `the distilled prompt scored on ${swap.model} with nothing else changed — a scoring pass, not an experiment`
          : 'the sdk champion has been scored on one model only',
        promoted: null,
        outcome: swap?.verdict ?? null,
      }),
    ];
  }

  const ordered = [...pairs].sort((a, b) => (a.at ?? 0) - (b.at ?? 0));
  const challengerPoints = ordered.map((pair) => {
    const pass = pair.candidate_run ? (byRun.get(pair.candidate_run) ?? null) : null;
    const accuracy = numberOrNull(pair.candidate_accuracy) ?? numberOrNull(pass?.accuracy);
    const outcome = pair.verdict?.outcome ?? null;
    const sameVersion = pair.version === (pair.baseline_version ?? sdk?.version);
    return stage({
      key: 'challenger',
      id: `challenger:${pair.version ?? ''}|${pair.candidate_model}|${pair.candidate_effort ?? ''}`,
      stage: sameVersion ? 'model swap' : 'new prompt',
      runtime: 'agent_sdk',
      version: pair.version ?? null,
      model: pair.candidate_model,
      effort: pair.candidate_effort ?? null,
      date: pass?.date ?? null,
      accuracy,
      present: accuracy !== null,
      note:
        accuracy === null
          ? 'no complete test pass of this agent yet'
          : sameVersion
            ? `the champion's prompt on ${pair.candidate_model} at effort ${pair.candidate_effort ?? '—'}, nothing else changed — ${pair.verdict?.reason ?? 'no verdict yet'}`
            : `a prompt the loop wrote for ${pair.candidate_model} at effort ${pair.candidate_effort ?? '—'}, judged against the champion's own runs — ${pair.verdict?.reason ?? 'no verdict yet'}`,
      promoted: null,
      outcome,
      current: isCurrent(
        champion,
        pair.version ?? null,
        pair.candidate_model,
        pair.candidate_effort ?? null,
      ),
    });
  });
  return [...fixed, ...challengerPoints];
}

/** The SDK campaign's own summary, if the payload carries one. */
export function sdkCampaign(campaigns: CampaignSummary[] | undefined): CampaignSummary | null {
  return campaigns?.length ? campaigns[0] : null;
}

// ---------------------------------------------------------------------------
// Formatting the two things `landing/format` does not cover
// ---------------------------------------------------------------------------

/** `786.8` → `13 min`; `44` → `44s`. Wall clock, at the precision it deserves. */
// ---------------------------------------------------------------------------
// The model swap: one prompt, several models
// ---------------------------------------------------------------------------

export type ModelEffect = 'reference' | 'better' | 'worse' | 'no-difference' | 'not-measured';

export interface ModelRow {
  model: string;
  /** The effort the agent ran at — with the model, what the agent is. */
  effort: string | null;
  /** `haiku-4-5` — the slug the run name carries; what the table leads with. */
  shortName: string;
  runName: string | null;
  accuracy: number | null;
  number: number | null;
  program: number | null;
  programAccuracy: number | null;
  cost: number | null;
  wall: number | null;
  isReference: boolean;
  deltaPp: number | null;
  pValue: number | null;
  /** the cluster bootstrap interval on Δ, as a *fraction* (−0.069 = −6.9pp), unlike deltaPp */
  ciLo: number | null;
  ciHi: number | null;
  fixed: number | null;
  broken: number | null;
  effect: ModelEffect;
  effectLabel: string;
  /** The s19 verdict on a challenger: beats / matches / falls_short; null = not decided. */
  verdict: 'beats' | 'matches' | 'falls_short' | null;
  verdictReason: string | null;
  /** Train, per conversation: right to the end, champion → challenger, of n. */
  trainBaselineFull: number | null;
  trainCandidateFull: number | null;
  trainConversations: number | null;
  trainPWorse: number | null;
}

/** `claude-haiku-4-5-20251001` → `haiku-4-5`, the same rule the run name uses. */
export function modelShortName(model: string): string {
  const parts = model
    .toLowerCase()
    .replace(/^claude-/, '')
    .split('-')
    .filter(Boolean);
  const last = parts[parts.length - 1];
  if (parts.length > 1 && /^\d{8}$/.test(last)) parts.pop();
  return parts.join('-') || model;
}

/**
 * The rows of the model-swap table: the reference model first, then each other
 * model with its paired verdict against the reference. Empty until a second
 * model has been scored — one model is a figure, not a comparison — so the
 * panel can say so rather than render a one-row table that looks like a result.
 */
export function modelRows(comparison: SdkModelComparison | null | undefined): ModelRow[] {
  const models = comparison?.models ?? [];
  if (models.length < 2) return [];
  // An agent is (model, effort): two efforts of one model are two rows.
  const key = (model: string | null | undefined, effort: string | null | undefined) =>
    `${model ?? ''}|${effort ?? ''}`;
  const referenceKey = key(comparison?.reference_model, comparison?.reference_effort);
  const pairs = new Map(
    (comparison?.pairs ?? []).map((p) => [key(p.candidate_model, p.candidate_effort), p]),
  );
  return models.map((arm) => {
    const model = arm.model ?? '';
    const effort = arm.effort ?? null;
    const isReference = key(model, effort) === referenceKey;
    const pair = pairs.get(key(model, effort)) ?? null;
    const pooled = pair?.train?.pooled ?? null;
    const deltaPp = numberOrNull(pair?.delta_pp);
    let effect: ModelEffect;
    if (isReference) effect = 'reference';
    else if (deltaPp === null) effect = 'not-measured';
    else if (pair?.significant) effect = deltaPp > 0 ? 'better' : 'worse';
    else effect = 'no-difference';
    const effectLabel = {
      reference: 'reference',
      better: 'significantly better',
      worse: 'significantly worse',
      'no-difference': 'no significant difference',
      'not-measured': 'not measured',
    }[effect];
    return {
      model,
      effort,
      shortName: modelShortName(model),
      runName: arm.run_name ?? null,
      accuracy: numberOrNull(arm.accuracy),
      number: numberOrNull(arm.by_turn_type?.number),
      program: numberOrNull(arm.by_turn_type?.program),
      programAccuracy: numberOrNull(arm.program_accuracy),
      cost: numberOrNull(arm.cost),
      wall: numberOrNull(arm.wall),
      isReference,
      deltaPp,
      pValue: numberOrNull(pair?.p_value),
      ciLo: numberOrNull(pair?.ci?.[0]),
      ciHi: numberOrNull(pair?.ci?.[1]),
      fixed: numberOrNull(pair?.fixed),
      broken: numberOrNull(pair?.broken),
      effect,
      effectLabel,
      verdict: pair?.verdict?.outcome ?? null,
      verdictReason: pair?.verdict?.reason ?? null,
      trainBaselineFull: numberOrNull(pooled?.baseline_full),
      trainCandidateFull: numberOrNull(pooled?.candidate_full),
      trainConversations: numberOrNull(pooled?.n_conversations),
      trainPWorse: numberOrNull(pooled?.p_worse),
    };
  });
}

export function formatWall(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return NO_VALUE;
  if (seconds < 90) return `${Math.round(seconds)}s`;
  return `${Math.round(seconds / 60)} min`;
}

/**
 * A p-value that stays honest at both ends: `0.000286` → `2.9e-4`, not `0.000`,
 * which reads as zero, and `0.917` → `0.917`.
 */
export function formatP(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return NO_VALUE;
  if (p > 0 && p < 0.001) return p.toExponential(1).replace('e-', 'e−');
  return p.toFixed(3);
}

/** A signed percentage-point figure already *in* points: `13.0252` → `+13.03pp`. */
export function formatPp(points: number | null | undefined, digits = 2): string {
  if (points === null || points === undefined || !Number.isFinite(points)) return NO_VALUE;
  return `${points > 0 ? '+' : ''}${points.toFixed(digits)}pp`;
}
