/**
 * The whole record on the gate split, as a model a chart, a ledger and a sentence can all read
 * (s21). Pure functions over `story.test_timeline` and `story.champion_steps`: no fetch, no React,
 * a test per rule.
 *
 * Four rules, each one the reason a figure would otherwise mislead:
 *
 *  1. **Scoring order on x, banded by date.** Thirteen passes ran on four days spread over six
 *     weeks; a calendar axis would stack eight of them on one day and leave a gap of forty days
 *     looking like the main event. Each pass gets a slot, the slots are grouped by the day their run
 *     name stamps, and a break mark sits between two days that are not consecutive.
 *  2. **An incomplete pass is never drawn.** A pass with unscored turns measured a subset of the
 *     split (the rate-limited sdk_v1 pass read 44.4%); it stays in the ledger with its reason and
 *     stays off the axis, where it would read as a measurement.
 *  3. **The champion line moves only on a promotion.** It steps at the pass that measured the new
 *     champion (`effective_at_ms`), not when the promotion was written, and a step that names the
 *     agent already leading — the universal-champion record of 2026-10-15 — is folded into the step
 *     it restates, not drawn as a second one.
 *  4. **Nothing is counted by hand.** Every count in a sentence is a length of a list here.
 */
import type { ChampionStep, PassOutcome, TestPass, TrainPass } from '../admin/api';
import { versionLabel } from '../admin/lib';
import { NO_VALUE, formatPercent } from '../landing/format';

/** `claude-haiku-4-5-20251001` → `haiku-4-5`, the run name's own rule; pipeline models as given. */
export function shortModel(model: string | null | undefined): string {
  if (!model) return '';
  const parts = model.toLowerCase().replace(/^claude-/, '').split('-').filter(Boolean);
  if (parts.length > 1 && /^\d{8}$/.test(parts[parts.length - 1])) parts.pop();
  return parts.join('-') || model;
}

/** `sdk-v1 · haiku-5-5` for a single-session pass; `llm-v8` for a pipeline one. */
export function passLabel(pass: Pick<TestPass, 'version' | 'runtime' | 'model'>): string {
  const version = versionLabel(pass.version) || pass.version || 'no version';
  return pass.runtime === 'agent_sdk' && pass.model ? `${version} · ${shortModel(pass.model)}` : version;
}

/** The agent a step or pass names: version, model and effort together. */
export function agentLabel(item: { version: string; runtime: string; model: string | null; effort: string | null }): string {
  const head = passLabel(item as TestPass);
  return item.runtime === 'agent_sdk' && item.effort ? `${head} · ${item.effort}` : head;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** `2026-09-05` → `5 Sep 2026`, read as a calendar date (no timezone shift). */
export function formatDay(date: string | null | undefined, withYear = true): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date ?? '');
  if (!m) return NO_VALUE;
  const day = `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]}`;
  return withYear ? `${day} ${m[1]}` : day;
}

/** Whole days from `a` to `b`, both `YYYY-MM-DD`; null when either is unreadable. */
export function daysBetween(a: string | null, b: string | null): number | null {
  const pa = /^(\d{4})-(\d{2})-(\d{2})$/.exec(a ?? '');
  const pb = /^(\d{4})-(\d{2})-(\d{2})$/.exec(b ?? '');
  if (!pa || !pb) return null;
  const ua = Date.UTC(Number(pa[1]), Number(pa[2]) - 1, Number(pa[3]));
  const ub = Date.UTC(Number(pb[1]), Number(pb[2]) - 1, Number(pb[3]));
  return Math.round((ub - ua) / 86_400_000);
}

/** Oldest first. A pass with no start time keeps its place after every timed one. */
export function orderPasses<T extends { at_ms: number | null }>(passes: readonly T[]): T[] {
  return passes
    .map((p, i) => ({ p, i }))
    .sort((a, b) => {
      const ta = a.p.at_ms ?? Number.POSITIVE_INFINITY;
      const tb = b.p.at_ms ?? Number.POSITIVE_INFINITY;
      return ta === tb ? a.i - b.i : ta - tb;
    })
    .map(({ p }) => p);
}

/** A pass the chart can draw: complete and scored. */
export function drawable(pass: TestPass): boolean {
  return pass.complete && typeof pass.accuracy === 'number' && Number.isFinite(pass.accuracy);
}

export interface DateBand {
  date: string | null;
  /** First and last slot index (inclusive) on the chart's x axis. */
  from: number;
  to: number;
  /** True when this day does not follow the previous band's day: the axis breaks before it. */
  breakBefore: boolean;
  /** Days since the previous band (null for the first band or an unreadable date). */
  gapDays: number | null;
}

/** Consecutive slots that share a day, and where the axis breaks between them. */
export function dateBands(points: readonly { date: string | null }[]): DateBand[] {
  const bands: DateBand[] = [];
  points.forEach((p, i) => {
    const last = bands[bands.length - 1];
    if (last && last.date === p.date) {
      last.to = i;
      return;
    }
    const gap = last ? daysBetween(last.date, p.date) : null;
    bands.push({
      date: p.date,
      from: i,
      to: i,
      breakBefore: Boolean(last) && (gap === null || gap > 1),
      gapDays: gap,
    });
  });
  return bands;
}

export interface StepSegment {
  /** The step that put this agent in the lead. */
  step: ChampionStep;
  /** Later events that named the same agent (e.g. `universal_champion`), folded in. */
  restated: ChampionStep[];
  accuracy: number;
  /** Slot index where the line reaches this level, and the slot where the next step takes over. */
  from: number;
  to: number;
  /** The drawn pass that measured it, when it is on the chart. */
  pass: TestPass | null;
}

function sameAgent(a: ChampionStep, b: ChampionStep): boolean {
  return a.version === b.version && (a.model ?? '') === (b.model ?? '') && (a.effort ?? '') === (b.effort ?? '');
}

/**
 * The champion of the day as a step line over the drawn passes.
 *
 * A step takes effect at the first drawn pass at or after its `effective_at_ms`. Steps with no
 * accuracy (the legacy promotions made before the gate split existed) are dropped; of the steps in
 * force before the first pass, only the last survives, starting at slot 0. A step naming the agent
 * already leading is folded into that segment as `restated`.
 */
export function championLine(points: readonly TestPass[], steps: readonly ChampionStep[]): StepSegment[] {
  if (!points.length) return [];
  const firstAt = points[0].at_ms ?? Number.NEGATIVE_INFINITY;
  const placed: { step: ChampionStep; restated: ChampionStep[]; from: number }[] = [];
  for (const step of steps) {
    const when = step.effective_at_ms ?? step.at_ms;
    if (typeof step.accuracy !== 'number' || !Number.isFinite(step.accuracy) || when == null) continue;
    let from = 0;
    if (when > firstAt) {
      from = points.findIndex((p) => p.at_ms != null && p.at_ms >= when);
      if (from === -1) continue; // recorded after the last pass: nothing on this axis to step at
    }
    // A later record at the same slot (or before it) supersedes what was placed there.
    while (placed.length && placed[placed.length - 1].from >= from && !sameAgent(placed[placed.length - 1].step, step)) {
      placed.pop();
    }
    const top = placed[placed.length - 1];
    if (top && sameAgent(top.step, step)) {
      top.restated.push(step);
      continue;
    }
    placed.push({ step, restated: [], from });
  }
  return placed.map((p, i) => ({
    step: p.step,
    restated: p.restated,
    accuracy: p.step.accuracy as number,
    from: p.from,
    to: i + 1 < placed.length ? placed[i + 1].from : points.length - 1,
    pass: points.find((pt) => Boolean(p.step.run_name) && pt.run_name === p.step.run_name) ?? null,
  }));
}

export interface TimelineModel {
  /** Every pass, oldest first: the ledger's rows. */
  passes: TestPass[];
  /** The drawn passes, oldest first: the chart's slots. */
  points: TestPass[];
  /** Passes kept off the axis, with their reasons. */
  excluded: TestPass[];
  bands: DateBand[];
  line: StepSegment[];
}

export function timelineModel(passes: readonly TestPass[] | undefined, steps: readonly ChampionStep[] | undefined): TimelineModel {
  const ordered = orderPasses(passes ?? []);
  const points = ordered.filter(drawable);
  return {
    passes: ordered,
    points,
    excluded: ordered.filter((p) => !drawable(p)),
    bands: dateBands(points),
    line: championLine(points, steps ?? []),
  };
}

// ---------------------------------------------------------------------------
// Outcomes: a word and a glyph for each, never colour alone
// ---------------------------------------------------------------------------

export type OutcomeTone = 'acc' | 'ok' | 'no' | 'warn' | 'err';

export const OUTCOMES: Record<PassOutcome, { word: string; tone: OutcomeTone; meaning: string }> = {
  champion: { word: 'champion', tone: 'acc', meaning: 'the champion’s own pass — the line itself' },
  promoted: { word: 'promoted', tone: 'acc', meaning: 'beat the champion of its day and became champion' },
  passed: { word: 'passed, not promoted', tone: 'ok', meaning: 'the gate passed; the registry did not move' },
  beats: { word: 'beats', tone: 'ok', meaning: 'a challenger agent that beat the champion (s19 rule)' },
  matches: { word: 'matches', tone: 'ok', meaning: 'a challenger agent within the 3pp margin (s19 rule)' },
  rejected: { word: 'rejected', tone: 'no', meaning: 'gated against the champion of its day and refused' },
  falls_short: { word: 'falls short', tone: 'err', meaning: 'a challenger agent outside the 3pp margin (s19 rule)' },
  baseline: { word: 'baseline', tone: 'no', meaning: 'only ever a gate baseline' },
  scored: { word: 'scored', tone: 'no', meaning: 'scored, never judged' },
  incomplete: { word: 'incomplete', tone: 'warn', meaning: 'turns unscored — never compared, never drawn' },
};

/** Which mark a pass gets on the chart: the shape carries the outcome as well as the colour. */
export type MarkShape = 'dot' | 'ring' | 'diamond' | 'square';

export function markOf(outcome: PassOutcome): { shape: MarkShape; cls: string } {
  switch (outcome) {
    case 'champion':
    case 'promoted':
      return { shape: 'dot', cls: 'm-lead' };
    case 'beats':
    case 'matches':
    case 'passed':
      return { shape: 'square', cls: 'm-ok' };
    case 'falls_short':
      return { shape: 'diamond', cls: 'm-short' };
    default:
      return { shape: 'ring', cls: 'm-plain' };
  }
}

// ---------------------------------------------------------------------------
// Numbers in words
// ---------------------------------------------------------------------------

/** `2.2923` (pp) → `+2.29pp`. */
export function formatPp(pp: number | null | undefined, digits = 2): string {
  if (pp === null || pp === undefined || !Number.isFinite(pp)) return NO_VALUE;
  const sign = pp > 0 ? '+' : pp < 0 ? '−' : '±';
  return `${sign}${Math.abs(pp).toFixed(digits)}pp`;
}

/** A p-value at the precision it deserves. */
export function formatP(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return NO_VALUE;
  if (p < 0.001) return p.toExponential(1);
  return p.toFixed(3);
}

// ---------------------------------------------------------------------------
// The story in sentences
// ---------------------------------------------------------------------------

export interface StorySummary {
  champion: StepSegment | null;
  /** The day the current champion's pass ran. */
  since: string | null;
  /** Drawn passes after the champion's own pass, oldest first. */
  challengersSince: TestPass[];
  /** How many of them beat it (promoted or beats). */
  nBeat: number;
  nPasses: number;
  nVersions: number;
  nModels: number;
}

export function storySummary(model: TimelineModel): StorySummary {
  const champion = model.line[model.line.length - 1] ?? null;
  const at = champion?.pass ? model.points.indexOf(champion.pass) : champion?.from ?? -1;
  const challengersSince = at >= 0 ? model.points.slice(at + 1) : [];
  return {
    champion,
    since: champion?.pass?.date ?? model.points[champion?.from ?? 0]?.date ?? null,
    challengersSince,
    nBeat: challengersSince.filter((p) => p.outcome === 'promoted' || p.outcome === 'beats').length,
    nPasses: model.passes.length,
    nVersions: new Set(model.passes.map((p) => p.version)).size,
    nModels: new Set(model.passes.map((p) => p.model ?? '')).size,
  };
}

/** One challenger in words: `haiku-5-5 −3.15pp (falls short)`. */
export function challengerPhrase(pass: TestPass): string {
  const who = pass.runtime === 'agent_sdk' ? passLabel(pass) : versionLabel(pass.version);
  return `${who} ${formatPp(pass.vs?.delta_pp)} (${OUTCOMES[pass.outcome].word})`;
}

/** The champion line's history in words: `llm-v2 77.1% → llm-v8 81.7% (+4.58pp) → …`. */
export function stepHistory(line: readonly StepSegment[]): string {
  return line
    .map((s) => {
      const delta = s.pass?.vs?.paired && s.pass.vs.delta_pp != null ? ` (${formatPp(s.pass.vs.delta_pp)} over ${versionLabel(s.pass.vs.version)})` : '';
      return `${agentLabel(s.step)} ${formatPercent(s.accuracy)}${delta}`;
    })
    .join(' → ');
}

// ---------------------------------------------------------------------------
// Ledger rows: test passes, and train passes behind the toggle
// ---------------------------------------------------------------------------

export interface LedgerRow {
  key: string;
  split: 'test' | 'train';
  date: string | null;
  at_ms: number | null;
  label: string;
  run_name: string;
  version: string;
  runtime: 'pipeline' | 'agent_sdk' | 'open_agent';
  model: string | null;
  effort: string | null;
  accuracy: number | null;
  n: number | null;
  outcome: PassOutcome | 'train';
  experiment: string | null;
  cost: number | null;
  championRun: boolean;
  vs: TestPass['vs'];
  reason: string;
  seed: string | null;
  stopAtFirstWrong: boolean;
}

export function ledgerRows(passes: readonly TestPass[], train: readonly TrainPass[] = []): LedgerRow[] {
  const test: LedgerRow[] = passes.map((p) => ({
    key: `test:${p.run_name}`,
    split: 'test',
    date: p.date,
    at_ms: p.at_ms,
    label: passLabel(p),
    run_name: p.run_name,
    version: p.version,
    runtime: p.runtime,
    model: p.model,
    effort: p.effort,
    accuracy: p.accuracy,
    n: p.n_scored ?? p.n_questions,
    outcome: p.outcome,
    experiment: p.label,
    cost: p.cost,
    championRun: p.champion_run,
    vs: p.vs,
    reason: p.reason,
    seed: null,
    stopAtFirstWrong: false,
  }));
  const trainRows: LedgerRow[] = train.map((t) => ({
    key: `train:${t.run_name}`,
    split: 'train',
    date: t.date,
    at_ms: t.at_ms,
    label: passLabel(t),
    run_name: t.run_name,
    version: t.version,
    runtime: t.runtime,
    model: t.model,
    effort: t.effort,
    accuracy: t.accuracy,
    n: t.n_questions,
    outcome: 'train',
    experiment: t.campaign,
    cost: null,
    championRun: false,
    vs: null,
    reason: '',
    seed: t.train_seed,
    stopAtFirstWrong: t.stop_at_first_wrong,
  }));
  return orderPasses([...test, ...trainRows]);
}

export interface LedgerFilters {
  runtime: string;
  model: string;
  outcome: string;
}

export const NO_FILTERS: LedgerFilters = { runtime: '', model: '', outcome: '' };

export function filterRows(rows: readonly LedgerRow[], f: LedgerFilters): LedgerRow[] {
  return rows.filter(
    (r) =>
      (!f.runtime || r.runtime === f.runtime) &&
      (!f.model || shortModel(r.model) === f.model) &&
      (!f.outcome || r.outcome === f.outcome),
  );
}

/**
 * The Overview's one-line record: who leads, at what, since when, and every challenger since with
 * its Δ and verdict — `Champion sdk-v1 · sonnet-5 · high at 90.5% since 5 Sep 2026; challengers
 * since: …`. Replaces the progression's "the last stage" framing, which read a challenger that
 * fell short as the end of the story.
 */
export function recordSentence(summary: StorySummary, gateQuestions: number | null = null): string {
  const c = summary.champion;
  if (!c) return 'No champion has been measured on the gate split yet';
  const on = gateQuestions != null ? ` on ${gateQuestions} unseen questions` : '';
  const head = `Champion ${agentLabel(c.step)} at ${formatPercent(c.accuracy)}${on} since ${formatDay(summary.since)}`;
  const list = summary.challengersSince;
  if (!list.length) return `${head}; nothing has challenged it since`;
  return `${head}; challengers since: ${list.map(challengerPhrase).join(', ')}`;
}
