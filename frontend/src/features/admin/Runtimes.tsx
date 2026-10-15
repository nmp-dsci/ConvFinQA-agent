import type { ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Figure, Loading, PageHead, Status } from '@/kit/ui';
import type { StatusTone } from '@/kit/ui';
import { getCampaigns } from './api';
import type { CampaignExperiment, CampaignSummary, RuntimeArm } from './api';
import { EmptyState, ErrorNote, TwoUp } from './ui';
import {
  ARMS,
  PANEL_STAGES,
  PAPER_HUMAN,
  formatP,
  formatPp,
  formatWall,
  isBaselineExperiment,
  modelRows,
  progression,
  runtimeVerdict,
  sdkCampaign,
  sliceRows,
} from './runtimeStory';
import type { ArmDescription, ModelRow, RuntimeVerdict, SliceRow } from './runtimeStory';
import { NO_VALUE, formatCount, formatPercent, formatUsd } from '../landing/format';
import { ProgressionChart } from './ProgressionChart';
import { versionLabel } from './lib';
import './runtimes.css';

// The judge moved to Scoring (slot 3); re-exported so `judgeStory.test.tsx` keeps its import.
export { JudgeTable, JudgeVerdictBanner } from './judge';

/**
 * Runtimes: one Claude Agent SDK session against four prompted agents.
 *
 * This page exists to carry a recommendation, and a recommendation is the one
 * kind of page that has to be auditable line by line — so the claim, the
 * evidence, the slice it lives in, the progression that led to it and the
 * reasons to doubt it are all on the same screen, and every figure on it is read
 * from `/eval/campaigns` (i.e. from `evaluation/story.json`, i.e. from the
 * tracking store and the three append-only ledgers). The one exception is the
 * paper's human-expert row, which is labelled as published wherever it appears
 * and is read from `features/system/paper.ts` rather than typed here.
 *
 * The shaping — what the recommendation is, which slice moved, what the four
 * stages of the progression are — lives in `runtimeStory.ts` and is unit-tested.
 * (Named that way, not `runtimes.ts`: this file is `Runtimes.tsx`, and on a
 * case-insensitive filesystem the two resolve to each other — which fails as an
 * undefined component at render time, not as a missing module.)
 * What is here is rendering.
 */

const SOURCE = '/eval/campaigns ← evaluation/story.json';

// ---------------------------------------------------------------------------
// Small figures with their reason when absent
// ---------------------------------------------------------------------------

interface Cell {
  label: string;
  value: string;
  /** Printed whenever `value` is an em dash. Never a bare blank. */
  reason: string;
  tone?: 'ok' | 'fail';
}

function Cells({ cells }: { cells: Cell[] }) {
  return (
    <div className="stats rt-cells">
      {cells.map((cell) => {
        const absent = cell.value === NO_VALUE;
        const tone = absent ? 'absent' : cell.tone === 'ok' ? 'good' : cell.tone === 'fail' ? 'bad' : '';
        return (
          <div key={cell.label} className="stat" data-absent={absent ? 'true' : 'false'}>
            <div className={`n ${tone}`}>{cell.value}</div>
            <span className="label">{cell.label}</span>
            {absent && <div className="why">{cell.reason}</div>}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 0 · The verdict
// ---------------------------------------------------------------------------

export function VerdictBanner({ verdict }: { verdict: RuntimeVerdict }) {
  const adopt = verdict.recommendation === 'adopt-agent-sdk';
  const unrun = verdict.recommendation === 'not-yet-run';
  const tone: StatusTone = unrun ? 'no' : adopt ? 'ok' : 'err';
  return (
    <section
      data-testid="runtime-verdict"
      data-recommendation={verdict.recommendation}
      data-tone={tone}
      className="card rt-verdict"
    >
      <div className="rt-verdict-head">
        <span className="label">recommendation</span>
        <Status tone={tone}>
          {unrun ? 'not yet run' : adopt ? 'adopt · agent sdk' : 'hold · pipeline'}
        </Status>
        {verdict.gateId && <span className="mono muted wrap-any">{verdict.gateId}</span>}
      </div>
      <p className="lead rt-headline">{verdict.headline}</p>
      <p className="small muted">{verdict.because}</p>
      <Cells
        cells={[
          {
            label: 'paired delta (sdk − pipeline)',
            value: formatPp(verdict.deltaPp),
            reason: 'no cross-runtime gate has been run',
            tone: verdict.deltaPp === null ? undefined : verdict.deltaPp > 0 ? 'ok' : 'fail',
          },
          {
            label: 'one-sided clustered p',
            value: formatP(verdict.pValue),
            reason: 'no gate to test',
          },
          {
            label: '95% CI on the delta',
            value:
              verdict.ciLo === null || verdict.ciHi === null
                ? NO_VALUE
                : `${formatPp(verdict.ciLo * 100, 1)} … ${formatPp(verdict.ciHi * 100, 1)}`,
            reason: 'no gate to bootstrap',
          },
          {
            label: 'fixed / broken',
            value:
              verdict.fixed === null && verdict.broken === null
                ? NO_VALUE
                : `${formatCount(verdict.fixed ?? 0)} / ${formatCount(verdict.broken ?? 0)}`,
            reason: 'no flips recorded',
          },
          {
            label: 'gate split',
            value:
              verdict.gateQuestions === null
                ? NO_VALUE
                : `${formatCount(verdict.gateQuestions)} q / ${formatCount(
                    verdict.gateReports ?? 0,
                  )} conv`,
            reason: 'the split manifest is not in this payload',
          },
        ]}
      />
      <p className="small muted">
        {verdict.baselineVersion && verdict.candidateVersion
          ? `${verdict.candidateVersion} against ${verdict.baselineVersion}, paired per question on one fixed gate split, one-sided cluster-corrected McNemar at α = 0.05.`
          : 'Both arms are compared on one fixed gate split, paired per question.'}
      </p>
    </section>
  );
}

// ---------------------------------------------------------------------------
// 1 · The two arms
// ---------------------------------------------------------------------------

function ArmFigure({
  label,
  value,
  reason,
  strong = false,
}: {
  label: string;
  value: string;
  reason?: string;
  strong?: boolean;
}) {
  const absent = value === NO_VALUE;
  return (
    <>
      <dt>{label}</dt>
      <dd className={absent ? 'num rt-absent' : strong ? 'num rt-strong' : 'num'}>
        {absent ? 'not yet run' : value}
        {absent && reason && <span className="rt-why">{reason}</span>}
      </dd>
    </>
  );
}

export function ArmCard({
  desc,
  arm,
  recommended,
}: {
  desc: ArmDescription;
  arm: RuntimeArm | null | undefined;
  recommended: boolean;
}) {
  const present = Boolean(arm && (arm.accuracy !== null || arm.version));
  return (
    <article
      data-testid={`arm-${desc.key}`}
      data-present={present ? 'true' : 'false'}
      className={recommended ? 'card hi rt-arm' : 'card rt-arm'}
    >
      <div className="rt-arm-head">
        <span className="label">{desc.key}</span>
        {recommended && <span className="tag acc">recommended</span>}
      </div>
      <h3>{desc.title}</h3>
      <p className="small muted">{desc.architecture}</p>
      {!present ? (
        <EmptyState>
          not yet run — this arm has no run on the gate split, so it has no figures
        </EmptyState>
      ) : (
        <>
          <div className="rt-arm-acc">
            <span className="num rt-big">{formatPercent(arm?.accuracy ?? null)}</span>
            <span className="small muted">
              execution accuracy ·{' '}
              <span className="mono">{arm?.version ? versionLabel(arm.version) : NO_VALUE}</span>
            </span>
          </div>
          <p className="mono muted wrap-any">{arm?.run_name ?? NO_VALUE}</p>
          <dl className="kv">
            <ArmFigure
              label="number turns"
              value={formatPercent(arm?.by_turn_type?.number ?? null)}
              reason="the run carries no per-turn-type slice"
              strong
            />
            <ArmFigure
              label="program turns"
              value={formatPercent(arm?.by_turn_type?.program ?? null)}
              reason="the run carries no per-turn-type slice"
              strong
            />
            <ArmFigure
              label="program accuracy (gold program reproduced)"
              value={formatPercent(arm?.program_accuracy ?? null)}
              reason="the run's predictions CSV is not on disk"
            />
          </dl>
          <div className="stats rt-stages">
            {PANEL_STAGES.map((stage) => {
              const value = arm?.panel?.[stage] ?? null;
              return (
                <div key={stage} data-absent={value === null ? 'true' : 'false'} className="stat">
                  <div className={value === null ? 'n absent' : 'n'}>
                    {value === null ? 'not scored' : formatPercent(value)}
                  </div>
                  <span className="label">{stage}</span>
                </div>
              );
            })}
          </div>
          <dl className="kv">
            <ArmFigure
              label="wall clock"
              value={formatWall(arm?.wall ?? null)}
              reason="the run logged no wall time"
            />
            <ArmFigure
              label="cost"
              value={formatUsd(arm?.cost ?? null)}
              reason="this arm's provider spend is not metered into the run"
            />
          </dl>
          <p className="small muted">{desc.aliasNote}</p>
        </>
      )}
    </article>
  );
}

// ---------------------------------------------------------------------------
// 3 · The model swap: one prompt, several models
// ---------------------------------------------------------------------------

function Heads({ cols }: { cols: Array<[string, boolean]> }) {
  return (
    <thead>
      <tr>
        {cols.map(([h, num]) => (
          <th key={h} scope="col" className={num ? 'num' : undefined}>
            {h}
          </th>
        ))}
      </tr>
    </thead>
  );
}

function effectTone(effect: string): StatusTone {
  return effect === 'better' || effect === 'significant' ? 'ok' : effect === 'worse' ? 'err' : 'no';
}

function deltaClass(effect: string): string {
  return effect === 'better' || effect === 'significant'
    ? 'num v-ok'
    : effect === 'worse'
      ? 'num v-fail'
      : 'num';
}

/** The s19 rule's three outcomes, as the table says them. */
const VERDICT_LABEL: Record<'beats' | 'matches' | 'falls_short', string> = {
  beats: 'beats the champion',
  matches: 'matches the champion',
  falls_short: 'falls short',
};

export function ModelSwapTable({ rows }: { rows: ModelRow[] }) {
  if (!rows.length) {
    return (
      <EmptyState>
        not yet run — the sdk champion has been scored on one model only, so there is nothing to
        compare
      </EmptyState>
    );
  }
  return (
    <div className="tw">
      <table data-testid="model-swap-table" className="rt-table">
        <Heads
          cols={[
            ['agent (model · effort)', false],
            ['accuracy', true],
            ['number turns', true],
            ['program turns', true],
            ['program acc.', true],
            ['cost', true],
            ['wall', true],
            ['delta vs champion', true],
            ['clustered p (direction of delta)', true],
            ['train: right to the end', true],
            ['verdict', false],
          ]}
        />
        <tbody>
          {rows.map((row) => (
            <tr
              key={`${row.model}|${row.effort ?? ''}`}
              data-model={row.shortName}
              data-effect={row.effect}
              data-verdict={row.verdict ?? undefined}
            >
              <td className="sub">
                <span className="mono">
                  {row.shortName}
                  {row.effort ? ` · ${row.effort}` : ''}
                </span>
                <span className="path wrap-any">{row.model}</span>
              </td>
              <td className="num rt-strong">{formatPercent(row.accuracy)}</td>
              <td className="num">{formatPercent(row.number)}</td>
              <td className="num">{formatPercent(row.program)}</td>
              <td className="num">{formatPercent(row.programAccuracy)}</td>
              <td className="num">{formatUsd(row.cost)}</td>
              <td className="num">{formatWall(row.wall)}</td>
              <td className={deltaClass(row.effect)}>
                {row.isReference ? (
                  'champion'
                ) : (
                  <>
                    {formatPp(row.deltaPp)}
                    {row.deltaPp !== null && (
                      <span className="path">
                        CI [{formatPp(row.ciLo === null ? null : row.ciLo * 100, 1)}, {formatPp(row.ciHi === null ? null : row.ciHi * 100, 1)}]
                      </span>
                    )}
                  </>
                )}
              </td>
              <td className="num">
                {row.isReference ? NO_VALUE : formatP(row.pValue)}
                {row.deltaPp !== null && (
                  <span className="path">
                    {formatCount(row.fixed)} / {formatCount(row.broken)} flips
                  </span>
                )}
              </td>
              <td className="num">
                {row.trainConversations === null ? (
                  NO_VALUE
                ) : (
                  <>
                    {formatCount(row.trainBaselineFull)} → {formatCount(row.trainCandidateFull)} /{' '}
                    {formatCount(row.trainConversations)}
                    <span className="path">p worse {formatP(row.trainPWorse)}</span>
                  </>
                )}
              </td>
              <td>
                {row.verdict ? (
                  <Status
                    tone={row.verdict === 'falls_short' ? effectTone('worse') : effectTone('better')}
                  >
                    {VERDICT_LABEL[row.verdict]}
                  </Status>
                ) : (
                  <Status tone={effectTone(row.effect)}>{row.effectLabel}</Status>
                )}
                {row.verdict && <span className="path">{row.effectLabel}</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 2 · Where the difference is
// ---------------------------------------------------------------------------

export function SliceTable({ rows }: { rows: SliceRow[] }) {
  if (!rows.length) {
    return <EmptyState>not yet run — the gate carries no per-turn-type split</EmptyState>;
  }
  return (
    <div className="tw">
      <table data-testid="slice-table" className="rt-table">
        <Heads
          cols={[
            ['turn type', false],
            ['n', true],
            ['pipeline', true],
            ['agent sdk', true],
            ['delta', true],
            ['fixed / broken', true],
            ['one-sided clustered p', true],
            ['verdict', false],
          ]}
        />
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} data-slice={row.key} data-effect={row.effect}>
              <td className="sub rt-wrap">
                {row.label}
                <span className="path rt-sans">{row.what}</span>
              </td>
              <td className="num">{formatCount(row.n)}</td>
              <td className="num">{formatPercent(row.baseline)}</td>
              <td className="num rt-strong">{formatPercent(row.candidate)}</td>
              <td className={deltaClass(row.effect)}>{formatPp(row.deltaPp)}</td>
              <td className="num">
                {formatCount(row.fixed)} / {formatCount(row.broken)}
              </td>
              <td className="num">{formatP(row.pValue)}</td>
              <td>
                <Status tone={effectTone(row.effect)}>{row.effectLabel}</Status>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 4 · What the loop could do with one prompt
// ---------------------------------------------------------------------------

function editSummary(exp: CampaignExperiment): string {
  const edits = exp.edits ?? [];
  if (!edits.length) return '';
  const classes = edits
    .map((e) => String(e.failure_class ?? e.change_kind ?? ''))
    .filter(Boolean);
  return classes.length ? `${edits.length} tagged edit(s): ${classes.join(', ')}` : '';
}

export function SdkExperimentList({
  experiments,
  campaign,
}: {
  experiments: CampaignExperiment[];
  campaign: CampaignSummary | null;
}) {
  if (!experiments.length) {
    return <EmptyState>not yet run — this arm has gated no experiment</EmptyState>;
  }
  return (
    <div data-testid="sdk-experiments">
      <ol className="rt-exps">
        {experiments.map((exp) => {
          const baseline = isBaselineExperiment(exp);
          const target = exp.target_class || exp.target_agent || NO_VALUE;
          const summary = editSummary(exp);
          return (
            <li
              key={`${exp.campaign}-${exp.label}-${exp.candidate_version}`}
              data-experiment={exp.label}
              data-promoted={exp.promoted ? 'true' : 'false'}
              className="card rt-exp"
            >
              <div className="rt-exp-head">
                <span className="mono rt-strong">{exp.label}</span>
                <Status tone={exp.promoted ? 'ok' : 'err'}>{exp.promoted ? 'promoted' : 'rejected'}</Status>
                <span className="tag">
                  {baseline ? `${target} · not an optimisation attempt` : `class: ${target}`}
                </span>
                <span className="num rt-exp-delta">
                  {formatPp(
                    (exp.accuracy_delta ?? null) === null ? null : (exp.accuracy_delta as number) * 100,
                  )}{' '}
                  · p={formatP(exp.cluster_p_one_sided)}
                </span>
              </div>
              <p className="small muted">
                <span className="mono">
                  {exp.baseline_version} → {exp.candidate_version}
                </span>{' '}
                · {formatPercent(exp.accuracy_baseline)} → {formatPercent(exp.accuracy_candidate)} ·{' '}
                {formatCount(exp.fixed)} fixed / {formatCount(exp.broken)} broken of{' '}
                {formatCount(exp.n_compared)} shared questions
              </p>
              {baseline && (
                <p className="small muted">
                  This row is the cross-runtime read-out itself: the distilled single-session prompt
                  against the optimised pipeline, not a change to either.
                </p>
              )}
              {summary && <p className="small muted">{summary}</p>}
              {exp.summary_of_changes && <p className="small">{exp.summary_of_changes}</p>}
            </li>
          );
        })}
      </ol>
      {campaign && (
        <p className="small muted">
          {campaign.name}: {formatCount(campaign.n_experiments)} of{' '}
          {campaign.cap === undefined ? 'an unreported cap' : formatCount(campaign.cap)} experiments
          used, {formatCount(campaign.n_promoted)} promoted
          {campaign.cap === undefined ? '' : `, ${formatCount(campaign.n_remaining)} remaining`}. The
          cap on this arm is deliberately lower than the pipeline&rsquo;s: one prompt is a smaller
          search space, and the point of a cap is that the review happens on schedule rather than
          never.
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------

function Sec({
  n,
  label,
  claim,
  lead,
  children,
  id,
}: {
  n: number;
  label: string;
  claim: ReactNode;
  lead?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section className="rt-sec" id={id}>
      <h2>
        {n} · {label} — {claim}
      </h2>
      {lead && <p>{lead}</p>}
      {children}
    </section>
  );
}

function pageTitle(verdict: RuntimeVerdict): ReactNode {
  if (verdict.recommendation === 'adopt-agent-sdk') {
    return (
      <>
        One Claude session beats four prompted agents by <em>{formatPp(verdict.deltaPp)}</em> on
        the same gate questions
      </>
    );
  }
  if (verdict.recommendation === 'stay-on-pipeline') {
    return (
      <>
        The four-agent pipeline <em>holds</em>: one Claude session did not clear the gate
      </>
    );
  }
  return (
    <>
      Two runtimes, one evaluator, and <em>no</em> cross-runtime gate run yet
    </>
  );
}

const EYEBROW = 'optimise · runtimes · two runtimes, one evaluator';
const LEAD =
  'Four prompted DeepSeek agents in a fixed order, and one Claude Agent SDK session whose only tools are the calculator functions, answer the same gate split under one evaluator and one significance rule. This page carries the recommendation and the evidence for it.';

export default function Runtimes() {
  const query = useQuery({
    // The same key the Experiments and Scoring pages use: one payload backs all three, and two
    // cache entries of the same story could show two different champions.
    queryKey: ['eval-campaigns'],
    queryFn: () => getCampaigns(),
    staleTime: 60_000,
  });
  const data = query.data;
  const comparison = data?.runtime_comparison ?? null;
  const verdict = runtimeVerdict(comparison, data?.split);
  const rows = sliceRows(comparison?.gate);
  const swapRows = modelRows(data?.sdk_model_comparison);
  const sdkExperiments = data?.sdk_experiments ?? [];
  const stages = progression(
    data?.champion_track,
    comparison,
    sdkExperiments,
    data?.sdk_model_comparison,
  );
  const campaign = sdkCampaign(data?.sdk_campaigns);
  const sdkArm = comparison?.agent_sdk ?? null;
  const nGateReads =
    (data?.experiments?.length ?? 0) + sdkExperiments.length + (comparison?.gate ? 1 : 0);

  if (query.isLoading) {
    return (
      <div data-testid="runtimes">
        <PageHead
          eyebrow={EYEBROW}
          title={
            <>
              Two runtimes, <em>one</em> evaluator
            </>
          }
          lead={LEAD}
        />
        <Loading what="the campaign record" />
      </div>
    );
  }

  const top = rows.find((r) => r.effect === 'significant') ?? null;
  const attempts = sdkExperiments.filter((e) => !isBaselineExperiment(e));
  const promotedAttempts = attempts.filter((e) => e.promoted).length;
  const loopClaim =
    attempts.length === 0
      ? 'the loop has not yet attempted the distilled prompt'
      : promotedAttempts === 0
        ? `the loop could not beat the distilled prompt (${attempts.length} of ${attempts.length} ${
            attempts.length === 1 ? 'attempt' : 'attempts'
          } rejected)`
        : `${promotedAttempts} of ${attempts.length} loop attempts beat the distilled prompt`;

  return (
    <div data-testid="runtimes">
      <PageHead eyebrow={EYEBROW} title={pageTitle(verdict)} lead={LEAD} />

      {query.error ? <ErrorNote error={query.error} /> : <VerdictBanner verdict={verdict} />}

      <Sec
        n={1}
        label="The two arms"
        claim={
          verdict.gateQuestions === null
            ? 'one evaluator, the same questions'
            : `one evaluator, the same ${formatCount(verdict.gateQuestions)} questions`
        }
        lead="Both arms are scored by the same evaluator on the same questions; the per-stage panel is derived from gold, with no model calls."
      >
        <p className="small muted">
          champion{' '}
          <span className="mono">{data?.champion ? versionLabel(data.champion) : NO_VALUE}</span>
          {data?.champion_agent && (
            <>
              {' '}
              on <span className="mono">{data.champion_agent.model}</span> · effort{' '}
              {data.champion_agent.effort}
            </>
          )}{' '}
          · pipeline champion{' '}
          <span className="mono">
            {data?.pipeline_champion ? versionLabel(data.pipeline_champion) : NO_VALUE}
          </span>
        </p>
        <TwoUp className="rt-arms">
          <ArmCard desc={ARMS.pipeline} arm={comparison?.pipeline} recommended={false} />
          <ArmCard
            desc={ARMS.agent_sdk}
            arm={comparison?.agent_sdk}
            recommended={verdict.recommendation === 'adopt-agent-sdk'}
          />
        </TwoUp>
        <p className="callout">
          The two arms differ in model <em>and</em> in architecture at once. Nothing here separates
          &ldquo;one session beats four&rdquo; from &ldquo;this model beats that one&rdquo; — see the
          caveats below.
        </p>
      </Sec>

      <Sec
        n={2}
        label="Where the difference is"
        claim={
          !rows.length
            ? 'the gate carries no per-turn-type split'
            : top
              ? `${top.label} carry ${formatPp(top.deltaPp)}`
              : 'no slice moved significantly'
        }
        lead="The dataset splits every turn into a lookup (number) and a computation (program). The aggregate delta is an average over the two and describes neither."
      >
        <SliceTable rows={rows} />
        {rows.some((r) => r.effect === 'no-effect') && (
          <p className="callout">
            Both arms have saturated the lookup, so a slice marked <b>no effect</b> is exactly that:
            the same accuracy either side, flips in both directions cancelling. The entire aggregate
            gain sits in the program turns.
          </p>
        )}
      </Sec>

      <Sec
        n={3}
        label="The progression"
        claim="every scored version, on the same held-back questions, in the order it happened"
      >
        <Figure
          title="fig 1 · execution accuracy on the unseen gate split, by stage"
          caption="Each stage in the order the work happened, on one zero-based axis. The pipeline points are the champion track's first and last."
          source={SOURCE}
        >
          <ProgressionChart points={stages} pipelineBaseline={comparison?.pipeline?.accuracy ?? null} />
        </Figure>
      </Sec>

      <Sec
        n={4}
        label="The model swap"
        claim={
          swapRows.length ? `one prompt, ${swapRows.length} agents` : 'one prompt, one agent so far'
        }
        lead="The same champion prompt, the same calculator tools, the same questions and the same scoring — only the agent (model, effort) differs. Each challenger is paired against the champion's own committed runs: on the gate split question by question, on its two early-stopped train draws conversation by conversation. The verdict is the s19 rule — beats is the house gate, matches is non-inferiority within the margin — and only a passing verdict can move the champion's model. The p is one-sided in the direction of the observed delta."
      >
        <ModelSwapTable rows={swapRows} />
        {swapRows.length > 0 && (
          <p className="callout">
            The delta is the model&rsquo;s contribution <em>at this prompt</em>, which was distilled
            from four DeepSeek agents and never optimised for either model — so neither figure is
            that model&rsquo;s ceiling. It answers &ldquo;how much of the SDK arm is the model&rdquo;
            for the session runtime only; the pipeline was never run on either of these models.
          </p>
        )}
      </Sec>

      <Sec
        n={5}
        label="The finding that matters most"
        claim={loopClaim}
        lead="Distilling four tuned prompts into one produced this arm's best result. The loop then did to that prompt exactly what it does to a pipeline prompt — diagnose the first-wrong cases, file each under a failure class, rewrite the highest-ranked one, gate it."
      >
        <SdkExperimentList experiments={sdkExperiments} campaign={campaign} />
      </Sec>

      <Sec n={6} label="Caveats" claim="three reasons to doubt the number above">
        <TwoUp>
          <div className="rt-caveats">
            <div className="callout">
              <span className="ttl">contamination cannot be excluded</span>
              <p>
                The single-session arm scores {formatPercent(sdkArm?.accuracy ?? null)} execution
                accuracy, above the {formatPercent(PAPER_HUMAN.exe)} human-expert figure{' '}
                {PAPER_HUMAN.citation} reports — on a dataset public since 2022. A model that has
                seen this corpus in training would look exactly like this. Program accuracy on the
                same run is {formatPercent(sdkArm?.program_accuracy ?? null)}, against the
                paper&rsquo;s {formatPercent(PAPER_HUMAN.prog)} for a human: it reaches the right
                numbers without reproducing the gold programs, so this is not a claim of human-level
                reasoning.
              </p>
            </div>
            <div className="callout">
              <span className="ttl">model and architecture moved together</span>
              <p>
                The pipeline arm is {ARMS.pipeline.title}; the SDK arm is {ARMS.agent_sdk.title}.
                Both variables changed in the same step.{' '}
                {swapRows.length > 0
                  ? 'The model swap above measures the session-side half — the same prompt on a second model — but no four-agent run on a Claude model exists, so the other half stays open.'
                  : 'The confound was never isolated — no single-session run on the pipeline’s model, and no four-agent run on the SDK’s.'}{' '}
                The measured delta is real; its attribution to &ldquo;one session is better than
                four&rdquo; is not established.
              </p>
            </div>
            <div className="callout">
              <span className="ttl">the gate split has been read many times</span>
              <p>
                The same {formatCount(verdict.gateQuestions)} questions across{' '}
                {formatCount(verdict.gateReports)} conversations have been read by at least{' '}
                {formatCount(nGateReads)} gated comparisons across both arms. Every read costs some
                of its unseen-ness, so the interval on this delta is optimistic in a way no single
                test reports. The sealed holdout — opened once, on the record — is the only split
                left unread.
              </p>
            </div>
          </div>
          <div className="card">
            <h3>What made the comparison possible</h3>
            <p className="small">
              MLflow and the eval loop give both runtimes one fixed gate split, one evaluator, one
              significance rule and one append-only record. Because the split, the scoring and the
              promotion test were fixed before this arm existed, swapping the runtime changed
              exactly one thing, and the difference could be measured rather than argued about.
              Every figure on this page is read back out of that record — the same record the
              published write-up is built from — so the two cannot disagree.
            </p>
            <p className="small muted">
              {data?.rule ? `Promotion rule: ${data.rule}.` : ''}{' '}
              {data?.generated_at ? `Record generated ${data.generated_at}.` : ''}
            </p>
          </div>
        </TwoUp>
      </Sec>

      <p className="rt-next">
        <Link to="/scoring">The confidence judge that was tried on this runtime →</Link>
      </p>
    </div>
  );
}
