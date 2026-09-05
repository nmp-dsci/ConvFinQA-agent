import { useQuery } from '@tanstack/react-query';
import { cn } from '@/lib/utils';
import { getCampaigns } from './api';
import type { CampaignExperiment, CampaignSummary, RuntimeArm } from './api';
import {
  AdminPage,
  Caveat,
  EmptyState,
  ErrorNote,
  LoadingRows,
  Panel,
  StatCells,
  TwoUp,
  Verdict,
} from './ui';
import {
  ARMS,
  PANEL_STAGES,
  PAPER_HUMAN,
  formatP,
  formatPp,
  formatWall,
  isBaselineExperiment,
  progression,
  runtimeVerdict,
  sdkCampaign,
  sliceRows,
} from './runtimeStory';
import type { ArmDescription, ProgressionPoint, RuntimeVerdict, SliceRow } from './runtimeStory';
import { NO_VALUE, formatCount, formatPercent, formatUsd } from '../landing/format';

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

// ---------------------------------------------------------------------------
// 1 · The verdict
// ---------------------------------------------------------------------------

export function VerdictBanner({ verdict }: { verdict: RuntimeVerdict }) {
  const adopt = verdict.recommendation === 'adopt-agent-sdk';
  const unrun = verdict.recommendation === 'not-yet-run';
  return (
    <section
      data-testid="runtime-verdict"
      data-recommendation={verdict.recommendation}
      className={cn(
        'min-w-0 rounded-md border bg-panel p-3',
        adopt ? 'border-good-line' : unrun ? 'border-line border-dashed' : 'border-amber-line',
      )}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="mono-caps">recommendation</span>
        <Verdict ok={unrun ? null : adopt}>
          {unrun ? 'not yet run' : adopt ? 'adopt · agent sdk' : 'hold · pipeline'}
        </Verdict>
        {verdict.gateId && (
          <span className="font-mono text-[10px] text-faint">{verdict.gateId}</span>
        )}
      </div>
      <p className="type-lede max-w-[80ch] text-text">{verdict.headline}</p>
      <p className="type-small mt-1 max-w-[92ch] text-muted">{verdict.because}</p>
      <div className="mt-3">
        <StatCells
          columns={3}
          cells={[
            {
              label: 'paired delta (sdk − pipeline)',
              value: formatPp(verdict.deltaPp),
              reason: 'no cross-runtime gate has been run',
              tone: verdict.deltaPp === null ? 'plain' : verdict.deltaPp > 0 ? 'good' : 'bad',
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
      </div>
      <p className="type-meta mt-2 text-faint">
        {verdict.baselineVersion && verdict.candidateVersion
          ? `${verdict.candidateVersion} against ${verdict.baselineVersion}, paired per question on one fixed gate split, one-sided cluster-corrected McNemar at α = 0.05.`
          : 'Both arms are compared on one fixed gate split, paired per question.'}
      </p>
    </section>
  );
}

// ---------------------------------------------------------------------------
// 2 · The two arms
// ---------------------------------------------------------------------------

function Figure({
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
    <div className="flex items-baseline justify-between gap-3">
      <span className="type-small text-muted">{label}</span>
      <span
        className={cn(
          'type-num shrink-0 text-[12px]',
          absent ? 'text-faint' : strong ? 'text-text' : 'text-muted',
        )}
        title={absent ? reason : undefined}
      >
        {absent ? 'not yet run' : value}
      </span>
    </div>
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
    <div
      data-testid={`arm-${desc.key}`}
      data-present={present ? 'true' : 'false'}
      className={cn(
        'min-w-0 rounded-[5px] border bg-panel-2 p-3',
        recommended ? 'border-good-line' : 'border-line',
      )}
    >
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <span className="mono-caps text-faint">{desc.key}</span>
        <span className="type-body font-medium text-text">{desc.title}</span>
        {recommended && <Verdict ok>recommended</Verdict>}
      </div>
      <p className="type-meta mb-2 text-faint">{desc.architecture}</p>
      {!present ? (
        <EmptyState>
          not yet run — this arm has no run on the gate split, so it has no figures
        </EmptyState>
      ) : (
        <>
          <div className="mb-2 flex flex-wrap items-baseline gap-2">
            <span className="type-num text-[27px] leading-none text-text">
              {formatPercent(arm?.accuracy ?? null)}
            </span>
            <span className="font-mono text-[11px] text-muted">{arm?.version ?? NO_VALUE}</span>
          </div>
          <p className="type-meta mb-2 break-all text-faint">{arm?.run_name ?? NO_VALUE}</p>
          <div className="flex flex-col gap-1">
            <Figure
              label="number turns"
              value={formatPercent(arm?.by_turn_type?.number ?? null)}
              reason="the run carries no per-turn-type slice"
              strong
            />
            <Figure
              label="program turns"
              value={formatPercent(arm?.by_turn_type?.program ?? null)}
              reason="the run carries no per-turn-type slice"
              strong
            />
            <Figure
              label="program accuracy (gold program reproduced)"
              value={formatPercent(arm?.program_accuracy ?? null)}
              reason="the run's predictions CSV is not on disk"
            />
          </div>
          <div className="mt-2 grid grid-cols-2 gap-1.5">
            {PANEL_STAGES.map((stage) => {
              const value = arm?.panel?.[stage] ?? null;
              return (
                <div
                  key={stage}
                  data-absent={value === null ? 'true' : 'false'}
                  className="rounded-[4px] border border-line px-2 py-1"
                >
                  <div className="mono-caps text-faint">{stage}</div>
                  <div
                    className={cn('type-num text-[12px]', value === null ? 'text-faint' : 'text-text')}
                  >
                    {value === null ? 'not scored' : formatPercent(value)}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-2 flex flex-col gap-1">
            <Figure
              label="wall clock"
              value={formatWall(arm?.wall ?? null)}
              reason="the run logged no wall time"
            />
            <Figure
              label="cost"
              value={formatUsd(arm?.cost ?? null)}
              reason="this arm's provider spend is not metered into the run"
            />
          </div>
          <p className="type-meta mt-2 text-faint">{desc.aliasNote}</p>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 3 · Where the difference is
// ---------------------------------------------------------------------------

export function SliceTable({ rows }: { rows: SliceRow[] }) {
  if (!rows.length) {
    return <EmptyState>not yet run — the gate carries no per-turn-type split</EmptyState>;
  }
  return (
    <div className="min-w-0 overflow-x-auto">
      <table data-testid="slice-table" className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b border-line">
            {[
              'turn type',
              'n',
              'pipeline',
              'agent sdk',
              'delta',
              'fixed / broken',
              'one-sided clustered p',
              'verdict',
            ].map((h) => (
              <th key={h} className="mono-caps py-1.5 pr-3 font-normal whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.key}
              data-slice={row.key}
              data-effect={row.effect}
              className="border-b border-line last:border-0"
            >
              <td className="py-2 pr-3">
                <div className="type-small text-text">{row.label}</div>
                <div className="type-meta text-faint">{row.what}</div>
              </td>
              <td className="type-num py-2 pr-3 text-[12px] text-muted">
                {formatCount(row.n)}
              </td>
              <td className="type-num py-2 pr-3 text-[12px] text-muted">
                {formatPercent(row.baseline)}
              </td>
              <td className="type-num py-2 pr-3 text-[12px] text-text">
                {formatPercent(row.candidate)}
              </td>
              <td
                className={cn(
                  'type-num py-2 pr-3 text-[12px]',
                  row.effect === 'significant'
                    ? 'text-good'
                    : row.effect === 'no-effect'
                      ? 'text-faint'
                      : 'text-muted',
                )}
              >
                {formatPp(row.deltaPp)}
              </td>
              <td className="type-num py-2 pr-3 text-[12px] text-muted">
                {formatCount(row.fixed)} / {formatCount(row.broken)}
              </td>
              <td className="type-num py-2 pr-3 text-[12px] text-muted">{formatP(row.pValue)}</td>
              <td className="py-2 pr-3">
                <Verdict ok={row.effect === 'significant' ? true : null}>{row.effectLabel}</Verdict>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 4 · The progression
// ---------------------------------------------------------------------------

const STAGE_FILL: Record<ProgressionPoint['key'], string> = {
  pipeline_raw: 'var(--muted)',
  pipeline_optimised: 'var(--amber)',
  sdk_distilled: 'var(--info)',
  sdk_optimised: 'var(--violet)',
};

/**
 * Accuracy across the four stages of the story, as bars on a zero baseline.
 *
 * Hand-drawn SVG, like the champion chart on the Campaigns page and for the same
 * reasons: four columns do not justify a charting dependency, and the CSP on the
 * published surfaces allows only a short list of CDNs. Three properties it holds
 * on purpose:
 *
 *  - Bars start at zero. The interesting range is 77–91%, and a chart cropped to
 *    it would make a nine-point difference look like a fourfold one.
 *  - An absent stage is a dashed empty slot labelled "not yet run", never a bar
 *    of no height, which would read as a measured zero.
 *  - Two reference lines: the paper's human-expert *execution* accuracy, marked
 *    as a published figure, and the incumbent pipeline champion, marked as
 *    measured here. The paper's program-accuracy figure is deliberately not
 *    drawn — it is a different quantity from the one on this axis, and a second
 *    horizontal line in the same units would invite exactly the comparison it
 *    does not support. It is quoted in the caveats instead.
 */
interface RefLine {
  key: string;
  /** A ratio in 0..1, never null — an absent reference is simply not pushed. */
  value: number;
  line: string;
  text: string;
  label: string;
  sub: string;
}

export function ProgressionChart({
  points,
  pipelineBaseline,
}: {
  points: ProgressionPoint[];
  pipelineBaseline: number | null;
}) {
  const drawable = points.filter((p) => p.present && p.accuracy !== null);
  if (!drawable.length) {
    return <EmptyState>not yet run — no stage of the progression has been scored</EmptyState>;
  }

  const w = 940;
  const h = 340;
  const left = 52;
  const right = 178;
  const top = 24;
  const bottom = 78;
  const plotW = w - left - right;
  const plotH = h - top - bottom;
  const slot = plotW / points.length;
  const barW = Math.min(74, slot * 0.52);
  const yOf = (v: number) => top + (1 - v) * plotH;
  const xOf = (i: number) => left + slot * (i + 0.5);

  // Built by pushing rather than filtering: a reference line with no value is
  // not a line at zero, it is no line, and the two are one `if` apart.
  const references: RefLine[] = [];
  if (typeof PAPER_HUMAN.exe === 'number' && Number.isFinite(PAPER_HUMAN.exe)) {
    references.push({
      key: 'human',
      value: PAPER_HUMAN.exe,
      line: 'var(--violet)',
      text: 'var(--violet)',
      label: `human expert ${formatPercent(PAPER_HUMAN.exe)}`,
      sub: 'published, not measured here',
    });
  }
  if (typeof pipelineBaseline === 'number' && Number.isFinite(pipelineBaseline)) {
    references.push({
      key: 'incumbent',
      value: pipelineBaseline,
      line: 'var(--amber-line)',
      text: 'var(--amber)',
      label: `pipeline champion ${formatPercent(pipelineBaseline)}`,
      sub: 'measured on this gate split',
    });
  }

  return (
    <figure className="m-0 min-w-0">
      <svg
        data-testid="progression-chart"
        viewBox={`0 0 ${w} ${h}`}
        className="w-full"
        role="img"
        aria-labelledby="progression-title progression-desc"
      >
        <title id="progression-title">
          Accuracy across four stages: multi-agent raw, multi-agent optimised, single-session SDK
          distilled, and the SDK optimisation attempt.
        </title>
        <desc id="progression-desc">
          {drawable
            .map((p) => `${p.stage}${p.version ? ` (${p.version})` : ''} ${formatPercent(p.accuracy)}`)
            .join('; ')}
          . Reference lines: {references.map((r) => `${r.label} — ${r.sub}`).join('; ')}.
          {points.some((p) => !p.present)
            ? ` Not yet run: ${points
                .filter((p) => !p.present)
                .map((p) => p.stage)
                .join(', ')}.`
            : ''}
        </desc>

        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line
              x1={left}
              y1={yOf(v)}
              x2={w - right}
              y2={yOf(v)}
              stroke="var(--line)"
              strokeWidth={1}
            />
            <text
              x={left - 8}
              y={yOf(v) + 3}
              textAnchor="end"
              fill="var(--faint)"
              className="font-mono text-[10px]"
            >
              {(v * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {references.map((r) => (
          <g key={r.key} data-reference={r.key}>
            <line
              x1={left}
              y1={yOf(r.value)}
              x2={w - right + 6}
              y2={yOf(r.value)}
              stroke={r.line}
              strokeWidth={1.4}
              strokeDasharray="6 4"
            />
            <text
              x={w - right + 12}
              y={yOf(r.value) + 2}
              fill={r.text}
              className="font-mono text-[10px]"
            >
              {r.label}
            </text>
            <text
              x={w - right + 12}
              y={yOf(r.value) + 14}
              fill="var(--faint)"
              className="font-mono text-[9px]"
            >
              {r.sub}
            </text>
          </g>
        ))}

        {/* The stepped line over the bar tops: the progression as one movement,
            broken wherever a stage is absent rather than bridged across it. */}
        {(() => {
          const runs: string[][] = [];
          let current: string[] = [];
          points.forEach((p, i) => {
            if (!p.present || p.accuracy === null) {
              if (current.length > 1) runs.push(current);
              current = [];
              return;
            }
            const y = yOf(p.accuracy);
            current.push(`${xOf(i) - barW / 2},${y}`, `${xOf(i) + barW / 2},${y}`);
            if (i < points.length - 1) {
              const next = points[i + 1];
              if (next.present && next.accuracy !== null) {
                current.push(`${xOf(i + 1) - barW / 2},${y}`);
              }
            }
          });
          if (current.length > 1) runs.push(current);
          return runs.map((pts, i) => (
            <polyline
              key={i}
              points={pts.join(' ')}
              fill="none"
              stroke="var(--text)"
              strokeOpacity={0.35}
              strokeWidth={1.2}
            />
          ));
        })()}

        {points.map((p, i) => {
          const x = xOf(i) - barW / 2;
          const absent = !p.present || p.accuracy === null;
          const rejected = p.promoted === false;
          return (
            <g key={p.key} data-stage={p.key} data-present={absent ? 'false' : 'true'}>
              {absent ? (
                <>
                  <rect
                    x={x}
                    y={top}
                    width={barW}
                    height={plotH}
                    fill="none"
                    stroke="var(--line-2)"
                    strokeDasharray="4 4"
                    rx={2}
                  />
                  <text
                    x={xOf(i)}
                    y={top + plotH / 2}
                    textAnchor="middle"
                    fill="var(--faint)"
                    className="font-mono text-[10px]"
                  >
                    not yet run
                  </text>
                </>
              ) : (
                <>
                  <rect
                    x={x}
                    y={yOf(p.accuracy as number)}
                    width={barW}
                    height={plotH - (yOf(p.accuracy as number) - top)}
                    fill={STAGE_FILL[p.key]}
                    fillOpacity={rejected ? 0.3 : 0.85}
                    stroke={STAGE_FILL[p.key]}
                    strokeDasharray={rejected ? '4 3' : undefined}
                    rx={2}
                  />
                  <text
                    x={xOf(i)}
                    y={yOf(p.accuracy as number) - 7}
                    textAnchor="middle"
                    fill="var(--text)"
                    className="type-num text-[11px]"
                  >
                    {formatPercent(p.accuracy)}
                  </text>
                </>
              )}
              <text
                x={xOf(i)}
                y={h - bottom + 18}
                textAnchor="middle"
                fill="var(--text)"
                className="font-mono text-[10px]"
              >
                {p.stage}
              </text>
              <text
                x={xOf(i)}
                y={h - bottom + 32}
                textAnchor="middle"
                fill="var(--faint)"
                className="font-mono text-[9px]"
              >
                {p.version ?? 'no version'}
                {rejected ? ' · rejected' : ''}
              </text>
            </g>
          );
        })}

        {/* Which runtime each pair belongs to, bracketed under the labels. */}
        {[
          { label: ARMS.pipeline.title, from: 0, to: 1 },
          { label: ARMS.agent_sdk.title, from: 2, to: 3 },
        ].map((group) => {
          const x1 = left + slot * group.from + 6;
          const x2 = left + slot * (group.to + 1) - 6;
          const y = h - bottom + 46;
          return (
            <g key={group.label}>
              <line x1={x1} y1={y} x2={x2} y2={y} stroke="var(--line-2)" strokeWidth={1} />
              <text
                x={(x1 + x2) / 2}
                y={y + 13}
                textAnchor="middle"
                fill="var(--muted)"
                className="font-mono text-[10px]"
              >
                {group.label}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="type-meta mt-2 text-faint">
        Execution accuracy on the fixed gate split, on a zero baseline. The dashed violet line is the
        human-expert execution accuracy reported by {PAPER_HUMAN.citation} on a{' '}
        {PAPER_HUMAN.evaluatedOn || 'sample of the paper’s test set'} — a published figure about a
        different question set, not a measurement of this system. The amber line is the incumbent
        pipeline champion measured here. A hatched bar was gated and rejected: it is what the loop
        tried, not what the runtime does.
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------------------
// 5 · What the loop could do with one prompt
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
    <div className="flex flex-col gap-2" data-testid="sdk-experiments">
      {experiments.map((exp) => {
        const baseline = isBaselineExperiment(exp);
        const target = exp.target_class || exp.target_agent || NO_VALUE;
        const summary = editSummary(exp);
        return (
          <div
            key={`${exp.campaign}-${exp.label}-${exp.candidate_version}`}
            data-experiment={exp.label}
            data-promoted={exp.promoted ? 'true' : 'false'}
            className="rounded-[5px] border border-line bg-panel-2 px-3 py-2.5"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-[12px] text-text">{exp.label}</span>
              <Verdict ok={exp.promoted}>{exp.promoted ? 'promoted' : 'rejected'}</Verdict>
              <span className="rounded-[4px] border border-line-2 px-1.5 py-0.5 font-mono text-[10px] text-muted">
                {baseline ? `${target} · not an optimisation attempt` : `class: ${target}`}
              </span>
              <span className="ml-auto type-num text-[12px] text-muted">
                {formatPp((exp.accuracy_delta ?? null) === null ? null : (exp.accuracy_delta as number) * 100)}{' '}
                · p={formatP(exp.cluster_p_one_sided)}
              </span>
            </div>
            <p className="type-meta mt-1 text-faint">
              <span className="font-mono text-muted">
                {exp.baseline_version} → {exp.candidate_version}
              </span>{' '}
              · {formatPercent(exp.accuracy_baseline)} → {formatPercent(exp.accuracy_candidate)} ·{' '}
              {formatCount(exp.fixed)} fixed / {formatCount(exp.broken)} broken of{' '}
              {formatCount(exp.n_compared)} shared questions
            </p>
            {baseline && (
              <p className="type-meta mt-1 text-faint">
                This row is the cross-runtime read-out itself: the distilled single-session prompt
                against the optimised pipeline, not a change to either.
              </p>
            )}
            {summary && <p className="type-meta mt-1 text-faint">{summary}</p>}
            {exp.summary_of_changes && (
              <p className="type-small mt-1 text-muted">{exp.summary_of_changes}</p>
            )}
          </div>
        );
      })}
      {campaign && (
        <p className="type-meta text-faint">
          {campaign.name}: {formatCount(campaign.n_experiments)} of{' '}
          {campaign.cap === undefined ? 'an unreported cap' : formatCount(campaign.cap)} experiments
          used, {formatCount(campaign.n_promoted)} promoted
          {campaign.cap === undefined
            ? ''
            : `, ${formatCount(campaign.n_remaining)} remaining`}
          . The cap on this arm is deliberately lower than the pipeline&rsquo;s: one prompt is a
          smaller search space, and the point of a cap is that the review happens on schedule rather
          than never.
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------

export default function Runtimes() {
  const query = useQuery({
    // The same key the Campaigns page uses: one payload backs both, and two
    // cache entries of the same story could show two different champions.
    queryKey: ['eval-campaigns'],
    queryFn: () => getCampaigns(),
    staleTime: 60_000,
  });
  const data = query.data;
  const comparison = data?.runtime_comparison ?? null;
  const verdict = runtimeVerdict(comparison, data?.split);
  const rows = sliceRows(comparison?.gate);
  const sdkExperiments = data?.sdk_experiments ?? [];
  const stages = progression(data?.champion_track, comparison, sdkExperiments);
  const campaign = sdkCampaign(data?.sdk_campaigns);
  const sdkArm = comparison?.agent_sdk ?? null;
  const nGateReads =
    (data?.experiments?.length ?? 0) + sdkExperiments.length + (comparison?.gate ? 1 : 0);

  if (query.isLoading) {
    return (
      <AdminPage
        eyebrow="runtime"
        title="Runtimes"
        sub="One Claude Agent SDK session against four prompted agents, on the same gate split."
        testId="runtimes"
      >
        <Panel title="reading the record" endpoint="/eval/campaigns">
          <LoadingRows rows={8} />
        </Panel>
      </AdminPage>
    );
  }

  return (
    <AdminPage
      eyebrow="runtime"
      title="Runtimes"
      sub="Two runtimes answer the same gate split under one evaluator and one significance rule: four prompted DeepSeek agents in a fixed order, and one Claude Agent SDK session whose only tools are the calculator functions. This page carries the recommendation and the evidence for it."
      testId="runtimes"
    >
      {query.error ? (
        <Panel title="verdict" endpoint="/eval/campaigns">
          <ErrorNote error={query.error} />
        </Panel>
      ) : (
        <VerdictBanner verdict={verdict} />
      )}

      <Panel
        title="the two arms on the gate split"
        endpoint="/eval/campaigns"
        note="Both arms are scored by the same evaluator on the same questions; the per-stage panel is derived from gold, with no model calls."
        right={
          <span className="type-small text-faint">
            champion {data?.champion ?? NO_VALUE} · sdk_champion {data?.sdk_champion ?? NO_VALUE}
          </span>
        }
      >
        <TwoUp>
          <ArmCard desc={ARMS.pipeline} arm={comparison?.pipeline} recommended={false} />
          <ArmCard
            desc={ARMS.agent_sdk}
            arm={comparison?.agent_sdk}
            recommended={verdict.recommendation === 'adopt-agent-sdk'}
          />
        </TwoUp>
        <Caveat>
          The two arms differ in model <em>and</em> in architecture at once. Nothing here separates
          &ldquo;one session beats four&rdquo; from &ldquo;this model beats that one&rdquo; — see the
          caveats below.
        </Caveat>
      </Panel>

      <Panel
        title="where the difference is"
        endpoint="/eval/campaigns"
        note="The dataset splits every turn into a lookup (number) and a computation (program). The aggregate delta is an average over the two and describes neither."
      >
        <SliceTable rows={rows} />
        {rows.some((r) => r.effect === 'no-effect') && (
          <Caveat>
            Both arms have saturated the lookup, so a slice marked <strong>no effect</strong> is
            exactly that: the same accuracy either side, flips in both directions cancelling. The
            entire aggregate gain sits in the program turns.
          </Caveat>
        )}
      </Panel>

      <Panel
        title="the progression"
        endpoint="/eval/campaigns"
        note="Four stages, in the order they happened. The pipeline points are the champion track's first and last; the SDK points are the distilled prompt and the loop's attempt on it."
      >
        <ProgressionChart points={stages} pipelineBaseline={comparison?.pipeline?.accuracy ?? null} />
      </Panel>

      <Panel
        title="the finding that matters most"
        endpoint="/eval/campaigns"
        note="Distilling four tuned prompts into one produced this arm's best result — and the optimisation loop could not improve on it."
      >
        <p className="type-small mb-3 max-w-[92ch] text-muted">
          The single-session arm&rsquo;s best prompt is the <em>distilled</em> one: the four tuned
          pipeline prompts written into a single system prompt, run with no further optimisation. The
          loop then did to that prompt exactly what it does to a pipeline prompt — diagnose the
          first-wrong cases, file each under a failure class, rewrite the highest-ranked one, gate it
          — and its one completed experiment was rejected by the same paired test that promoted the
          distillation. That is the result, stated as it happened: the optimiser could not beat the
          distillation, on this split, within this campaign&rsquo;s cap.
        </p>
        <SdkExperimentList experiments={sdkExperiments} campaign={campaign} />
      </Panel>

      <TwoUp>
        <Panel
          title="what made the comparison possible"
          endpoint="/eval/campaigns"
          note="why a runtime swap is a measurable claim here"
        >
          <p className="type-small max-w-[80ch] text-muted">
            MLflow and the eval loop give both runtimes one fixed gate split, one evaluator, one
            significance rule and one append-only record. Because the split, the scoring and the
            promotion test were already fixed before this arm existed, swapping the runtime changed
            exactly one thing and the difference could be measured rather than argued about. Every
            figure on this page is read back out of that record — the same record the published
            write-up is built from — which is why the two cannot disagree.
          </p>
          <p className="type-meta mt-2 text-faint">
            {data?.rule ? `Promotion rule: ${data.rule}.` : ''}{' '}
            {data?.generated_at ? `Record generated ${data.generated_at}.` : ''}
          </p>
        </Panel>

        <Panel
          title="caveats"
          endpoint="/eval/campaigns"
          note="not footnotes — read these with the number above"
        >
          <ul className="flex flex-col gap-2">
            <li className="rounded-[4px] border border-amber-line px-2.5 py-2">
              <div className="mono-caps text-amber">contamination cannot be excluded</div>
              <p className="type-small mt-1 text-muted">
                The single-session arm scores {formatPercent(sdkArm?.accuracy ?? null)} execution
                accuracy, above the {formatPercent(PAPER_HUMAN.exe)} human-expert figure{' '}
                {PAPER_HUMAN.citation} reports — on a dataset that has been public since 2022. A
                model that has seen this corpus in training would look exactly like this. The
                program accuracy on the same run is{' '}
                {formatPercent(sdkArm?.program_accuracy ?? null)}, against the paper&rsquo;s{' '}
                {formatPercent(PAPER_HUMAN.prog)} for a human: it is reaching the right numbers
                without reproducing the gold programs, so this is not a claim of human-level
                reasoning.
              </p>
            </li>
            <li className="rounded-[4px] border border-amber-line px-2.5 py-2">
              <div className="mono-caps text-amber">model and architecture moved together</div>
              <p className="type-small mt-1 text-muted">
                The pipeline arm is {ARMS.pipeline.title}; the SDK arm is {ARMS.agent_sdk.title}. Both
                variables changed in the same step and the confound was never isolated — no
                single-session run on the pipeline&rsquo;s model, and no four-agent run on the
                SDK&rsquo;s. The measured delta is real; the attribution of it to &ldquo;one session
                is better than four&rdquo; is not established.
              </p>
            </li>
            <li className="rounded-[4px] border border-amber-line px-2.5 py-2">
              <div className="mono-caps text-amber">the gate split has been read many times</div>
              <p className="type-small mt-1 text-muted">
                The same {formatCount(verdict.gateQuestions)} questions across{' '}
                {formatCount(verdict.gateReports)} conversations have now been read by at least{' '}
                {formatCount(nGateReads)} gated comparisons across both arms. Every read costs some of its
                unseen-ness, so the interval on this delta is optimistic in a way no single test
                reports. The sealed holdout — opened once, on the record — is the only split left
                that has not been read.
              </p>
            </li>
          </ul>
        </Panel>
      </TwoUp>
    </AdminPage>
  );
}
