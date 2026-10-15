import { Fragment, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import { Link } from 'react-router-dom';
import './optimise.css';
import { Figure, Status } from '@/kit/ui';
import type { StatusTone } from '@/kit/ui';
import { fmtPp } from '@/kit/format';
import { formatPercent } from '../landing/format';
import { getCampaigns } from './api';
import type { CampaignExperiment, ChampionPoint } from './api';
import { CHAMPION_ROW, InstrumentTable } from './InstrumentTable';
import { bundleLine, clip, formatCount, formatEpochMs, formatRunDuration, formatStamp, versionLabel } from './lib';
import {
  AdminPage,
  Caveat,
  EmptyState,
  ErrorNote,
  Lamp,
  LampRow,
  LoadingRows,
  Panel,
  StatCells,
  TwoUp,
  Verdict,
  WriteGate,
} from './ui';
import { useExperiments, useRegistry, useVersionRows } from './useAdminData';
import { promoteVersion, setChallenger } from '../../lib/api';
import { ApiError } from '../../api';
import { qk } from '../../lib/queryClient';
import { useMode } from '../../modeStore';
import type { ExperimentRun, PromotionEvent, RegistryVersion } from '../../types';

/**
 * Optimise › Campaigns: how the agent is improved, as rounds. The champion's track, every
 * experiment a campaign gated (one opens in place as target → rewrite → gate), the promote
 * control, and the runs, registry and append-only promotion record behind them.
 *
 * The promotion history is append-only and is presented that way — a list of
 * events with the actor and the comparator's reason attached, not a "current
 * state" that quietly forgets how it got there. The promote control is the one
 * write on this page and is wrapped in a real disabled fieldset with the reason
 * printed beside it; the server refuses the same call independently.
 *
 * Campaigns group runs into the experiments they belong to (one subagent's
 * prompt rewritten and gated per experiment); this is pipeline-only by design
 * — the single-session challenger runs a separate cap and gate. See Runtimes
 * for that comparison and the runtime recommendation.
 */

const AGENTS = ['triage', 'preprocess', 'retriever', 'calculator'] as const;
type Agent = (typeof AGENTS)[number];
const isAgent = (a: string | null | undefined): a is Agent => AGENTS.includes(a as Agent);

/** How each series is drawn, said in words for the key under the chart. */
const SERIES_KEY: Record<'overall' | Agent, string> = {
  overall: 'overall gate accuracy (heavy ink)',
  triage: 'triage (grey, dashed)',
  preprocess: 'preprocess (amber)',
  retriever: 'retriever (blue)',
  calculator: 'calculator (green, dotted)',
};

function pct(value: number | null | undefined, digits = 1) {
  return value === null || value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`;
}

const pp = (value: number | null | undefined) => fmtPp(value, 2);

function runStatusTone(status: string): StatusTone {
  const s = status.toUpperCase();
  if (s === 'FINISHED') return 'ok';
  if (s === 'FAILED' || s === 'KILLED') return 'err';
  if (s === 'RUNNING' || s === 'SCHEDULED') return 'warn';
  return 'no';
}

// ---------------------------------------------------------------------------
// Fig · the champion over time
// ---------------------------------------------------------------------------

/**
 * Overall accuracy plus each subagent's own gold-derived metric, at every point
 * the champion moved. Hand-drawn SVG on the kit's `.dia` vocabulary: five series
 * over a handful of points does not justify a charting dependency, and this way
 * the shape is the same in the app and on the published page.
 */
function ChampionChart({ track }: { track: ChampionPoint[] }) {
  const points = track.filter((p) => p.accuracy !== null && p.accuracy !== undefined);
  if (points.length < 2) return null;

  const w = 900;
  const h = 310;
  const [left, right, top, bottom] = [58, 132, 24, 56];
  const series: Array<{ name: 'overall' | Agent; values: Array<number | null> }> = [
    { name: 'overall', values: points.map((p) => p.accuracy ?? null) },
    ...AGENTS.map((a) => ({ name: a, values: points.map((p) => p.panel?.[a] ?? null) })),
  ];
  const all = series.flatMap((s) => s.values).filter((v): v is number => v !== null);
  const rawLo = Math.min(...all);
  const rawHi = Math.max(...all);
  const pad = Math.max(0.02, (rawHi - rawLo) * 0.25);
  const lo = Math.max(0, rawLo - pad);
  const hi = Math.min(1, rawHi + pad);
  const xOf = (i: number) => left + (i * (w - left - right)) / Math.max(1, points.length - 1);
  const yOf = (v: number) => top + ((hi - v) / (hi - lo)) * (h - top - bottom);

  // End labels are placed after the lines and nudged apart: two series can
  // finish within a few tenths of a point of each other, and overprinted
  // labels read as a rendering fault rather than as two close values.
  const placed = series
    .map((s) => {
      const lastIdx = s.values.reduce<number>((acc, v, i) => (v !== null ? i : acc), -1);
      return lastIdx < 0 ? null : { y: yOf(s.values[lastIdx] as number), name: s.name };
    })
    .filter((v): v is { y: number; name: 'overall' | Agent } => v !== null)
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < placed.length; i += 1) {
    if (placed[i].y - placed[i - 1].y < 15) placed[i].y = placed[i - 1].y + 15;
  }

  return (
    <div className="figscroll">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="dia opt-chart"
        style={{ minWidth: 640 }}
        role="img"
        aria-label="Champion accuracy and per-subagent metrics at every promotion"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const v = lo + f * (hi - lo);
          return (
            <g key={f}>
              <line className="gl" x1={left} y1={yOf(v)} x2={w - right} y2={yOf(v)} />
              <text className="tx k" x={left - 8} y={yOf(v) + 4} textAnchor="end">
                {(v * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
        {points.map((p, i) => (
          <g key={`${p.version}-${i}`}>
            <text className="tx k" x={xOf(i)} y={h - bottom + 22} textAnchor="middle">
              {versionLabel(p.version)}
            </text>
            {p.target_agent && (
              <text
                className={`tx lbl s-${isAgent(p.target_agent) ? p.target_agent : 'overall'}`}
                x={xOf(i)}
                y={h - bottom + 40}
                textAnchor="middle"
              >
                ↑ {p.target_agent}
              </text>
            )}
          </g>
        ))}
        {series.map((s) => {
          const pts = s.values
            .map((v, i) => (v === null ? null : `${xOf(i)},${yOf(v)}`))
            .filter((v): v is string => v !== null);
          if (pts.length < 2) return null;
          return (
            <g key={s.name} className={`s-${s.name}`}>
              <polyline className={`ln s-${s.name}`} points={pts.join(' ')} />
              {s.values.map((v, i) =>
                v === null ? null : <circle key={i} className="dot" cx={xOf(i)} cy={yOf(v)} r={3.2} />,
              )}
            </g>
          );
        })}
        {placed.map((l) => (
          <text key={l.name} className={`tx lbl s-${l.name}`} x={w - right + 10} y={l.y + 4}>
            {l.name}
          </text>
        ))}
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// One experiment, opened in place: target → rewrite → gate
// ---------------------------------------------------------------------------

type Step = 'target' | 'rewrite' | 'gate';

function DiffBlock({ diff }: { diff: string }) {
  return (
    <div className="code diff opt-diff">
      <pre>
        {diff.split('\n').map((line, i) => {
          const cls =
            line.startsWith('+') && !line.startsWith('+++')
              ? 'add'
              : line.startsWith('-') && !line.startsWith('---')
                ? 'del'
                : line.startsWith('@@')
                  ? 'hunk'
                  : 'ctx';
          return (
            <span key={i} className={cls}>
              {line || ' '}
            </span>
          );
        })}
      </pre>
    </div>
  );
}

function ExperimentOpen({ exp }: { exp: CampaignExperiment }) {
  const [step, setStep] = useState<Step>('gate');
  const p = exp.cluster_p_one_sided;
  const n = exp.n_compared ?? 0;
  const steps: Array<{ key: Step; n: string; title: string; body: string }> = [
    {
      key: 'target',
      n: '1',
      title: 'Target',
      body: `${exp.target_agent}, the subagent the pooled attribution ranked first against ${versionLabel(exp.baseline_version)}`,
    },
    {
      key: 'rewrite',
      n: '2',
      title: 'Rewrite',
      body: exp.summary_of_changes ? clip(exp.summary_of_changes, 140) : 'no summary recorded',
    },
    {
      key: 'gate',
      n: '3',
      title: 'Gate',
      body: `${pp(exp.accuracy_delta)} on ${formatCount(n)} shared questions · ${exp.fixed ?? 0} fixed, ${exp.broken ?? 0} broken · p=${p === null ? '—' : p.toFixed(3)} → ${exp.promoted ? 'promoted' : 'rejected'}`,
    },
  ];
  return (
    <section className="round-open" aria-label={`experiment ${exp.label || exp.candidate_version}`}>
      <p className="label">
        {exp.campaign} · {versionLabel(exp.baseline_version)} → {versionLabel(exp.candidate_version)}
        {exp.at ? ` · ${formatEpochMs(exp.at)}` : ''}
      </p>
      <h3>
        {versionLabel(exp.candidate_version)}: {pct(exp.accuracy_baseline)} → {pct(exp.accuracy_candidate)} (
        {pp(exp.accuracy_delta)}), {exp.promoted ? 'promoted' : 'rejected'}
      </h3>
      <div className="steps3" role="tablist" aria-label="the experiment's three steps">
        {steps.map((x, i) => (
          <button
            key={x.key}
            type="button"
            role="tab"
            aria-selected={step === x.key}
            className={`step ${step === x.key ? 'on' : ''}`}
            onClick={() => setStep(x.key)}
          >
            <span className="label">
              {x.n} · {x.title}
            </span>
            <span className="small">{x.body}</span>
            {i < steps.length - 1 && (
              <span className="arrow" aria-hidden="true">
                →
              </span>
            )}
          </button>
        ))}
      </div>
      {step === 'target' && (
        <>
          <p>
            One subagent&rsquo;s prompt per experiment: this one rewrote <b>{exp.target_agent}</b>, so any
            move in the champion has a named cause.
          </p>
          {exp.rationale ? <p className="small">{exp.rationale}</p> : <p className="small muted">— no rationale recorded</p>}
        </>
      )}
      {step === 'rewrite' && (
        <>
          <p>{exp.summary_of_changes || '— no summary of changes recorded'}</p>
          {exp.diff ? <DiffBlock diff={exp.diff} /> : <p className="small muted">— no prompt diff recorded for this experiment</p>}
        </>
      )}
      {step === 'gate' && (
        <>
          <p>
            {exp.fixed ?? 0} fixed and {exp.broken ?? 0} broken of {formatCount(n)} shared questions; Δ{' '}
            {pp(exp.accuracy_delta)}, 95% CI [{pp(exp.delta_ci_lo)}, {pp(exp.delta_ci_hi)}], one-sided
            cluster-corrected p={p === null ? '—' : p.toFixed(3)}.{' '}
            <Verdict ok={exp.promoted}>{exp.promoted ? 'promoted' : 'rejected'}</Verdict>
          </p>
          <StatCells
            columns={4}
            cells={AGENTS.map((a) => {
              const b = exp.panel_baseline?.[a];
              const c = exp.panel_candidate?.[a];
              const dir = b == null || c == null ? '' : c > b ? ' ▲' : c < b ? ' ▼' : '';
              return { label: `${a} before → after`, value: `${pct(b)} → ${pct(c)}${dir}` };
            })}
          />
        </>
      )}
    </section>
  );
}

function ExperimentTable({
  experiments,
  champion,
}: {
  experiments: CampaignExperiment[];
  champion: string | null | undefined;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const keyOf = (e: CampaignExperiment) => `${e.campaign}-${e.label}-${e.candidate_version}`;
  return (
    <div className="tw opt-exp">
      <table style={{ minWidth: 760 }}>
        <caption>
          Every challenger, promoted or not, newest campaign last. Click a row to open it as target → rewrite → gate.
          Source: /eval/campaigns.
        </caption>
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Target</th>
            <th>Verdict</th>
            <th className="num">Accuracy</th>
            <th className="num">Δ</th>
            <th className="num">Fixed / broken</th>
            <th className="num">p (one-sided)</th>
          </tr>
        </thead>
        <tbody>
          {experiments.map((exp) => {
            const k = keyOf(exp);
            const isOpen = open === k;
            const p = exp.cluster_p_one_sided;
            const toggle = () => setOpen(isOpen ? null : k);
            const cls = ['click', isOpen ? 'on' : exp.candidate_version === champion ? CHAMPION_ROW : '']
              .filter(Boolean)
              .join(' ');
            return (
              <Fragment key={k}>
                <tr className={cls} onClick={toggle}>
                  <td className="sub">
                    <button
                      type="button"
                      className="linkish"
                      aria-expanded={isOpen}
                      onClick={(e) => {
                        e.stopPropagation();
                        toggle();
                      }}
                    >
                      {exp.label || exp.candidate_version}
                    </button>
                    <span className="path">
                      {exp.campaign} · {versionLabel(exp.baseline_version)} → {versionLabel(exp.candidate_version)}
                    </span>
                  </td>
                  <td className="mono">{exp.target_agent}</td>
                  <td>
                    <Verdict ok={exp.promoted}>{exp.promoted ? 'promoted' : 'rejected'}</Verdict>
                  </td>
                  <td className="num">
                    {pct(exp.accuracy_baseline)} → {pct(exp.accuracy_candidate)}
                  </td>
                  <td className="num">{pp(exp.accuracy_delta)}</td>
                  <td className="num">
                    {exp.fixed ?? 0} / {exp.broken ?? 0} of {formatCount(exp.n_compared ?? 0)}
                  </td>
                  <td className="num">{p === null ? '—' : p.toFixed(3)}</td>
                </tr>
                {isOpen && (
                  <tr className="detail">
                    <td colSpan={7}>
                      <ExperimentOpen exp={exp} />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------

function RunDetail({ run }: { run: ExperimentRun }) {
  const metrics = Object.entries(run.metrics);
  const params = Object.entries(run.params);
  return (
    <div className="card opt-run">
      <p className="label">
        {run.run_name} · run_id {run.run_id}
      </p>
      <div className="grid2">
        <div>
          <h3>Metrics · {metrics.length}</h3>
          {metrics.length === 0 ? (
            <p className="small muted">— this run logged no metrics</p>
          ) : (
            <dl className="kv">
              {metrics.map(([key, value]) => (
                <Fragment key={key}>
                  <dt>{key}</dt>
                  <dd className="num">
                    {key.includes('acc') && value <= 1 ? formatPercent(value) : formatCount(value)}
                  </dd>
                </Fragment>
              ))}
            </dl>
          )}
        </div>
        <div>
          <h3>Params · {params.length}</h3>
          <dl className="kv">
            {params.map(([key, value]) => (
              <Fragment key={key}>
                <dt>{key}</dt>
                <dd className="mono wrap-any">{value}</dd>
              </Fragment>
            ))}
          </dl>
        </div>
      </div>
    </div>
  );
}

function Section({ testId, n, label, claim, children }: { testId: string; n: number; label: string; claim: ReactNode; children: ReactNode }) {
  return (
    <section className="opt-sec" data-testid={testId}>
      <h2>
        {n} · {label} — {claim}
      </h2>
      {children}
    </section>
  );
}

// ---------------------------------------------------------------------------

export default function Experiments() {
  const health = useMode((s) => s.health);
  const isDemo = health?.mode === 'demo';
  const experiments = useExperiments();
  const registry = useRegistry();
  const { rows, champion } = useVersionRows();
  const queryClient = useQueryClient();

  const [openRun, setOpenRun] = useState<string | null>(null);
  const [kindFilter, setKindFilter] = useState('');
  const [target, setTarget] = useState('');
  const [writeError, setWriteError] = useState<string | null>(null);
  const [writeOk, setWriteOk] = useState<string | null>(null);
  const [campaignOnly, setCampaignOnly] = useState('');

  const campaignsQuery = useQuery({
    queryKey: ['eval-campaigns'],
    queryFn: () => getCampaigns(),
    staleTime: 60_000,
  });
  const campaignData = campaignsQuery.data;
  const shownExperiments = useMemo(
    () => (campaignData?.experiments ?? []).filter((e) => !campaignOnly || e.campaign === campaignOnly),
    [campaignData, campaignOnly],
  );

  const runs = experiments.data?.runs ?? [];
  const kinds = useMemo(() => [...new Set(runs.map((r) => r.kind))].sort(), [runs]);
  const shownRuns = kindFilter ? runs.filter((r) => r.kind === kindFilter) : runs;

  const canPromote = registry.data?.can_promote ?? false;
  const versionNames = rows.map((r) => r.version);
  // Default to the newest non-champion — the version an operator is actually
  // deciding about — rather than the oldest one in the list.
  const selectedTarget = target || versionNames.filter((v) => v !== champion).slice(-1)[0] || '';

  /**
   * Why writes are refused, in the terms of this deployment.
   *
   * The demo and a dev box with no `OWNER_TOKEN` are different refusals and
   * must not share a sentence: one is a permanent property of the public
   * deployment, the other is a five-second fix on a laptop.
   */
  const writeReason = isDemo
    ? 'Read-only demo. The write routes refuse this with a 501 (not_available_demo) even if a client forges the request, and this container holds no owner token to present.'
    : 'No OWNER_TOKEN is configured on this backend, so admin writes are refused with a 403 (owner_token_unset). Set one and reload to enable promotion.';

  function afterWrite() {
    void queryClient.invalidateQueries({ queryKey: qk.registry });
    void queryClient.invalidateQueries({ queryKey: qk.experiments });
    void queryClient.invalidateQueries({ queryKey: qk.health });
  }

  const promote = useMutation({
    mutationFn: (version: string) => promoteVersion(version),
    onSuccess: (_, version) => {
      setWriteError(null);
      setWriteOk(`${version} promoted to champion.`);
      afterWrite();
    },
    onError: (err) => {
      setWriteOk(null);
      // A 409 is the comparator refusing, which is the gate working — say so
      // rather than presenting it as a broken request.
      const status = err instanceof ApiError ? err.status : 0;
      const prefix = status === 409 ? 'The gate refused this promotion: ' : '';
      setWriteError(`${prefix}${err instanceof Error ? err.message : String(err)}`);
    },
  });

  const challenger = useMutation({
    mutationFn: (version: string) => setChallenger(version),
    onSuccess: (_, version) => {
      setWriteError(null);
      setWriteOk(`challenger alias now points at ${version}.`);
      afterWrite();
    },
    onError: (err) => {
      setWriteOk(null);
      setWriteError(err instanceof Error ? err.message : String(err));
    },
  });

  // -------------------------------------------------------------------------

  const runColumns = useMemo<Array<ColumnDef<ExperimentRun, unknown>>>(
    () => [
      {
        id: 'run',
        header: 'run',
        accessorFn: (r) => r.run_name,
        meta: { align: 'left', mono: false, width: '150px' },
        cell: ({ row }) => (
          <button
            type="button"
            className="linkish"
            aria-expanded={openRun === row.original.run_id}
            onClick={() => setOpenRun((id) => (id === row.original.run_id ? null : row.original.run_id))}
          >
            {row.original.run_name}
          </button>
        ),
      },
      {
        id: 'kind',
        header: 'kind',
        accessorFn: (r) => r.kind,
        meta: { align: 'left', width: '58px' },
      },
      {
        id: 'status',
        header: 'status',
        accessorFn: (r) => r.status,
        meta: { align: 'left', width: '76px' },
        cell: ({ row }) => (
          <Status tone={runStatusTone(row.original.status)}>{row.original.status.toLowerCase()}</Status>
        ),
      },
      {
        id: 'bundle',
        header: 'bundle',
        accessorFn: (r) => r.bundle_id,
        meta: { width: '96px' },
        cell: ({ row }) => <span title={bundleLine(row.original.params)}>{row.original.bundle_id}</span>,
      },
      {
        id: 'accuracy',
        header: 'accuracy',
        accessorFn: (r) => r.metrics.accuracy ?? r.metrics.exe_acc ?? -1,
        cell: ({ row }) => {
          const value = row.original.metrics.accuracy ?? row.original.metrics.exe_acc;
          return value === undefined ? <span className="muted">—</span> : formatPercent(value);
        },
      },
      {
        id: 'started',
        header: 'started',
        accessorFn: (r) => r.start_time,
        cell: ({ row }) => formatEpochMs(row.original.start_time),
      },
      {
        id: 'duration',
        header: 'took',
        accessorFn: (r) => r.end_time - r.start_time,
        cell: ({ row }) => formatRunDuration(row.original.start_time, row.original.end_time),
      },
    ],
    [openRun],
  );

  const registryColumns = useMemo<Array<ColumnDef<RegistryVersion, unknown>>>(
    () => [
      {
        id: 'version',
        header: 'version',
        accessorFn: (r) => r.version,
        meta: { align: 'left', mono: false, width: '104px' },
        cell: ({ row }) => {
          const aliases = Object.entries(registry.data?.aliases ?? {})
            .filter(([, v]) => v === row.original.version)
            .map(([alias]) => alias);
          return (
            <span>
              {row.original.version}
              {aliases.length > 0 && <span className="muted"> · {aliases.join(' · ')}</span>}
            </span>
          );
        },
      },
      {
        id: 'bundle',
        header: 'bundle',
        accessorFn: (r) => r.bundle_id,
        meta: { width: '96px' },
        cell: ({ row }) => (
          <span title={bundleLine(row.original.bundle as unknown as Record<string, unknown>)}>
            {row.original.bundle_id}
          </span>
        ),
      },
      {
        id: 'accuracy',
        header: 'accuracy',
        accessorFn: (r) => r.metrics.accuracy ?? -1,
        cell: ({ row }) => formatPercent(row.original.metrics.accuracy ?? null),
      },
      {
        id: 'runs',
        header: 'runs',
        accessorFn: (r) => r.runs.length,
        cell: ({ row }) => formatCount(row.original.runs.length),
      },
      {
        id: 'source',
        header: 'source',
        accessorFn: (r) => r.source,
        meta: { align: 'left', width: '68px' },
      },
      {
        id: 'registered',
        header: 'registered',
        accessorFn: (r) => r.registered_at,
        cell: ({ row }) => formatStamp(row.original.registered_at),
      },
    ],
    [registry.data?.aliases],
  );

  const historyColumns = useMemo<Array<ColumnDef<PromotionEvent, unknown>>>(
    () => [
      {
        id: 'at',
        header: 'when',
        accessorFn: (r) => r.at,
        meta: { align: 'left', width: '110px' },
        cell: ({ row }) => formatStamp(row.original.at),
      },
      {
        id: 'event',
        header: 'event',
        accessorFn: (r) => r.event,
        meta: { align: 'left', width: '72px' },
        cell: ({ row }) => <span className="tag">{row.original.event}</span>,
      },
      {
        id: 'version',
        header: 'version',
        accessorFn: (r) => r.version,
        meta: { align: 'left', mono: false, width: '80px' },
      },
      {
        id: 'previous',
        header: 'replaced',
        accessorFn: (r) => r.previous_champion ?? '',
        meta: { width: '80px' },
        cell: ({ row }) => (
          <span className={row.original.previous_champion ? '' : 'muted'}>
            {row.original.previous_champion ?? 'none'}
          </span>
        ),
      },
      {
        id: 'actor',
        header: 'actor',
        accessorFn: (r) => r.actor,
        meta: { align: 'left', width: '80px' },
      },
      {
        id: 'forced',
        header: 'forced',
        accessorFn: (r) => (r.forced ? 1 : 0),
        cell: ({ row }) =>
          row.original.forced ? <Verdict ok={false}>forced</Verdict> : <span className="muted">no</span>,
      },
      {
        id: 'reason',
        header: 'reason',
        accessorFn: (r) => r.reason,
        meta: { align: 'left', mono: false, wrap: true, width: '220px' },
        cell: ({ row }) => <span className="muted">{clip(row.original.reason, 140)}</span>,
      },
    ],
    [],
  );

  const history = [...(registry.data?.history ?? [])].reverse();

  // ---- the headline, computed from what the page already holds -------------
  const allExperiments = campaignData?.experiments ?? [];
  const nPromoted = allExperiments.filter((e) => e.promoted).length;
  const track = (campaignData?.champion_track ?? []).filter((p) => p.accuracy !== null && p.accuracy !== undefined);
  const firstPoint = track[0] ?? null;
  // This page is the pipeline's own campaigns, so its champion is the pipeline
  // lineage's best. The project's champion (an agent_sdk agent since
  // 2026-10-15) is on the Runtimes page.
  const champ = campaignData?.pipeline_champion ?? campaignData?.champion ?? null;
  const champAcc =
    campaignData?.pipeline_champion_accuracy ?? campaignData?.champion_accuracy ?? null;

  const title: ReactNode = campaignsQuery.isLoading ? (
    <>
      Reading the <em>campaign</em> record…
    </>
  ) : allExperiments.length === 0 ? (
    <>
      No experiment has been <em>gated</em> yet
    </>
  ) : (
    <>
      {allExperiments.length} pipeline experiments gated, {nPromoted} promoted: the champion{' '}
      {champ ? versionLabel(champ) : '—'} scores <em>{pct(champAcc)}</em> on the gate split
      {firstPoint && track.length > 1 && firstPoint.version !== champ
        ? `, against ${pct(firstPoint.accuracy)} for ${versionLabel(firstPoint.version)}`
        : ''}
    </>
  );

  const moves = Math.max(0, track.length - 1);
  const trackClaim =
    track.length > 1
      ? `${moves} promotion${moves === 1 ? '' : 's'} moved the champion from ${pct(firstPoint?.accuracy)} to ${pct(track[track.length - 1].accuracy)}`
      : 'the champion has not moved yet';

  const nShownPromoted = shownExperiments.filter((e) => e.promoted).length;

  return (
    <AdminPage
      testId="admin-experiments"
      eyebrow={`optimise · campaigns · ${formatCount(campaignData?.campaigns?.length ?? 0)} campaigns · champion ${champ ? versionLabel(champ) : '—'}`}
      title={title}
      sub="A campaign rewrites one subagent's prompt per experiment and promotes it only when a one-sided, cluster-corrected McNemar test on the fixed gate split says it helped. Every attempt is below, rejected ones included, with the runs, registry and promotion record behind them."
    >
      <LampRow>
        <Lamp
          label="tracking"
          value={experiments.data?.source === 'live' ? 'live mlflow' : 'snapshot'}
          tone={experiments.data?.source === 'live' ? 'good' : 'info'}
          dashed={experiments.data?.source !== 'live'}
          title={
            experiments.data?.source === 'live'
              ? String(
                  (experiments.data?.tracking as Record<string, unknown> | undefined)
                    ?.tracking_uri ?? '',
                )
              : `committed export${
                  experiments.data?.exported_at ? ` from ${formatStamp(experiments.data.exported_at)}` : ''
                }`
          }
        />
        {Object.entries(registry.data?.aliases ?? {}).map(([alias, version]) => (
          <Lamp key={alias} label={alias} value={versionLabel(version)} tone="info" title={version} />
        ))}
        <Lamp
          label="writes"
          value={canPromote ? 'enabled' : 'refused'}
          tone={canPromote ? 'good' : 'idle'}
          dashed={!canPromote}
          title={canPromote ? 'This backend accepts a promotion.' : writeReason}
        />
        <Lamp label="runs" value={formatCount(runs.length)} tone="idle" dashed />
      </LampRow>

      {experiments.error ? <ErrorNote error={experiments.error} /> : null}

      <Section testId="experiments-campaign-track" n={1} label="Track" claim={trackClaim}>
        <p>
          Every point is a four-agent pipeline version with its per-subagent accuracy. A single Claude
          session has no subagents to plot; its own campaign (capped at 2 experiments, not 5) and the
          sdk_v1-vs-pipeline comparison are on <Link to="/optimise/runtimes">Runtimes</Link>.
        </p>
        {campaignsQuery.isLoading ? (
          <LoadingRows rows={6} />
        ) : campaignsQuery.error ? (
          <ErrorNote error={campaignsQuery.error} />
        ) : !campaignData?.champion_track?.length ? (
          <EmptyState>
            {campaignData?.experiments?.length
              ? `${campaignData.experiments.length} experiment${campaignData.experiments.length === 1 ? '' : 's'} gated, none significant — the experiments are below.`
              : 'No experiment has been gated yet.'}
          </EmptyState>
        ) : (
          <Figure
            title="Fig 1 · the champion at every promotion: overall gate accuracy and each subagent's own gold-derived metric"
            caption={
              <>
                <span className="opt-key">
                  {(['overall', ...AGENTS] as const).map((s) => (
                    <span key={s} className={`s-${s}`}>
                      {SERIES_KEY[s]}
                    </span>
                  ))}
                </span>
                Only promoted experiments move these lines; the arrow under a version names the one
                subagent that experiment rewrote. {campaignData.rule}
              </>
            }
            source={<code>/eval/campaigns</code>}
          >
            <ChampionChart track={campaignData.champion_track} />
          </Figure>
        )}
      </Section>

      <Section
        testId="experiments-campaigns"
        n={2}
        label="Experiments"
        claim={`${shownExperiments.length} gated, ${nShownPromoted} promoted${campaignOnly ? ` in ${campaignOnly}` : ''}`}
      >
        {(campaignData?.campaigns?.length ?? 0) > 1 && (
          <div className="chips" role="group" aria-label="filter by campaign">
            {['', ...(campaignData?.campaigns ?? []).map((c) => c.name)].map((name) => (
              <button
                key={name || 'all'}
                type="button"
                aria-pressed={name === campaignOnly}
                onClick={() => setCampaignOnly(name)}
                className={`chip${name === campaignOnly ? ' on' : ''}`}
              >
                {name || 'all campaigns'}
              </button>
            ))}
          </div>
        )}
        {shownExperiments.length === 0 ? (
          <EmptyState>Nothing gated yet.</EmptyState>
        ) : (
          <ExperimentTable experiments={shownExperiments} champion={champ} />
        )}
      </Section>

      <Panel
        testId="experiments-promote"
        title="Promote — the comparator decides, not this form"
        endpoint="POST /admin/registry/promote"
        note="this form uses the legacy comparator rule — net positive on the shared questions (tracking/comparator.py promotable); campaigns promote only on net positive and one-sided cluster McNemar p < 0.05"
      >
        <WriteGate enabled={canPromote} reason={writeReason} testId="promote-gate">
          <label className="opt-field">
            <span className="label">version</span>
            <select
              value={selectedTarget}
              onChange={(e) => setTarget(e.target.value)}
              data-testid="promote-version"
            >
              {versionNames.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="btn"
            data-testid="promote-submit"
            onClick={() => promote.mutate(selectedTarget)}
          >
            {promote.isPending ? 'Promoting…' : 'Promote to champion'}
          </button>
          <button
            type="button"
            className="linkbtn"
            data-testid="challenger-submit"
            onClick={() => challenger.mutate(selectedTarget)}
          >
            {challenger.isPending ? 'setting…' : 'Set as challenger'}
          </button>
        </WriteGate>

        {writeError && (
          <div className="callout err opt-msg" role="alert">
            <span className="ttl">✗ Refused</span>
            {writeError}
          </div>
        )}
        {writeOk && (
          <p className="small v-ok opt-msg" role="status">
            ✓ {writeOk}
          </p>
        )}
        <Caveat>
          The gate is three layers deep. This control is a real{' '}
          <code>&lt;fieldset disabled&gt;</code>, the write routes are refused by{' '}
          <code>require_owner</code> before the handler body runs, and{' '}
          <code>_demo_write_blocked()</code> refuses again inside it. A forged request gets a 403 or
          a 501, not a promotion.
        </Caveat>
      </Panel>

      <Section
        testId="experiments-runs"
        n={3}
        label="Runs"
        claim={`${formatCount(runs.length)} recorded ${
          experiments.data?.source === 'snapshot' ? 'in the committed MLflow snapshot' : 'in the live MLflow store'
        }`}
      >
        {kinds.length > 0 && (
          <div className="chips" role="group" aria-label="filter by run kind">
            {['', ...kinds].map((kind) => (
              <button
                key={kind || 'all'}
                type="button"
                aria-pressed={kindFilter === kind}
                onClick={() => setKindFilter(kind)}
                className={`chip${kindFilter === kind ? ' on' : ''}`}
              >
                {kind || 'all kinds'}
              </button>
            ))}
          </div>
        )}
        {experiments.isLoading ? (
          <LoadingRows rows={5} />
        ) : shownRuns.length === 0 ? (
          <EmptyState>
            No runs recorded. `uv run convfinqa-mlflow backfill` rebuilds the history from the
            committed artifacts.
          </EmptyState>
        ) : (
          <>
            <InstrumentTable
              data={shownRuns}
              columns={runColumns}
              rowKey={(r) => r.run_id}
              rowClass={(r) => (r.run_id === openRun ? 'on' : undefined)}
              minWidth={720}
              initialSorting={[{ id: 'started', desc: true }]}
              caption={
                <>
                  {experiments.data?.source === 'snapshot'
                    ? 'From the committed MLflow snapshot — the demo image carries no tracking store.'
                    : 'From the live MLflow tracking store.'}{' '}
                  Click a run name for its params and metrics. Source: /admin/experiments.
                </>
              }
            />
            {openRun &&
              (() => {
                const run = runs.find((r) => r.run_id === openRun);
                return run ? <RunDetail run={run} /> : null;
              })()}
            <p className="small muted">
              GEPA runs remain broken against DeepSeek — <code>dspy_lm_kwargs()</code> still hits the
              thinking-mode 400 — so any GEPA row here predates that regression.
            </p>
          </>
        )}
      </Section>

      <TwoUp>
        <Section
          testId="experiments-registry"
          n={4}
          label="Registry"
          claim={`${formatCount(registry.data?.versions?.length ?? 0)} versions, champion ${champion ? versionLabel(champion) : '—'}`}
        >
          {registry.isLoading ? (
            <LoadingRows rows={3} />
          ) : registry.error ? (
            <ErrorNote error={registry.error} />
          ) : (
            <InstrumentTable
              data={registry.data?.versions ?? []}
              columns={registryColumns}
              rowKey={(r) => r.version}
              rowClass={(r) => (r.version === champion ? CHAMPION_ROW : undefined)}
              minWidth={620}
              emptyLabel="nothing registered yet"
              caption="A version label means nothing when every model is an API; the bundle id is what an answer is attributable to. Source: /admin/registry."
            />
          )}
        </Section>

        <Section
          testId="experiments-holdout"
          n={5}
          label="Held out"
          claim="never-seen accuracy sits beside the overall, never blended in"
        >
          <InstrumentTable
            data={rows}
            columns={
              [
                {
                  id: 'version',
                  header: 'version',
                  accessorFn: (r) => r.version,
                  meta: { align: 'left', mono: false, width: '104px' },
                  cell: ({ row }) => (
                    <Link to={`/runs?version=${row.original.version}`}>{row.original.version}</Link>
                  ),
                },
                {
                  id: 'holdout',
                  header: 'never-seen',
                  accessorFn: (r) => r.holdout ?? -1,
                  cell: ({ row }) => formatPercent(row.original.holdout),
                },
                {
                  id: 'holdoutN',
                  header: 'n',
                  accessorFn: (r) => r.holdoutN ?? -1,
                  cell: ({ row }) => formatCount(row.original.holdoutN),
                },
                {
                  id: 'overall',
                  header: 'overall',
                  accessorFn: (r) => r.overall ?? -1,
                  cell: ({ row }) => formatPercent(row.original.overall),
                },
                {
                  id: 'n',
                  header: 'n',
                  accessorFn: (r) => r.nQuestions ?? -1,
                  cell: ({ row }) => formatCount(row.original.nQuestions),
                },
              ] as Array<ColumnDef<(typeof rows)[number], unknown>>
            }
            rowKey={(r) => r.version}
            rowClass={(r) => (r.isChampion ? CHAMPION_ROW : undefined)}
            minWidth={460}
            caption="The only endpoint that can separate optimizer_train from never_seen. Source: /admin/experiments."
          />
          <Caveat>
            The two columns are different populations and are never averaged. &ldquo;Held
            out&rdquo; here means <code>data.loader.optimizer_split()</code> — the 309 questions no
            optimizer ever saw — not <code>train_report_ids</code>, which is a different 60/40 split
            that agrees with it on only 78 of 120 conversations.
          </Caveat>
        </Section>
      </TwoUp>

      <Section
        testId="experiments-history"
        n={6}
        label="History"
        claim={`${formatCount(history.length)} promotion event${history.length === 1 ? '' : 's'}, append-only`}
      >
        {registry.isLoading ? (
          <LoadingRows rows={2} />
        ) : history.length === 0 ? (
          <EmptyState>No promotion has been recorded on this deployment.</EmptyState>
        ) : (
          <InstrumentTable
            data={history}
            columns={historyColumns}
            rowKey={(r, i) => `${r.at}:${i}`}
            minWidth={720}
            caption="Nothing here is edited or removed, forced promotions included; newest first. Source: /admin/registry."
          />
        )}
      </Section>
    </AdminPage>
  );
}
