import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
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
 * Experiments: every run, every campaign, the registry, and what was promoted
 * — one page instead of three that used to say the same thing three ways.
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

const KIND_TONE: Record<string, string> = {
  eval: 'text-info',
  gepa: 'text-violet',
  s7: 'text-amber',
};

const AGENTS = ['triage', 'preprocess', 'retriever', 'calculator'] as const;
type Agent = (typeof AGENTS)[number];

const AGENT_COLOR: Record<Agent, string> = {
  triage: 'var(--info)',
  preprocess: 'var(--violet)',
  retriever: 'var(--amber)',
  calculator: 'var(--good)',
};

function pct(value: number | null | undefined, digits = 1) {
  return value === null || value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`;
}

function pp(value: number | null | undefined) {
  return value === null || value === undefined ? '—' : `${(value * 100 >= 0 ? '+' : '')}${(value * 100).toFixed(2)}pp`;
}

/**
 * Overall accuracy plus each subagent's own gold-derived metric, at every point
 * the champion moved. Hand-drawn SVG rather than a charting dependency: five
 * series over a handful of points does not justify one, and this way the shape
 * is the same in the app and on the published page.
 */
function ChampionChart({ track }: { track: ChampionPoint[] }) {
  const points = track.filter((p) => p.accuracy !== null && p.accuracy !== undefined);
  if (points.length < 2) return null;

  const w = 900;
  const h = 300;
  const [left, right, top, bottom] = [58, 132, 24, 48];
  const series: Array<{ name: string; colour: string; width: number; values: Array<number | null> }> = [
    { name: 'overall', colour: 'var(--text)', width: 2.4, values: points.map((p) => p.accuracy ?? null) },
    ...AGENTS.map((a) => ({
      name: a,
      colour: AGENT_COLOR[a],
      width: 1.4,
      values: points.map((p) => p.panel?.[a] ?? null),
    })),
  ];
  const all = series.flatMap((s) => s.values).filter((v): v is number => v !== null);
  const rawLo = Math.min(...all);
  const rawHi = Math.max(...all);
  const pad = Math.max(0.02, (rawHi - rawLo) * 0.25);
  const lo = Math.max(0, rawLo - pad);
  const hi = Math.min(1, rawHi + pad);
  const xOf = (i: number) => left + (i * (w - left - right)) / Math.max(1, points.length - 1);
  const yOf = (v: number) => top + ((hi - v) / (hi - lo)) * (h - top - bottom);

  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="Champion accuracy and per-subagent metrics">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const v = lo + f * (hi - lo);
          return (
            <g key={f}>
              <line x1={left} y1={yOf(v)} x2={w - right} y2={yOf(v)} stroke="currentColor" className="text-line" strokeWidth={1} />
              <text x={left - 8} y={yOf(v) + 3} textAnchor="end" className="fill-faint font-mono text-[10px]">
                {(v * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
        {points.map((p, i) => (
          <g key={`${p.version}-${i}`}>
            <text x={xOf(i)} y={h - 26} textAnchor="middle" className="fill-text font-mono text-[10px]">
              {versionLabel(p.version)}
            </text>
            {p.target_agent && (
              <text
                x={xOf(i)}
                y={h - 12}
                textAnchor="middle"
                className="font-mono text-[9px]"
                fill={AGENT_COLOR[p.target_agent as Agent] ?? 'var(--faint)'}
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
            <g key={s.name}>
              <polyline points={pts.join(' ')} fill="none" stroke={s.colour} strokeWidth={s.width} strokeLinejoin="round" />
              {s.values.map((v, i) =>
                v === null ? null : <circle key={i} cx={xOf(i)} cy={yOf(v)} r={3} fill={s.colour} />,
              )}
            </g>
          );
        })}
        {/* End labels are placed after the lines and nudged apart: two series can
            finish within a few tenths of a point of each other, and overprinted
            labels read as a rendering fault rather than as two close values. */}
        {(() => {
          const placed: Array<{ y: number; colour: string; name: string }> = series
            .map((s) => {
              const lastIdx = s.values.reduce<number>((acc, v, i) => (v !== null ? i : acc), -1);
              return lastIdx < 0
                ? null
                : { y: yOf(s.values[lastIdx] as number), colour: s.colour, name: s.name };
            })
            .filter((v): v is { y: number; colour: string; name: string } => v !== null)
            .sort((a, b) => a.y - b.y);
          const gap = 12;
          for (let i = 1; i < placed.length; i += 1) {
            if (placed[i].y - placed[i - 1].y < gap) placed[i].y = placed[i - 1].y + gap;
          }
          return placed.map((l) => (
            <text key={l.name} x={w - right + 8} y={l.y + 3} fill={l.colour} className="font-mono text-[10px]">
              {l.name}
            </text>
          ));
        })()}
      </svg>
      <figcaption className="type-meta mt-2 text-faint">
        Overall gate accuracy in white; each subagent&rsquo;s own gold-derived metric in colour. The
        arrow under a version names the one subagent that experiment rewrote.
      </figcaption>
    </figure>
  );
}

function ExperimentCard({ exp }: { exp: CampaignExperiment }) {
  const [open, setOpen] = useState(false);
  const p = exp.cluster_p_one_sided;
  return (
    <div className="rounded-[5px] border border-line bg-panel">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full flex-wrap items-center gap-2.5 px-3 py-2.5 text-left"
      >
        <span className="font-mono text-[12px] text-text">{exp.label || exp.candidate_version}</span>
        <Verdict ok={exp.promoted}>{exp.promoted ? 'promoted' : 'rejected'}</Verdict>
        <span
          className="rounded-[4px] border px-1.5 py-0.5 font-mono text-[10px]"
          style={{ color: AGENT_COLOR[exp.target_agent as Agent], borderColor: AGENT_COLOR[exp.target_agent as Agent] }}
        >
          {exp.target_agent}
        </span>
        <span className="ml-auto type-num text-[12px] text-muted">
          {pp(exp.accuracy_delta)} · p={p === null ? '—' : p.toFixed(3)}
        </span>
      </button>
      {open && (
        <div className="border-t border-line px-3 py-3">
          <p className="type-small mb-2 text-muted">
            <span className="font-mono text-text">
              {exp.baseline_version} → {exp.candidate_version}
            </span>{' '}
            — {exp.summary_of_changes || 'no summary recorded'}
          </p>
          {exp.rationale && <p className="type-small mb-3 text-muted">{exp.rationale}</p>}
          <div className="mb-3 grid grid-cols-2 gap-2 md:grid-cols-4">
            {AGENTS.map((a) => (
              <div key={a} className="rounded-[4px] border border-line-2 bg-panel-2 px-2 py-1.5">
                <div className="mono-caps text-faint">{a}</div>
                <div className="type-num text-[12px]">
                  {pct(exp.panel_baseline?.[a])} → {pct(exp.panel_candidate?.[a])}
                </div>
              </div>
            ))}
          </div>
          <div className="type-small text-faint">
            {exp.fixed ?? 0} fixed / {exp.broken ?? 0} broken of {exp.n_compared ?? 0} shared questions ·
            95% CI [{pp(exp.delta_ci_lo)}, {pp(exp.delta_ci_hi)}]
          </div>
          {exp.diff && (
            <pre className="mt-2 max-h-80 overflow-auto rounded-[4px] border border-line bg-ground p-2.5 font-mono text-[11px] leading-relaxed">
              {exp.diff.split('\n').map((line, i) => (
                <div
                  key={i}
                  className={cn(
                    line.startsWith('+') && !line.startsWith('+++') && 'text-good',
                    line.startsWith('-') && !line.startsWith('---') && 'text-bad',
                    line.startsWith('@@') && 'text-violet',
                  )}
                >
                  {line}
                </div>
              ))}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function RunDetail({ run }: { run: ExperimentRun }) {
  const metrics = Object.entries(run.metrics);
  const params = Object.entries(run.params);
  return (
    <div className="grid grid-cols-1 gap-3 rounded-[5px] border border-line bg-panel-2 p-3 md:grid-cols-2">
      <div className="min-w-0">
        <div className="mono-caps mb-1">metrics</div>
        {metrics.length === 0 ? (
          <p className="type-meta text-faint">this run logged no metrics</p>
        ) : (
          <ul className="flex flex-col gap-0.5">
            {metrics.map(([key, value]) => (
              <li
                key={key}
                className="flex items-baseline justify-between gap-3 border-b border-line py-0.5 last:border-0"
              >
                <span className="type-small text-muted">{key}</span>
                <span className="type-num text-[11px] text-text">
                  {key.includes('acc') && value <= 1 ? formatPercent(value) : formatCount(value)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="min-w-0">
        <div className="mono-caps mb-1">params</div>
        <ul className="flex flex-col gap-0.5">
          {params.map(([key, value]) => (
            <li
              key={key}
              className="flex items-baseline justify-between gap-3 border-b border-line py-0.5 last:border-0"
            >
              <span className="type-small text-muted">{key}</span>
              <span className="type-num text-[11px] break-all text-text">{value}</span>
            </li>
          ))}
        </ul>
        <div className="type-meta mt-2 break-all text-faint">run_id {run.run_id}</div>
      </div>
    </div>
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
            onClick={() => setOpenRun((id) => (id === row.original.run_id ? null : row.original.run_id))}
            className="text-left hover:text-amber"
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
        cell: ({ row }) => (
          <span className={KIND_TONE[row.original.kind] ?? 'text-faint'}>{row.original.kind}</span>
        ),
      },
      {
        id: 'status',
        header: 'status',
        accessorFn: (r) => r.status,
        meta: { align: 'left', width: '76px' },
        cell: ({ row }) => (
          <span className={row.original.status === 'FINISHED' ? 'text-good' : 'text-muted'}>
            {row.original.status.toLowerCase()}
          </span>
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
          return value === undefined ? <span className="text-faint">—</span> : formatPercent(value);
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
    [],
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
              {aliases.length > 0 && <span className="text-faint"> · {aliases.join(' · ')}</span>}
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
        cell: ({ row }) => <span className="text-amber">{row.original.event}</span>,
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
          <span className={row.original.previous_champion ? '' : 'text-faint'}>
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
          row.original.forced ? (
            <Verdict ok={false}>forced</Verdict>
          ) : (
            <span className="text-faint">no</span>
          ),
      },
      {
        id: 'reason',
        header: 'reason',
        accessorFn: (r) => r.reason,
        meta: { align: 'left', mono: false, wrap: true, width: '220px' },
        cell: ({ row }) => <span className="text-muted">{clip(row.original.reason, 140)}</span>,
      },
    ],
    [],
  );

  const history = [...(registry.data?.history ?? [])].reverse();

  return (
    <AdminPage
      testId="admin-experiments"
      eyebrow="admin · experiments"
      title="Experiments"
      sub="Every campaign, every run, the registry, and the append-only promotion record — one page, not three."
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

      <Panel
        testId="experiments-campaign-track"
        title="Campaign track — four-agent pipeline only"
        endpoint="/eval/campaigns"
        note={campaignData?.rule || 'the promotion rule'}
        right={
          <span className="type-small text-faint">
            champion {campaignData?.champion ? versionLabel(campaignData.champion) : '—'}
          </span>
        }
      >
        <p className="type-small mb-2 text-faint">
          Every point here is a four-agent pipeline version, plotted with its per-subagent
          accuracy — a single Claude session has no subagents, so it has nothing to plot on this
          chart. Its own campaign (capped at 2 experiments, not 5) and the sdk_v1-vs-pipeline
          comparison are on{' '}
          <Link to="/admin/runtimes" className="text-amber underline underline-offset-4">
            Runtimes
          </Link>
          .
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
          <>
            <ChampionChart track={campaignData.champion_track} />
            <Caveat>
              Only promoted experiments move this line. Each rewrites exactly one subagent's
              prompt; a paired significance test on the fixed gate split decides promotion.
            </Caveat>
          </>
        )}
      </Panel>

      <Panel
        testId="experiments-campaigns"
        title="Campaign experiments"
        endpoint="/eval/campaigns"
        note="every challenger, promoted or not"
        right={
          <span className="type-small text-faint">
            {shownExperiments.length} experiments · {shownExperiments.filter((e) => e.promoted).length} promoted
          </span>
        }
      >
        {(campaignData?.campaigns?.length ?? 0) > 1 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {['', ...(campaignData?.campaigns ?? []).map((c) => c.name)].map((name) => (
              <button
                key={name || 'all'}
                type="button"
                onClick={() => setCampaignOnly(name)}
                className={cn(
                  'rounded-[4px] border px-2.5 py-1 mono-caps transition-colors',
                  name === campaignOnly
                    ? 'border-amber-line bg-amber-soft text-amber'
                    : 'border-line text-muted hover:border-line-2',
                )}
              >
                {name || 'all'}
              </button>
            ))}
          </div>
        )}
        {shownExperiments.length === 0 ? (
          <EmptyState>Nothing gated yet.</EmptyState>
        ) : (
          <div className="flex flex-col gap-2">
            {shownExperiments.map((exp) => (
              <ExperimentCard key={`${exp.campaign}-${exp.label}-${exp.candidate_version}`} exp={exp} />
            ))}
          </div>
        )}
      </Panel>

      <Panel
        testId="experiments-promote"
        title="Promote"
        endpoint="POST /admin/registry/promote"
        note="promotion needs accuracy ≥ champion and zero pass→fail flips; the comparator, not this form, decides"
      >
        <WriteGate enabled={canPromote} reason={writeReason} testId="promote-gate">
          <label className="mono-caps flex items-center gap-1.5">
            version
            <select
              value={selectedTarget}
              onChange={(e) => setTarget(e.target.value)}
              data-testid="promote-version"
              className="rounded-[4px] border border-line-2 bg-panel-2 px-2 py-1 font-mono text-[11px] text-text disabled:cursor-not-allowed"
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
            data-testid="promote-submit"
            onClick={() => promote.mutate(selectedTarget)}
            className="rounded-[4px] border border-amber-line bg-amber-soft px-2.5 py-1 font-mono text-[11px] text-amber hover:bg-amber hover:text-amber-ink disabled:cursor-not-allowed disabled:hover:bg-amber-soft disabled:hover:text-amber"
          >
            {promote.isPending ? 'promoting…' : 'Promote to champion'}
          </button>
          <button
            type="button"
            data-testid="challenger-submit"
            onClick={() => challenger.mutate(selectedTarget)}
            className="rounded-[4px] border border-line-2 px-2.5 py-1 font-mono text-[11px] text-muted hover:border-amber-line hover:text-amber disabled:cursor-not-allowed"
          >
            {challenger.isPending ? 'setting…' : 'Set as challenger'}
          </button>
        </WriteGate>

        {writeError && (
          <p className="type-small mt-2 rounded-[4px] border border-bad px-2 py-1.5 text-bad">
            {writeError}
          </p>
        )}
        {writeOk && (
          <p className="type-small mt-2 rounded-[4px] border border-good-line px-2 py-1.5 text-good">
            {writeOk}
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

      <Panel
        testId="experiments-runs"
        title="Runs"
        endpoint="/admin/experiments"
        note={
          experiments.data?.source === 'snapshot'
            ? 'from the committed MLflow snapshot — the demo image carries no tracking store'
            : 'from the live MLflow tracking store'
        }
        right={
          <div className="flex flex-wrap gap-1">
            <button
              type="button"
              onClick={() => setKindFilter('')}
              className={cn(
                'rounded-[4px] border px-1.5 py-0.5 font-mono text-[10px] uppercase',
                kindFilter === ''
                  ? 'border-amber-line bg-amber-soft text-amber'
                  : 'border-line text-faint hover:text-text',
              )}
            >
              all
            </button>
            {kinds.map((kind) => (
              <button
                key={kind}
                type="button"
                onClick={() => setKindFilter(kind)}
                className={cn(
                  'rounded-[4px] border px-1.5 py-0.5 font-mono text-[10px] uppercase',
                  kindFilter === kind
                    ? 'border-amber-line bg-amber-soft text-amber'
                    : 'border-line text-faint hover:text-text',
                )}
              >
                {kind}
              </button>
            ))}
          </div>
        }
      >
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
              minWidth={720}
              initialSorting={[{ id: 'started', desc: true }]}
            />
            {openRun && (
              <div className="mt-2">
                {(() => {
                  const run = runs.find((r) => r.run_id === openRun);
                  return run ? <RunDetail run={run} /> : null;
                })()}
              </div>
            )}
            <p className="type-meta mt-2 text-faint">
              Click a run name to see its params and metrics. GEPA runs remain broken against
              DeepSeek — <code>dspy_lm_kwargs()</code> still hits the thinking-mode 400 — so any
              GEPA row here predates that regression.
            </p>
          </>
        )}
      </Panel>

      <TwoUp>
        <Panel
          testId="experiments-registry"
          title="Registered versions"
          endpoint="/admin/registry"
          note="a version label means nothing when every model is an API; the bundle id is what an answer is attributable to"
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
            />
          )}
        </Panel>

        <Panel
          testId="experiments-holdout"
          title="Held-out accuracy per version"
          endpoint="/admin/experiments"
          to="/admin/evaluations"
          note="the only endpoint that can separate optimizer_train from never_seen"
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
                    <Link
                      to={`/admin/evaluations?version=${row.original.version}`}
                      className="hover:text-amber"
                    >
                      {row.original.version}
                    </Link>
                  ),
                },
                {
                  id: 'holdout',
                  header: 'never-seen',
                  accessorFn: (r) => r.holdout ?? -1,
                  cell: ({ row }) => (
                    <span className="text-good">{formatPercent(row.original.holdout)}</span>
                  ),
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
          />
          <Caveat>
            The two columns are different populations and are never averaged. &ldquo;Held
            out&rdquo; here means <code>data.loader.optimizer_split()</code> — the 309 questions no
            optimizer ever saw — not <code>train_report_ids</code>, which is a different 60/40 split
            that agrees with it on only 78 of 120 conversations.
          </Caveat>
        </Panel>
      </TwoUp>

      <Panel
        testId="experiments-history"
        title="Promotion history"
        endpoint="/admin/registry"
        note="append-only: nothing here is edited or removed, including forced promotions"
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
          />
        )}
      </Panel>
    </AdminPage>
  );
}
