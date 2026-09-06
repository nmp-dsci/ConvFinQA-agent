import { useMemo } from 'react';
import { useQueries, useQuery } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import { Link } from 'react-router-dom';
import { HudTile } from '../landing/HudTile';
import {
  NO_VALUE,
  formatFilingId,
  formatLatency,
  formatPercent,
  formatPointsDelta,
  formatUsd,
} from '../landing/format';
import { getCampaigns, listTraces } from './api';
import { CHAMPION_ROW, InstrumentTable } from './InstrumentTable';
import {
  PROG_ACC_CAVEAT,
  absenceReason,
  formatCount,
  relativeTime,
  sourceNote,
} from './lib';
import type { VersionRow } from './lib';
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
} from './ui';
import {
  ak,
  useDeploymentSource,
  useProductionMetrics,
  useRegistry,
  useResearchStatus,
  useTraceStats,
  useVersionRows,
} from './useAdminData';
import { compareVersions } from '../../lib/api';
import { qk } from '../../lib/queryClient';
import { useMode } from '../../modeStore';
import type { ComparisonResult, TraceSummary } from '../../types';

/**
 * The page an engineer opens first.
 *
 * Six lamps for the facts that decide how to read everything else, four HUD
 * tiles, and the two tables the mock calls out: the gate result per bundle, and
 * the latest turns with their cost. Nothing here is a dead end — every tile is
 * a link, every version cell opens the comparison that produced it, and every
 * turn row opens its own trace.
 */

// ---------------------------------------------------------------------------
// The bundle table
// ---------------------------------------------------------------------------

interface BundleRow extends VersionRow {
  /** The comparator's verdict against the champion, when there is one. */
  comparison: ComparisonResult | undefined;
  comparisonPending: boolean;
}

function GateCell({ row }: { row: BundleRow }) {
  if (row.isChampion) {
    return <Verdict ok={true}>champion</Verdict>;
  }
  if (row.comparisonPending) {
    return <span className="text-faint">checking…</span>;
  }
  if (!row.comparison) {
    return <span className="text-faint">{NO_VALUE}</span>;
  }
  return (
    <Link to={`/admin/evaluations?candidate=${row.version}&flips=open`}>
      <Verdict ok={row.comparison.promotable}>
        {row.comparison.promotable ? 'pass' : 'refused'}
      </Verdict>
    </Link>
  );
}

function FlipsCell({ row }: { row: BundleRow }) {
  if (row.isChampion || !row.comparison) {
    return <span className="text-faint">{NO_VALUE}</span>;
  }
  const { improvements, regressions } = row.comparison;
  return (
    <Link
      to={`/admin/evaluations?candidate=${row.version}&flips=open`}
      className="hover:text-amber"
      title="Open the flips drawer for this pair"
    >
      <span className="text-good">+{improvements.length}</span>
      <span className="text-faint"> / </span>
      <span className="text-bad">−{regressions.length}</span>
    </Link>
  );
}

// ---------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------

export default function AdminOverview() {
  const health = useMode((s) => s.health);
  const isDemo = health?.mode === 'demo';
  const { rows, champion, isLoading, error, experimentsSource } = useVersionRows();
  const registry = useRegistry();
  const research = useResearchStatus();
  const stats = useTraceStats();
  const metricsQuery = useProductionMetrics();
  const source = useDeploymentSource();
  const metrics = metricsQuery.data?.sources?.[source] ?? null;

  // The newest version that is not the champion is the live promotion
  // question: is something waiting, and did the gate refuse it?
  const challengerName = rows.filter((r) => !r.isChampion).slice(-1)[0]?.version;

  // Every other version is compared too, so the gate column is a verdict per
  // bundle rather than a verdict on the newest one and dashes for the rest.
  // A row whose flips a reader cannot open is a row they have to take on faith.
  const others = rows.filter((r) => !r.isChampion).map((r) => r.version);
  const comparisons = useQueries({
    queries: others.map((version) => ({
      queryKey: qk.compare(champion ?? '', version),
      queryFn: () => compareVersions(champion as string, version),
      enabled: Boolean(champion),
    })),
  });
  const gate = comparisons[others.indexOf(challengerName ?? '')] ?? {
    data: undefined,
    isLoading: false,
  };

  // The same key the Campaigns and Runtimes pages use, so one fetch serves all
  // three and the overview cannot show a different champion from them.
  const story = useQuery({
    queryKey: ['eval-campaigns'],
    queryFn: () => getCampaigns(),
    staleTime: 60_000,
  });
  const runtime = story.data?.runtime_comparison ?? null;
  const runtimeGate = runtime?.gate ?? null;
  const swap = story.data?.sdk_model_comparison ?? null;
  const swapArm = swap?.models?.find((m) => m.model !== swap.reference_model) ?? null;
  const swapPair = swap?.pairs?.[0] ?? null;
  const track = story.data?.champion_track ?? [];
  const trackFirst = track[0] ?? null;
  const short = (model: string | null | undefined) =>
    (model ?? '').replace(/^claude-/, '').replace(/-\d{8}$/, '');
  const runtimeCells = [
    {
      label: `${trackFirst?.version ?? 'pipeline'} · raw pipeline`,
      value: formatPercent(trackFirst?.accuracy),
      reason: 'no champion track recorded',
    },
    {
      label: `${story.data?.champion ?? 'champion'} · optimised pipeline`,
      value: formatPercent(runtime?.pipeline?.accuracy ?? story.data?.champion_accuracy),
      reason: 'the champion has no gate run',
    },
    {
      label: `${runtime?.agent_sdk?.version ?? 'sdk'} · one session · ${short(runtime?.agent_sdk?.model) || 'claude'}`,
      value: formatPercent(runtime?.agent_sdk?.accuracy),
      reason: 'the single-session arm has not been run',
      tone: (runtimeGate?.promoted ? 'good' : 'plain') as 'good' | 'plain',
    },
    {
      label: `same prompt · ${short(swapArm?.model) || 'second model'}`,
      value: formatPercent(swapArm?.accuracy),
      reason: 'the sdk champion has been scored on one model only',
      tone: (swapPair?.significant && (swapPair?.delta_pp ?? 0) < 0 ? 'bad' : 'plain') as
        | 'bad'
        | 'plain',
    },
  ];

  const traces = useQuery({
    queryKey: ak.traceList('', '', '', 8),
    queryFn: () => listTraces({ limit: 8 }),
    staleTime: 15_000,
  });

  const championRow = rows.find((r) => r.isChampion) ?? rows[rows.length - 1];

  const bundleRows: BundleRow[] = useMemo(
    () =>
      rows.map((row) => {
        const index = others.indexOf(row.version);
        return {
          ...row,
          comparison: index >= 0 ? comparisons[index]?.data : undefined,
          comparisonPending: index >= 0 ? (comparisons[index]?.isLoading ?? false) : false,
        };
      }),
    // `comparisons` is a fresh array each render; its data is what matters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rows, others.join(','), comparisons.map((c) => c.status).join(',')],
  );

  const bundleColumns = useMemo<Array<ColumnDef<BundleRow, unknown>>>(
    () => [
      {
        id: 'bundle',
        header: 'bundle',
        accessorFn: (r) => r.version,
        meta: { align: 'left', mono: false, width: '108px' },
        cell: ({ row }) => (
          <Link to="/admin/experiments" className="hover:text-amber">
            {row.original.version}
            {row.original.isChampion && <span className="text-faint"> · champion</span>}
          </Link>
        ),
      },
      {
        id: 'holdout',
        header: 'never-seen',
        accessorFn: (r) => r.holdout ?? -1,
        cell: ({ row }) => (
          <Link
            to={`/admin/evaluations?version=${row.original.version}`}
            className="hover:text-amber"
            title={`${row.original.holdoutN ?? 0} questions no optimizer ever saw`}
          >
            {formatPercent(row.original.holdout)}
          </Link>
        ),
      },
      {
        id: 'overall',
        header: 'overall',
        accessorFn: (r) => r.overall ?? -1,
        cell: ({ row }) => (
          <Link
            to={`/admin/evaluations?version=${row.original.version}`}
            className="hover:text-amber"
            title={`${row.original.nQuestions ?? 0} questions, seen and unseen mixed`}
          >
            {formatPercent(row.original.overall)}
          </Link>
        ),
      },
      {
        id: 'prog',
        header: 'program',
        accessorFn: (r) => r.progAcc ?? -1,
        cell: ({ row }) => (
          <Link
            to="/admin/system"
            className="hover:text-amber"
            title="Program accuracy — see the note below the table"
          >
            {formatPercent(row.original.progAcc)}
          </Link>
        ),
      },
      {
        id: 'flips',
        header: 'flips',
        enableSorting: false,
        cell: ({ row }) => <FlipsCell row={row.original} />,
      },
      {
        id: 'gate',
        header: 'gate',
        enableSorting: false,
        cell: ({ row }) => <GateCell row={row.original} />,
      },
    ],
    [],
  );

  const turnColumns = useMemo<Array<ColumnDef<TraceSummary, unknown>>>(
    () => [
      {
        id: 'turn',
        header: 'filing · turn',
        accessorFn: (r) => r.report_id,
        meta: { align: 'left', mono: false, width: '150px' },
        cell: ({ row }) => (
          <Link to={`/admin/traces/${row.original.trace_id}`} className="hover:text-amber">
            {formatFilingId(row.original.report_id)}
            <span className="text-faint"> · {row.original.turn_index}</span>
          </Link>
        ),
      },
      {
        // The tiles above describe one source; this table shows whatever was
        // answered most recently. Without this column a demo deployment reading
        // a dev trace store would present live-serving timings under a set of
        // demo tiles and never say which was which.
        id: 'source',
        header: 'src',
        accessorFn: (r) => r.source,
        meta: { align: 'left', width: '54px' },
        cell: ({ row }) => (
          <span
            className={
              row.original.source === 'demo'
                ? 'text-amber'
                : row.original.source === 'eval'
                  ? 'text-violet'
                  : 'text-info'
            }
          >
            {row.original.source}
          </span>
        ),
      },
      {
        id: 'answer',
        header: 'answer',
        accessorFn: (r) => r.answer ?? '',
        meta: { width: '84px' },
        cell: ({ row }) => (
          <span className="text-text">{row.original.answer || NO_VALUE}</span>
        ),
      },
      {
        id: 'gold',
        header: 'gold',
        accessorFn: (r) => r.gold_answer ?? '',
        meta: { width: '84px' },
        cell: ({ row }) => {
          const { gold_answer: gold, correct } = row.original;
          if (!gold) return <span className="text-faint">{NO_VALUE}</span>;
          return (
            <span className={correct ? 'text-good' : 'text-bad'}>
              {gold} {correct ? '✓' : '✗'}
            </span>
          );
        },
      },
      {
        id: 'latency',
        header: 'ms',
        accessorFn: (r) => r.latency_ms ?? -1,
        cell: ({ row }) => formatLatency(row.original.latency_ms),
      },
      {
        id: 'cost',
        header: '$',
        accessorFn: (r) => r.cost_usd ?? -1,
        cell: ({ row }) => formatUsd(row.original.cost_usd),
      },
      {
        id: 'when',
        header: 'when',
        accessorFn: (r) => r.created_at,
        cell: ({ row }) => (
          <span className="text-faint">{relativeTime(row.original.created_at)}</span>
        ),
      },
    ],
    [],
  );

  // ---------------------------------------------------------------------------

  const latency = metrics?.latency_ms ?? null;
  const cost = metrics?.cost_usd ?? null;
  const errors = metrics?.errors ?? null;
  const errorSeries = metrics?.series.map((s) => (s.n_turns > 0 ? s.n_errors : null)) ?? [];

  const researchLamp = (() => {
    if (research.isLoading) return { value: 'reading…', tone: 'idle' as const };
    if (research.data?.busy) return { value: 'running', tone: 'good' as const };
    return { value: 'idle', tone: 'idle' as const };
  })();

  return (
    <AdminPage
      testId="admin-overview"
      eyebrow="admin · overview"
      title="Overview"
      sub={
        <>
          Champion, gate, and what production has been doing.{' '}
          {isDemo
            ? 'Read-only in the demo — every write is a disabled control, and the server refuses the request anyway.'
            : 'This deployment holds a key and answers with the model.'}
        </>
      }
    >
      <LampRow>
        <Lamp
          label="mode"
          value={isDemo ? 'demo · replay' : 'live'}
          tone={isDemo ? 'amber' : 'good'}
          dashed={isDemo}
          to="/admin/system"
          title={
            isDemo
              ? 'No API key on this deployment. Chat replays turns recorded in development.'
              : `Live against ${health?.bundle.lm_mini ?? 'the champion model'}.`
          }
        />
        <Lamp
          label="champion"
          value={champion ?? 'unset'}
          tone="info"
          to="/admin/experiments"
          title={health ? `bundle ${health.bundle_id}` : undefined}
        />
        <Lamp
          label="gate"
          value={
            !challengerName
              ? 'no challenger'
              : gate.isLoading
                ? 'checking…'
                : gate.data?.promotable
                  ? `${challengerName} pass`
                  : `${challengerName} refused`
          }
          tone={!challengerName ? 'idle' : gate.data?.promotable ? 'good' : 'bad'}
          dashed={!challengerName}
          to={`/admin/evaluations?candidate=${challengerName ?? ''}&flips=open`}
          title={gate.data?.reason}
        />
        <Lamp
          label="keys"
          value={isDemo ? 'none' : 'configured'}
          tone={isDemo ? 'good' : 'info'}
          to="/admin/system"
          title={
            isDemo
              ? 'The demo container holds no API key at all — it cannot make a billable call.'
              : 'This process can construct a model through llm.py.'
          }
        />
        <Lamp
          label="research"
          value={researchLamp.value}
          tone={researchLamp.tone}
          dashed={!research.data?.busy}
          to="/admin/research"
        />
        <Lamp
          label="traces"
          value={stats.data ? formatCount(stats.data.n_turns) : '…'}
          tone={stats.data?.n_turns ? 'info' : 'idle'}
          dashed={!stats.data?.n_turns}
          to="/admin/traces"
          title={`${stats.data?.n_reports ?? 0} filings covered`}
        />
      </LampRow>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-4">
        <HudTile
          label="never-seen accuracy"
          value={formatPercent(championRow?.holdout)}
          meta={
            <>
              {champion ?? 'champion'} · {formatCount(championRow?.holdoutN ?? null)} questions no
              optimizer saw
            </>
          }
          reason="no holdout figure in the experiments payload"
          tone="good"
          loading={isLoading}
          to={`/admin/evaluations?version=${championRow?.version ?? ''}`}
          drill="/admin/evaluations"
        />
        <HudTile
          label="p50 latency"
          value={formatLatency(latency?.p50)}
          meta={<>p95 {formatLatency(latency?.p95)} · {formatCount(latency?.n_measured ?? 0)} measured</>}
          reason={absenceReason(metrics, 'latency')}
          loading={metricsQuery.isLoading}
          series={metrics?.series.map((s) => s.p50_latency_ms)}
          to="/admin/traces"
          drill="/admin/traces"
        />
        <HudTile
          label="cost per turn"
          value={formatUsd(cost?.per_turn)}
          meta={<>{formatUsd(cost?.total ?? null)} over {formatCount(metrics?.n_turns ?? 0)} turns</>}
          reason={absenceReason(metrics, 'cost')}
          loading={metricsQuery.isLoading}
          series={metrics?.series.map((s) => (s.n_turns > 0 ? s.cost_usd : null))}
          to="/admin/traces"
          drill="/admin/traces"
        />
        <HudTile
          label="errors"
          value={metrics ? formatCount(errors?.n_errors ?? 0) : NO_VALUE}
          meta={
            <>
              {formatPercent(errors?.error_rate, 1)} of {formatCount(metrics?.n_turns ?? 0)} turns
              served
            </>
          }
          reason={absenceReason(metrics, 'accuracy')}
          tone={errors && errors.n_errors > 0 ? 'bad' : 'plain'}
          loading={metricsQuery.isLoading}
          series={errorSeries}
          to="/admin/traces?error=any"
          drill="/admin/traces"
        />
      </div>

      <p className="type-meta text-faint">{sourceNote(source, metricsQuery.data?.generated_at)}</p>

      <Panel
        testId="overview-runtimes"
        title="Runtime decision"
        endpoint="/eval/campaigns"
        to="/admin/runtimes"
        note="the champion track, the single-session challenger and the same prompt on a second model — one gate split, one evaluator"
      >
        {story.isLoading ? (
          <LoadingRows rows={1} />
        ) : (
          <>
            <StatCells columns={4} cells={runtimeCells} />
            <p className="type-meta mt-2 text-faint">
              {runtimeGate?.promoted
                ? `Paired on the gate split: ${formatPointsDelta((runtimeGate.delta_pp ?? 0) / 100)} for the single session over ${story.data?.champion ?? 'the champion'}, one-sided clustered p ${runtimeGate.p_value ?? '—'} — the recommendation on /admin/runtimes is to move the runtime. Serving still runs the four-agent champion.`
                : 'No cross-runtime gate recorded yet — the single-session arm appears here once it has been run and gated on the gate split.'}
            </p>
          </>
        )}
      </Panel>

      {error ? <ErrorNote error={error} /> : null}

      <TwoUp>
        <Panel
          testId="overview-bundles"
          title="Champion vs challengers"
          endpoint="/admin/experiments"
          to="/admin/experiments"
          note={
            experimentsSource === 'snapshot'
              ? 'read from the committed MLflow snapshot, not a live tracking store'
              : 'read from the live MLflow tracking store'
          }
        >
          {isLoading ? (
            <LoadingRows />
          ) : (
            <>
              <InstrumentTable
                data={bundleRows}
                columns={bundleColumns}
                rowKey={(r) => r.version}
                rowClass={(r) => (r.isChampion ? CHAMPION_ROW : undefined)}
                minWidth={520}
                emptyLabel="no version has a committed predictions CSV"
              />
              <Caveat>
                These rows are the legacy 770-question scoring (v1–v3_1). Every version from v4 on
                was promoted by the eval loop on the 349-question gate split and has no legacy CSV,
                so the current champion is not a row here — its evidence is on{' '}
                <Link to="/admin/campaigns" className="text-amber underline-offset-2 hover:underline">
                  Campaigns
                </Link>{' '}
                and the runtime decision above.{' '}
                <strong className="text-muted">never-seen</strong> and{' '}
                <strong className="text-muted">overall</strong> are two populations, never averaged:
                overall mixes the 461 questions the optimizer trained on with the 309 it never saw.{' '}
                {PROG_ACC_CAVEAT}
              </Caveat>
            </>
          )}
        </Panel>

        <Panel
          testId="overview-turns"
          title="Recent turns"
          endpoint="/admin/traces"
          to="/admin/traces"
          note={
            traces.data?.length
              ? `newest ${traces.data.length} of ${formatCount(stats.data?.n_turns ?? 0)} captured turns`
              : undefined
          }
        >
          {traces.isLoading ? (
            <LoadingRows />
          ) : traces.error ? (
            <ErrorNote error={traces.error} />
          ) : traces.data && traces.data.length > 0 ? (
            <InstrumentTable
              data={traces.data}
              columns={turnColumns}
              rowKey={(r) => r.trace_id}
              rowClass={(r) => (r.error ? 'bg-bad/5' : undefined)}
              minWidth={620}
            />
          ) : (
            <EmptyState>
              No turns captured yet. This is the normal state of a fresh deployment — the trace
              store fills as questions are asked.
            </EmptyState>
          )}
        </Panel>
      </TwoUp>

      <Panel
        title="Registry"
        endpoint="/admin/registry"
        to="/admin/experiments"
        note="aliases decide what serves; the promotion history is append-only"
      >
        {registry.isLoading ? (
          <LoadingRows rows={2} />
        ) : registry.error ? (
          <ErrorNote error={registry.error} />
        ) : (
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            {Object.entries(registry.data?.aliases ?? {}).map(([alias, version]) => (
              <span key={alias} className="flex items-baseline gap-1.5">
                <span className="mono-caps">{alias}</span>
                <span className="type-num text-[12px] text-text">{version}</span>
              </span>
            ))}
            <span className="flex items-baseline gap-1.5">
              <span className="mono-caps">registered</span>
              <span className="type-num text-[12px] text-text">
                {formatCount(registry.data?.versions.length ?? 0)}
              </span>
            </span>
            <span className="flex items-baseline gap-1.5">
              <span className="mono-caps">promotions</span>
              <Link
                to="/admin/experiments"
                className="type-num text-[12px] text-text hover:text-amber"
              >
                {formatCount(registry.data?.history.length ?? 0)}
              </Link>
            </span>
            <span className="flex items-baseline gap-1.5">
              <span className="mono-caps">writes</span>
              <span className="type-num text-[12px] text-muted">
                {registry.data?.can_promote ? 'enabled' : 'refused'}
              </span>
            </span>
          </div>
        )}
      </Panel>
    </AdminPage>
  );
}
