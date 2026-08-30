import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import { Link, useSearchParams } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Sparkline } from '../landing/Sparkline';
import {
  NO_VALUE,
  formatFilingId,
  formatLatency,
  formatPercent,
  formatUsd,
} from '../landing/format';
import { listTraces } from './api';
import { InstrumentTable } from './InstrumentTable';
import {
  EMPTY_TRACE_FILTER,
  absenceReason,
  clip,
  formatCount,
  matchesTraceFilter,
  observedErrorCodes,
  relativeTime,
  sourceNote,
} from './lib';
import type { CorrectnessFilter, TraceFilter } from './lib';
import {
  AdminPage,
  EmptyState,
  ErrorNote,
  Lamp,
  LampRow,
  LoadingRows,
  Panel,
  StatCells,
} from './ui';
import { ak, useDeploymentSource, useProductionMetrics, useTraceStats } from './useAdminData';
import type { MetricsSource, SourceMetrics } from '../../lib/api';
import type { TraceSummary } from '../../types';

/**
 * Turns. Not "observability" — turns.
 *
 * There is no spans tab and no service tab here on purpose. What this system
 * records is one row per answered question with its four stage captures, and a
 * page that offered a span waterfall would be promising a resolution the trace
 * store does not have.
 *
 * The production-metrics strip above the table is grouped by source and never
 * summed, because replay timing is not latency: a turn recorded at 6.7s and
 * played back in 2s is a 6.7s turn, and one blended p50 across `serving`,
 * `demo` and `eval` would be a number describing nothing.
 */

const SOURCE_BLURB: Record<MetricsSource, string> = {
  serving: 'live turns answered by the model in this process',
  demo: 'recorded turns replayed from the demo pack — no model call, no timing of its own',
  eval: 'turns scored by a batch evaluation run, not user traffic',
};

function SourceCard({
  source,
  metrics,
  active,
  onSelect,
}: {
  source: MetricsSource;
  metrics: SourceMetrics | undefined;
  active: boolean;
  onSelect: () => void;
}) {
  const latency = metrics?.latency_ms;
  const cost = metrics?.cost_usd;
  const accuracy = metrics?.accuracy;
  const errors = metrics?.errors;
  const series = metrics?.series ?? [];

  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      data-testid={`source-${source}`}
      className={cn(
        // `flex flex-col` matters: a <button> centres its content box when the
        // grid stretches it taller than its content, which would leave the
        // three source cards with their headings at three different heights.
        'flex min-w-0 flex-col rounded-md border bg-panel p-3 text-left transition-colors',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber',
        active ? 'border-amber-line bg-panel-2' : 'border-line hover:bg-panel-2',
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="mono-caps">{source}</span>
        <span className="type-num text-[12px] text-text">
          {formatCount(metrics?.n_turns ?? 0)} turns
        </span>
      </div>
      <p className="type-meta mt-1 text-faint">{SOURCE_BLURB[source]}</p>

      <div className="mt-2">
        <StatCells
          columns={2}
          cells={[
            {
              label: 'p50 latency',
              value: formatLatency(latency?.p50),
              reason: absenceReason(metrics ?? null, 'latency'),
            },
            {
              label: 'p95',
              value: formatLatency(latency?.p95),
              reason: absenceReason(metrics ?? null, 'latency'),
            },
            {
              label: 'cost / turn',
              value: formatUsd(cost?.per_turn),
              reason: absenceReason(metrics ?? null, 'cost'),
            },
            {
              label: 'accuracy',
              value: formatPercent(accuracy?.accuracy),
              reason: absenceReason(metrics ?? null, 'accuracy'),
              tone: 'good',
            },
          ]}
        />
      </div>

      <div className="mt-auto pt-2">
        <div className="mono-caps mb-0.5">turns per hour · 24 h</div>
        {/*
          A source that has served nothing gets the label, not a flat line at
          the floor. Twenty-four measured zeros are a real observation, but on
          a card whose headline count is also zero the line reads as activity
          rather than as its absence.
        */}
        <Sparkline
          values={metrics && metrics.n_turns > 0 ? series.map((s) => s.n_turns) : []}
          tone="amber"
          emptyLabel="nothing served in this window"
        />
      </div>

      {errors && errors.n_errors > 0 && (
        <p className="type-meta mt-1.5 text-bad">
          {formatCount(errors.n_errors)} errors ({formatPercent(errors.error_rate)}) ·{' '}
          {Object.entries(errors.by_code)
            .filter(([, n]) => n > 0)
            .map(([code, n]) => `${code} ${n}`)
            .join(' · ')}
        </p>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------

export default function Traces() {
  const [params, setParams] = useSearchParams();
  const deploymentSource = useDeploymentSource();
  const metricsQuery = useProductionMetrics();
  const stats = useTraceStats();

  const [filter, setFilter] = useState<TraceFilter>({
    ...EMPTY_TRACE_FILTER,
    source: params.get('source') ?? '',
    errorCode: params.get('error') ?? '',
    sessionId: params.get('session') ?? '',
    reportId: params.get('report') ?? '',
  });
  const [limit, setLimit] = useState(300);

  function update(patch: Partial<TraceFilter>) {
    const next = { ...filter, ...patch };
    setFilter(next);
    const q = new URLSearchParams();
    if (next.source) q.set('source', next.source);
    if (next.errorCode) q.set('error', next.errorCode);
    if (next.sessionId) q.set('session', next.sessionId);
    if (next.reportId) q.set('report', next.reportId);
    setParams(q, { replace: true });
  }

  // `source`, `session_id` and `report_id` are filtered by the server so the
  // page does not download 500 rows to throw 490 away; correctness and error
  // code have no server-side parameter and are applied here.
  const traces = useQuery({
    queryKey: ak.traceList(filter.source, filter.reportId, filter.sessionId, limit),
    queryFn: () =>
      listTraces({
        source: filter.source,
        reportId: filter.reportId,
        sessionId: filter.sessionId,
        limit,
      }),
    staleTime: 15_000,
  });

  const rows = useMemo(
    () => (traces.data ?? []).filter((row) => matchesTraceFilter(row, filter)),
    [traces.data, filter],
  );

  const errorCodes = useMemo(() => observedErrorCodes(traces.data ?? []), [traces.data]);

  const columns = useMemo<Array<ColumnDef<TraceSummary, unknown>>>(
    () => [
      {
        id: 'when',
        header: 'when',
        accessorFn: (r) => r.created_at,
        meta: { align: 'left', width: '86px' },
        cell: ({ row }) => (
          <Link
            to={`/admin/traces/${row.original.trace_id}`}
            className="hover:text-amber"
            title={row.original.created_at}
          >
            {relativeTime(row.original.created_at)}
          </Link>
        ),
      },
      {
        id: 'source',
        header: 'source',
        accessorFn: (r) => r.source,
        meta: { align: 'left', width: '62px' },
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
        id: 'filing',
        header: 'filing · turn',
        accessorFn: (r) => r.report_id,
        meta: { align: 'left', mono: false, width: '146px' },
        cell: ({ row }) => (
          <Link
            to={`/admin/traces/${row.original.trace_id}`}
            className="hover:text-amber"
            title={row.original.report_id}
          >
            {formatFilingId(row.original.report_id)}
            <span className="text-faint"> · {row.original.turn_index}</span>
          </Link>
        ),
      },
      {
        id: 'question',
        header: 'question',
        accessorFn: (r) => r.question,
        meta: { align: 'left', mono: false, wrap: true, width: '220px' },
        cell: ({ row }) => <span className="text-muted">{clip(row.original.question, 110)}</span>,
      },
      {
        id: 'answer',
        header: 'answer',
        accessorFn: (r) => r.answer ?? '',
        meta: { width: '84px' },
        cell: ({ row }) => <span className="text-text">{row.original.answer || NO_VALUE}</span>,
      },
      {
        id: 'gold',
        header: 'gold',
        accessorFn: (r) => r.gold_answer ?? '',
        meta: { width: '84px' },
        cell: ({ row }) => {
          const { gold_answer: gold, correct } = row.original;
          if (!gold) return <span className="text-faint" title="no gold answer for this turn">—</span>;
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
        cell: ({ row }) => (
          <span className={row.original.latency_ms === null ? 'text-faint' : undefined}>
            {formatLatency(row.original.latency_ms)}
          </span>
        ),
      },
      {
        id: 'tokens',
        header: 'tokens',
        accessorFn: (r) => r.total_tokens ?? -1,
        cell: ({ row }) => (
          <span className={row.original.total_tokens === null ? 'text-faint' : undefined}>
            {formatCount(row.original.total_tokens)}
          </span>
        ),
      },
      {
        id: 'cost',
        header: '$',
        accessorFn: (r) => r.cost_usd ?? -1,
        cell: ({ row }) => (
          <span className={row.original.cost_usd == null ? 'text-faint' : undefined}>
            {formatUsd(row.original.cost_usd)}
          </span>
        ),
      },
      {
        id: 'error',
        header: 'error',
        accessorFn: (r) => r.error_code ?? '',
        meta: { align: 'left', width: '104px' },
        cell: ({ row }) => {
          const code = row.original.error_code || (row.original.error ? 'unknown' : '');
          if (!code) return <span className="text-faint">—</span>;
          return (
            <span className="text-bad" title={row.original.error ?? undefined}>
              {code}
            </span>
          );
        },
      },
    ],
    [],
  );

  const captureEnabled = metricsQuery.data?.trace_capture_enabled ?? true;

  return (
    <AdminPage
      testId="admin-traces"
      eyebrow="admin · traces"
      title="Traces"
      sub="One row per answered turn, with the four stage captures behind it. Grouped by source, never blended — a replayed turn's timing is not latency."
    >
      <LampRow>
        <Lamp
          label="capture"
          value={captureEnabled ? 'on' : 'off'}
          tone={captureEnabled ? 'good' : 'bad'}
          dashed={!captureEnabled}
          title={
            captureEnabled
              ? 'Every turn is written to the trace store.'
              : 'Trace capture is disabled on this deployment, so this page can only be empty.'
          }
        />
        <Lamp
          label="turns"
          value={formatCount(stats.data?.n_turns ?? 0)}
          tone="info"
          to="/admin/traces"
        />
        <Lamp
          label="filings"
          value={formatCount(stats.data?.n_reports ?? 0)}
          tone="idle"
          dashed
        />
        <Lamp
          label="this deployment"
          value={deploymentSource}
          tone={deploymentSource === 'demo' ? 'amber' : 'good'}
          dashed={deploymentSource === 'demo'}
          title={SOURCE_BLURB[deploymentSource]}
        />
      </LampRow>

      <Panel
        testId="traces-metrics"
        title="Production metrics, by source"
        endpoint="/metrics/production"
        note={sourceNote(deploymentSource, metricsQuery.data?.generated_at)}
      >
        {metricsQuery.isLoading ? (
          <LoadingRows rows={4} />
        ) : metricsQuery.data === null ? (
          <EmptyState>
            This backend does not serve <code>/metrics/production</code>.
          </EmptyState>
        ) : (
          <div className="grid grid-cols-1 gap-2 lg:grid-cols-3">
            {(['serving', 'demo', 'eval'] as MetricsSource[]).map((source) => (
              <SourceCard
                key={source}
                source={source}
                metrics={metricsQuery.data?.sources?.[source]}
                active={filter.source === source}
                onSelect={() => update({ source: filter.source === source ? '' : source })}
              />
            ))}
          </div>
        )}
        <p className="type-meta mt-2 text-faint">
          Click a source to filter the table below. Latency, tokens and cost are{' '}
          <code>null</code>—not zero—wherever nothing was metered; the demo pack carries no timing
          at all, so its cards read as em dashes by design rather than by failure.
        </p>
      </Panel>

      <Panel
        testId="traces-table"
        title="Turns"
        endpoint="/traces"
        note={
          traces.data
            ? `${formatCount(rows.length)} of ${formatCount(traces.data.length)} fetched · newest first`
            : undefined
        }
        right={
          <div className="flex flex-wrap items-center gap-1.5">
            <select
              value={filter.source}
              onChange={(e) => update({ source: e.target.value })}
              aria-label="Source"
              className="rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-text"
            >
              <option value="">all sources</option>
              <option value="serving">serving</option>
              <option value="demo">demo</option>
              <option value="eval">eval</option>
            </select>
            <select
              value={filter.correctness}
              onChange={(e) => update({ correctness: e.target.value as CorrectnessFilter })}
              aria-label="Correctness"
              className="rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-text"
            >
              <option value="all">any result</option>
              <option value="correct">correct</option>
              <option value="incorrect">incorrect</option>
              <option value="unscored">no gold</option>
            </select>
            <select
              value={filter.errorCode}
              onChange={(e) => update({ errorCode: e.target.value })}
              aria-label="Error code"
              className="rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-text"
            >
              <option value="">any error state</option>
              <option value="any">errors only</option>
              {errorCodes.map((code) => (
                <option key={code} value={code}>
                  {code}
                </option>
              ))}
            </select>
            <input
              value={filter.reportId}
              onChange={(e) => update({ reportId: e.target.value })}
              placeholder="report id"
              aria-label="Report id"
              className="w-28 rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-text placeholder:text-faint"
            />
            <input
              value={filter.sessionId}
              onChange={(e) => update({ sessionId: e.target.value })}
              placeholder="session id"
              aria-label="Session id"
              className="w-28 rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-text placeholder:text-faint"
            />
            <input
              value={filter.q}
              onChange={(e) => update({ q: e.target.value })}
              placeholder="text"
              aria-label="Search question and answer text"
              className="w-24 rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-text placeholder:text-faint"
            />
          </div>
        }
      >
        {traces.isLoading ? (
          <LoadingRows rows={6} />
        ) : traces.error ? (
          <ErrorNote error={traces.error} />
        ) : rows.length === 0 ? (
          <EmptyState>
            {traces.data && traces.data.length > 0
              ? 'No turn matches these filters.'
              : captureEnabled
                ? 'No turns captured yet. A fresh deployment is empty here until someone asks a question.'
                : 'Trace capture is disabled on this deployment, so nothing is recorded.'}
          </EmptyState>
        ) : (
          <>
            <InstrumentTable
              data={rows}
              columns={columns}
              rowKey={(r) => r.trace_id}
              rowClass={(r) => (r.error ? 'bg-bad/5' : undefined)}
              minWidth={980}
              maxHeight={560}
            />
            {traces.data && traces.data.length >= limit && (
              <button
                type="button"
                onClick={() => setLimit((n) => Math.min(n + 200, 500))}
                className="mono-caps mt-2 w-full rounded-[4px] border border-line py-1.5 hover:border-amber-line hover:text-amber"
              >
                fetch more · the server returns at most 500 rows a page
              </button>
            )}
          </>
        )}
      </Panel>
    </AdminPage>
  );
}
