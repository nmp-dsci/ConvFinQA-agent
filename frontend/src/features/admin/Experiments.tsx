import { useMemo, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { formatPercent } from '../landing/format';
import { CHAMPION_ROW, InstrumentTable } from './InstrumentTable';
import {
  bundleLine,
  clip,
  formatCount,
  formatEpochMs,
  formatRunDuration,
  formatStamp,
} from './lib';
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
 * Experiments: what was run, what is registered, and what was promoted.
 *
 * The promotion history is append-only and is presented that way — a list of
 * events with the actor and the comparator's reason attached, not a "current
 * state" that quietly forgets how it got there. The promote control is the one
 * write on this page and is wrapped in a real disabled fieldset with the reason
 * printed beside it; the server refuses the same call independently.
 */

const KIND_TONE: Record<string, string> = {
  eval: 'text-info',
  gepa: 'text-violet',
  s7: 'text-amber',
};

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
      sub="Every run that produced a version, the registry that decides which one serves, and the append-only record of every promotion."
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
          <Lamp key={alias} label={alias} value={version} tone="info" />
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
