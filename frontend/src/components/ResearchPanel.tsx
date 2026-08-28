import { useCallback, useEffect, useRef, useState } from 'react';
import * as api from '../api';
import { useIsDemo } from '../modeStore';
import type { ResearchStatus, RulesPayload } from '../types';
import {
  Badge,
  DemoGate,
  EmptyState,
  ErrorNote,
  Mono,
  Panel,
  ScrollX,
  Spinner,
  formatTime,
} from './ui';

function RulesBrowser() {
  const [payload, setPayload] = useState<RulesPayload | null>(null);
  const [variants, setVariants] = useState<string[]>([]);
  const [variant, setVariant] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getRules(variant).then(setPayload).catch((e) => setError(String(e)));
  }, [variant]);

  useEffect(() => {
    api
      .getRules()
      .then((p) => setVariant((v) => v || p.variant))
      .catch(() => undefined);
    fetch(`${api.getApiBase()}/admin/rules/variants`)
      .then((r) => r.json())
      .then(setVariants)
      .catch(() => undefined);
  }, []);

  const agents = Object.entries(payload?.agents ?? {});
  const total = agents.reduce((n, [, a]) => n + a.rules.length, 0);

  return (
    <Panel
      title="Rules browser"
      subtitle="What the auto-research harness has actually promoted into each agent's prompt — previously only readable as raw JSONL."
      actions={
        <select
          value={variant}
          onChange={(e) => setVariant(e.target.value)}
          className="bg-panel2 text-sm rounded px-2 py-1 border border-white/10"
        >
          {variants.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      }
    >
      {error && <ErrorNote error={error} />}
      {!payload ? (
        <Spinner />
      ) : total === 0 ? (
        <EmptyState
          title={`No promoted rules in ${payload.variant}`}
          hint="Rules are only promoted when a proposed fix verifiably repairs its case. A round that resolves nothing promotes nothing — which is itself a result worth seeing."
        />
      ) : (
        <div className="p-4 space-y-4">
          {agents.map(([agent, data]) => (
            <div key={agent}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium capitalize">{agent}</span>
                <Badge>{data.rules.length} promoted</Badge>
                <Badge>{data.attempts.length} attempts</Badge>
              </div>
              <ul className="space-y-1">
                {data.rules.map((rule, i) => (
                  <li key={i} className="text-xs bg-bg/40 rounded p-2">
                    <div className="whitespace-pre-wrap">{String(rule.rule ?? '')}</div>
                    {rule.verified_on_case ? (
                      <div className="text-textMuted mt-1">
                        verified on <Mono>{String(rule.verified_on_case)}</Mono>
                      </div>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

/**
 * Research console: launch an auto-research round from the app and watch it run.
 *
 * Visible in both deployments — seeing what the loop is and what it produced is
 * most of its value — but launching is dev-only and owner-gated. The demo shows
 * completed rounds with the controls inert.
 */
export function ResearchPanel() {
  const [status, setStatus] = useState<ResearchStatus | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(5);
  const [retryN, setRetryN] = useState(1);
  const [variant, setVariant] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const logEndRef = useRef<HTMLDivElement | null>(null);
  const isDemo = useIsDemo();

  const refresh = useCallback(async () => {
    try {
      const s = await api.getResearchStatus();
      setStatus(s);
      if (s.current?.log_tail?.length) setLog(s.current.log_tail);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Subscribe to progress. The stream is the same SSE machinery the chat uses,
  // so there is one streaming client in the app rather than two.
  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;
    api
      .streamResearch(controller.signal, (event) => {
        if (event.event === 'log') {
          setLog((lines) => [...lines.slice(-400), String(event.line ?? '')]);
        } else if (event.event === 'job_start' || event.event === 'job_end') {
          void refresh();
        } else if (event.event === 'status') {
          setStatus(event as unknown as ResearchStatus);
        }
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: 'end' });
  }, [log]);

  async function start(kind: string) {
    setError(null);
    setLog([]);
    try {
      await api.startResearch({ kind, limit, retry_n: retryN, variant });
      await refresh();
    } catch (err) {
      setError(err instanceof api.ApiError ? err.message : String(err));
    }
  }

  const busy = status?.busy ?? false;

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {error && <ErrorNote error={error} />}

      <Panel
        title="Research console"
        subtitle="Diagnose failing cases, propose a prompt fix, verify it repairs the case, promote only what verifiably worked. Results land as the next challenger."
        actions={
          isDemo ? <Badge tone="warn">launch disabled in demo</Badge> : null
        }
      >
        <div className="p-4 space-y-4">
          <DemoGate reason="Research rounds run in the dev deployment">
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-textMuted">
                cases
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                  className="block bg-panel2 text-sm rounded px-2 py-1 border border-white/10 w-20 mt-1 text-textMain"
                />
              </label>
              <label className="text-xs text-textMuted">
                attempts per case
                <input
                  type="number"
                  min={1}
                  max={3}
                  value={retryN}
                  onChange={(e) => setRetryN(Number(e.target.value))}
                  className="block bg-panel2 text-sm rounded px-2 py-1 border border-white/10 w-20 mt-1 text-textMain"
                />
              </label>
              <label className="text-xs text-textMuted">
                output variant
                <input
                  value={variant}
                  onChange={(e) => setVariant(e.target.value)}
                  placeholder="v3_2"
                  className="block bg-panel2 text-sm rounded px-2 py-1 border border-white/10 w-28 mt-1 text-textMain"
                />
              </label>
              <button
                type="button"
                disabled={busy}
                onClick={() => void start('s7')}
                className="px-3 py-1.5 text-sm rounded bg-accent2 text-bg font-medium disabled:opacity-40"
              >
                {busy ? 'Round in flight…' : 'Run auto-research round'}
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => void start('gepa_smoke')}
                className="px-3 py-1.5 text-sm rounded bg-panel2 hover:bg-accent disabled:opacity-40"
              >
                GEPA smoke run
              </button>
              {busy && (
                <button
                  type="button"
                  onClick={() => void api.cancelResearch().then(refresh)}
                  className="px-3 py-1.5 text-sm rounded bg-danger/20 text-danger"
                >
                  Cancel
                </button>
              )}
            </div>
          </DemoGate>

          <p className="text-[11px] text-textMuted max-w-2xl">
            One round at a time, deliberately — these saturate the provider and cost real money,
            and a queue would turn an impatient click into four concurrent rounds. Whatever a
            round produces still has to clear the same held-out evaluation and the same
            comparator before it can become champion.
          </p>

          {(busy || log.length > 0) && (
            <div>
              <div className="text-xs uppercase tracking-wide text-textMuted mb-1">Progress</div>
              <pre className="text-[11px] bg-bg/70 rounded p-3 max-h-72 overflow-y-auto whitespace-pre-wrap">
                {log.join('\n') || 'waiting for output…'}
                <div ref={logEndRef} />
              </pre>
            </div>
          )}
        </div>
      </Panel>

      <Panel title="Recent rounds">
        {(status?.history ?? []).length === 0 ? (
          <EmptyState title="No rounds recorded in this process" />
        ) : (
          <ScrollX>
            <table className="w-full text-sm">
              <thead className="text-xs text-textMuted border-b border-white/5">
                <tr className="text-left">
                  <th className="px-3 py-2 font-medium">Job</th>
                  <th className="px-3 py-2 font-medium">Kind</th>
                  <th className="px-3 py-2 font-medium">Started</th>
                  <th className="px-3 py-2 font-medium">Finished</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {(status?.history ?? []).map((job) => (
                  <tr key={job.job_id} className="border-b border-white/5">
                    <td className="px-3 py-2">
                      <Mono>{job.job_id}</Mono>
                    </td>
                    <td className="px-3 py-2">{job.kind}</td>
                    <td className="px-3 py-2 text-xs text-textMuted">
                      {formatTime(job.started_at)}
                    </td>
                    <td className="px-3 py-2 text-xs text-textMuted">
                      {formatTime(job.finished_at)}
                    </td>
                    <td className="px-3 py-2">
                      <Badge
                        tone={
                          job.status === 'succeeded'
                            ? 'good'
                            : job.status === 'failed'
                              ? 'bad'
                              : 'neutral'
                        }
                      >
                        {job.status}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollX>
        )}
      </Panel>

      <RulesBrowser />
    </div>
  );
}
