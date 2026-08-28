import { useCallback, useEffect, useMemo, useState } from 'react';
import * as api from '../api';
import { useIsDemo, useMode } from '../modeStore';
import type { ComparisonResult, ExperimentsPayload, VersionAccuracy } from '../types';
import {
  Badge,
  Delta,
  DemoGate,
  EmptyState,
  ErrorNote,
  Mono,
  Panel,
  Pct,
  ScrollX,
  Spinner,
  formatTime,
} from './ui';

/** Accuracy across versions. Honest about the dip — v3_1 scored below v2. */
function AccuracyTrend({
  versions,
  champion,
}: {
  versions: VersionAccuracy[];
  champion: string | null;
}) {
  if (!versions.length) return null;
  // Scale to the held-out number, which is the one the bars represent.
  const max = Math.max(...versions.map((v) => v.holdout_accuracy || v.accuracy), 0.01);

  return (
    <div className="p-4">
      <div className="flex items-end gap-4 h-40">
        {versions.map((entry) => {
          const isChampion = entry.version === champion;
          const height = Math.max(4, ((entry.holdout_accuracy || entry.accuracy) / max) * 100);
          return (
            <div key={entry.version} className="flex-1 flex flex-col items-center gap-2 min-w-0">
              <div className="text-center">
                <div className="text-xs tabular-nums font-medium">
                  <Pct value={entry.holdout_accuracy} />
                </div>
                <div className="text-[10px] tabular-nums text-textMuted">
                  <Pct value={entry.accuracy} /> all
                </div>
              </div>
              <div className="w-full flex items-end h-full">
                <div
                  className={`w-full rounded-t transition-all ${
                    isChampion ? 'bg-accent2' : 'bg-accent/60'
                  }`}
                  style={{ height: `${height}%` }}
                  title={`${entry.version}: ${(entry.holdout_accuracy * 100).toFixed(2)}% on ${entry.holdout_n_questions} never-seen questions (${(entry.accuracy * 100).toFixed(2)}% over all ${entry.n_questions})`}
                />
              </div>
              <div className="text-xs text-center truncate w-full">
                <Mono>{entry.version}</Mono>
                {isChampion && (
                  <div className="mt-1">
                    <Badge tone="good">champion</Badge>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-[11px] text-textMuted mt-4">
        Bars show accuracy on the conversations no optimizer ever saw; the grey figure beneath
        is the same version across all 770 scored questions, which includes the 120
        conversations GEPA trained on. The champion is not always the newest — v3_1 scored
        below v2 and stays registered with its evidence rather than being deleted.
      </p>
    </div>
  );
}

function ComparisonView({ result }: { result: ComparisonResult }) {
  return (
    <div className="p-4 space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <Badge tone={result.promotable ? 'good' : 'bad'}>
          {result.promotable ? 'promotable' : 'blocked'}
        </Badge>
        <span className="text-textMuted">{result.reason}</span>
      </div>

      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="bg-bg/40 rounded p-3">
          <div className="text-xs text-textMuted">{result.baseline_version}</div>
          <div className="text-lg font-semibold">
            <Pct value={result.baseline_accuracy_all} />
          </div>
        </div>
        <div className="bg-bg/40 rounded p-3">
          <div className="text-xs text-textMuted">{result.candidate_version}</div>
          <div className="text-lg font-semibold">
            <Pct value={result.candidate_accuracy_all} />
          </div>
        </div>
        <div className="bg-bg/40 rounded p-3">
          <div className="text-xs text-textMuted">delta over {result.n_compared} shared questions</div>
          <div className="text-lg font-semibold">
            <Delta value={result.accuracy_delta} />
          </div>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <div className="text-xs uppercase tracking-wide text-danger mb-2">
            Broken — pass → fail ({result.regressions.length})
          </div>
          {result.regressions.length === 0 ? (
            <div className="text-xs text-textMuted">None. This is the condition that gates promotion.</div>
          ) : (
            <ul className="space-y-1 max-h-64 overflow-y-auto">
              {result.regressions.map((flip) => (
                <li key={`${flip.report_id}-${flip.q_order}`} className="text-xs bg-danger/10 rounded p-2">
                  <Mono>
                    {flip.report_id} q{flip.q_order}
                  </Mono>
                  <div className="mt-1 text-textMuted">{flip.question}</div>
                  <div className="mt-1">
                    gold <Mono>{flip.gold_answer}</Mono> · was{' '}
                    <Mono className="text-accent2">{flip.baseline_answer}</Mono> · now{' '}
                    <Mono className="text-danger">{flip.candidate_answer}</Mono>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-accent2 mb-2">
            Fixed — fail → pass ({result.improvements.length})
          </div>
          <ul className="space-y-1 max-h-64 overflow-y-auto">
            {result.improvements.map((flip) => (
              <li key={`${flip.report_id}-${flip.q_order}`} className="text-xs bg-accent/10 rounded p-2">
                <Mono>
                  {flip.report_id} q{flip.q_order}
                </Mono>
                <div className="mt-1 text-textMuted">{flip.question}</div>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {result.notes.map((note) => (
        <p key={note} className="text-[11px] text-textMuted">
          {note}
        </p>
      ))}
    </div>
  );
}

export function ExperimentsPanel() {
  const [payload, setPayload] = useState<ExperimentsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [baseline, setBaseline] = useState('');
  const [candidate, setCandidate] = useState('');
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const isDemo = useIsDemo();
  const ownerToken = useMode((s) => s.ownerToken);
  const setToken = useMode((s) => s.setOwnerToken);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getExperiments();
      setPayload(data);
      setError(null);
      const versions = data.versions.map((v) => v.version);
      if (versions.length >= 2) {
        setBaseline((b) => b || data.registry.aliases.champion || versions[0]);
        setCandidate((c) => c || versions[versions.length - 1]);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const champion = payload?.registry?.aliases?.champion ?? null;
  const challenger = payload?.registry?.aliases?.challenger ?? null;

  const runsByKind = useMemo(() => {
    const groups: Record<string, number> = {};
    for (const run of payload?.runs ?? []) groups[run.kind] = (groups[run.kind] ?? 0) + 1;
    return groups;
  }, [payload]);

  async function runComparison() {
    setNotice(null);
    try {
      setComparison(await api.compareVersions(baseline, candidate));
    } catch (err) {
      setError(String(err));
    }
  }

  async function promote(version: string) {
    setNotice(null);
    try {
      await api.promoteVersion(version);
      setNotice(`${version} promoted to champion.`);
      await load();
    } catch (err) {
      // A refused promotion is the comparator doing its job, not a bug.
      setNotice(err instanceof api.ApiError ? err.message : String(err));
    }
  }

  if (loading && !payload) return <Spinner label="Loading experiments…" />;

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {error && <ErrorNote error={error} />}

      <Panel
        title="Experiments"
        subtitle={
          payload?.source === 'snapshot'
            ? `Read from the committed snapshot${
                payload.exported_at ? ` exported ${formatTime(payload.exported_at)}` : ''
              } — the demo ships history without a tracking server.`
            : 'Read live from the local MLflow store.'
        }
        actions={
          <button
            type="button"
            onClick={() => void load()}
            className="text-sm px-2 py-1 rounded bg-panel2 hover:bg-accent"
          >
            Refresh
          </button>
        }
      >
        <div className="flex flex-wrap gap-6 px-4 py-3 border-b border-white/5">
          {Object.entries(runsByKind).map(([kind, count]) => (
            <div key={kind}>
              <div className="text-xl font-semibold tabular-nums">{count}</div>
              <div className="text-xs text-textMuted">
                {kind === 's7' ? 'auto-research rounds' : `${kind} runs`}
              </div>
            </div>
          ))}
          {champion && (
            <div>
              <div className="text-xl font-semibold">
                <Mono className="text-base">{champion}</Mono>
              </div>
              <div className="text-xs text-textMuted">champion</div>
            </div>
          )}
          {challenger && (
            <div>
              <div className="text-xl font-semibold">
                <Mono className="text-base">{challenger}</Mono>
              </div>
              <div className="text-xs text-textMuted">challenger</div>
            </div>
          )}
        </div>
        <AccuracyTrend versions={payload?.versions ?? []} champion={champion} />
      </Panel>

      <Panel
        title="Compare two versions"
        subtitle="Overall delta plus the per-question flip lists. Promotion needs accuracy ≥ champion AND no pass→fail flips."
        actions={
          <>
            <select
              value={baseline}
              onChange={(e) => setBaseline(e.target.value)}
              className="bg-panel2 text-sm rounded px-2 py-1 border border-white/10"
            >
              {(payload?.versions ?? []).map((v) => (
                <option key={v.version} value={v.version}>
                  {v.version}
                </option>
              ))}
            </select>
            <span className="text-textMuted text-sm">vs</span>
            <select
              value={candidate}
              onChange={(e) => setCandidate(e.target.value)}
              className="bg-panel2 text-sm rounded px-2 py-1 border border-white/10"
            >
              {(payload?.versions ?? []).map((v) => (
                <option key={v.version} value={v.version}>
                  {v.version}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => void runComparison()}
              disabled={!baseline || !candidate || baseline === candidate}
              className="text-sm px-2 py-1 rounded bg-accent2 text-bg disabled:opacity-40"
            >
              Compare
            </button>
          </>
        }
      >
        {comparison ? (
          <ComparisonView result={comparison} />
        ) : (
          <EmptyState title="Pick two versions and compare" />
        )}
      </Panel>

      <Panel
        title="Registry — champion / challenger"
        subtitle="Every registered bundle, never deleted. Promotion is an append-only event."
        actions={
          !isDemo && (
            <input
              type="password"
              value={ownerToken}
              onChange={(e) => setToken(e.target.value)}
              placeholder="owner token"
              className="bg-panel2 text-sm rounded px-2 py-1 border border-white/10 w-40"
            />
          )
        }
      >
        {notice && <div className="mx-4 mt-3 text-xs text-amber-300">{notice}</div>}
        <ScrollX>
          <table className="w-full text-sm">
            <thead className="text-xs text-textMuted border-b border-white/5">
              <tr className="text-left">
                <th className="px-3 py-2 font-medium">Version</th>
                <th className="px-3 py-2 font-medium">Source</th>
                <th className="px-3 py-2 font-medium">Accuracy</th>
                <th className="px-3 py-2 font-medium">Bundle</th>
                <th className="px-3 py-2 font-medium">Registered</th>
                <th className="px-3 py-2 font-medium">Alias</th>
                <th className="px-3 py-2 font-medium" />
              </tr>
            </thead>
            <tbody>
              {(payload?.registry?.versions ?? []).map((version) => (
                <tr key={version.version} className="border-b border-white/5">
                  <td className="px-3 py-2">
                    <Mono>{version.version}</Mono>
                  </td>
                  <td className="px-3 py-2">
                    <Badge tone={version.source === 's7' ? 'accent' : 'neutral'}>
                      {version.source}
                    </Badge>
                  </td>
                  <td className="px-3 py-2">
                    <Pct value={version.metrics?.accuracy} />
                  </td>
                  <td className="px-3 py-2">
                    <Mono className="text-textMuted">{version.bundle_id}</Mono>
                  </td>
                  <td className="px-3 py-2 text-xs text-textMuted">
                    {formatTime(version.registered_at)}
                  </td>
                  <td className="px-3 py-2">
                    {version.version === champion && <Badge tone="good">champion</Badge>}
                    {version.version === challenger && <Badge tone="accent">challenger</Badge>}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {version.version !== champion && (
                      <DemoGate reason="Promotion runs in the dev deployment">
                        <button
                          type="button"
                          onClick={() => void promote(version.version)}
                          className="text-xs px-2 py-1 rounded bg-panel2 hover:bg-accent"
                        >
                          Promote
                        </button>
                      </DemoGate>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollX>
      </Panel>

      <Panel title="Promotion history" subtitle="Append-only. Nothing here is ever overwritten.">
        {(payload?.registry?.history ?? []).length === 0 ? (
          <EmptyState title="No promotions recorded yet" />
        ) : (
          <ol className="p-4 space-y-3">
            {(payload?.registry?.history ?? []).map((event, i) => (
              <li key={`${event.at}-${i}`} className="flex gap-3 text-sm">
                <span className="text-textMuted text-xs whitespace-nowrap w-40 shrink-0">
                  {formatTime(event.at)}
                </span>
                <div className="min-w-0">
                  <div>
                    <Mono>{event.version}</Mono> became champion
                    {event.previous_champion ? (
                      <span className="text-textMuted">
                        , replacing <Mono>{event.previous_champion}</Mono>
                      </span>
                    ) : null}
                    {event.forced && (
                      <span className="ml-2">
                        <Badge tone="warn">forced</Badge>
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-textMuted mt-0.5">{event.reason}</div>
                </div>
              </li>
            ))}
          </ol>
        )}
      </Panel>

      <Panel title="Runs" subtitle="Every eval, GEPA and auto-research run recorded.">
        <ScrollX>
          <table className="w-full text-sm">
            <thead className="text-xs text-textMuted border-b border-white/5">
              <tr className="text-left">
                <th className="px-3 py-2 font-medium">Run</th>
                <th className="px-3 py-2 font-medium">Kind</th>
                <th className="px-3 py-2 font-medium">Started</th>
                <th className="px-3 py-2 font-medium">Accuracy</th>
                <th className="px-3 py-2 font-medium">Bundle</th>
                <th className="px-3 py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {(payload?.runs ?? []).map((run) => (
                <tr key={run.run_id} className="border-b border-white/5 hover:bg-panel2/60">
                  <td className="px-3 py-2">
                    <Mono>{run.run_name}</Mono>
                  </td>
                  <td className="px-3 py-2">
                    <Badge tone={run.kind === 's7' ? 'accent' : 'neutral'}>{run.kind}</Badge>
                  </td>
                  <td className="px-3 py-2 text-xs text-textMuted">{formatTime(run.start_time)}</td>
                  <td className="px-3 py-2">
                    {run.metrics.accuracy !== undefined ? (
                      <Pct value={run.metrics.accuracy} />
                    ) : (
                      <span className="text-textMuted">—</span>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <Mono className="text-textMuted">{run.bundle_id}</Mono>
                  </td>
                  <td className="px-3 py-2 text-xs">{run.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollX>
      </Panel>
    </div>
  );
}
