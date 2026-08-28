import { useCallback, useEffect, useState } from 'react';
import * as api from '../api';
import type { TraceDetail, TraceSummary } from '../types';
import { GoldComparison, StageTimeline } from './StageTimeline';
import {
  Badge,
  CorrectMark,
  EmptyState,
  ErrorNote,
  Mono,
  Panel,
  ScrollX,
  Spinner,
  formatMs,
  formatTime,
} from './ui';

export function TracesPanel() {
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [selected, setSelected] = useState<TraceDetail | null>(null);
  const [source, setSource] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{ n_turns: number; n_reports: number } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [rows, s] = await Promise.all([
        api.listTraces({ source: source || undefined }),
        api.getTraceStats(),
      ]);
      setTraces(rows);
      setStats(s);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [source]);

  useEffect(() => {
    void load();
  }, [load]);

  async function open(traceId: string) {
    try {
      setSelected(await api.getTrace(traceId));
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <Panel
        title="Traces"
        subtitle="Every turn this deployment has answered, stage by stage. Logfire keeps the deep external trace; this is the in-app view."
        actions={
          <>
            <select
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="bg-panel2 text-sm rounded px-2 py-1 border border-white/10"
            >
              <option value="">all sources</option>
              <option value="serving">serving</option>
              <option value="demo">demo replay</option>
              <option value="eval">eval</option>
            </select>
            <button
              type="button"
              onClick={() => void load()}
              className="text-sm px-2 py-1 rounded bg-panel2 hover:bg-accent"
            >
              Refresh
            </button>
          </>
        }
      >
        {stats && (
          <div className="flex gap-6 px-4 py-3 border-b border-white/5 text-sm">
            <div>
              <div className="text-xl font-semibold tabular-nums">{stats.n_turns}</div>
              <div className="text-xs text-textMuted">turns traced</div>
            </div>
            <div>
              <div className="text-xl font-semibold tabular-nums">{stats.n_reports}</div>
              <div className="text-xs text-textMuted">reports touched</div>
            </div>
          </div>
        )}

        {error && <ErrorNote error={error} />}
        {loading ? (
          <Spinner />
        ) : traces.length === 0 ? (
          <EmptyState
            title="No traces yet"
            hint="Ask a question in Chat and it will appear here with its full stage timeline."
          />
        ) : (
          <ScrollX>
            <table className="w-full text-sm">
              <thead className="text-xs text-textMuted border-b border-white/5">
                <tr className="text-left">
                  <th className="px-3 py-2 font-medium">When</th>
                  <th className="px-3 py-2 font-medium">Source</th>
                  <th className="px-3 py-2 font-medium">Report</th>
                  <th className="px-3 py-2 font-medium">Question</th>
                  <th className="px-3 py-2 font-medium">Answer</th>
                  <th className="px-3 py-2 font-medium text-center">✓</th>
                  <th className="px-3 py-2 font-medium text-right">Latency</th>
                  <th className="px-3 py-2 font-medium text-right">Tokens</th>
                </tr>
              </thead>
              <tbody>
                {traces.map((trace) => (
                  <tr
                    key={trace.trace_id}
                    onClick={() => void open(trace.trace_id)}
                    className="border-b border-white/5 hover:bg-panel2 cursor-pointer"
                  >
                    <td className="px-3 py-2 text-xs text-textMuted whitespace-nowrap">
                      {formatTime(trace.created_at)}
                    </td>
                    <td className="px-3 py-2">
                      <Badge tone={trace.source === 'demo' ? 'warn' : 'neutral'}>
                        {trace.source}
                      </Badge>
                    </td>
                    <td className="px-3 py-2">
                      <Mono>{trace.report_id}</Mono>
                    </td>
                    <td className="px-3 py-2 max-w-md truncate">{trace.question}</td>
                    <td className="px-3 py-2">
                      <Mono>{trace.answer || '—'}</Mono>
                    </td>
                    <td className="px-3 py-2 text-center">
                      <CorrectMark correct={trace.correct} />
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-xs">
                      {formatMs(trace.latency_ms)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-xs">
                      {trace.total_tokens ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollX>
        )}
      </Panel>

      {selected && (
        <Panel
          title={`Trace — ${selected.report_id} · turn ${selected.turn_index}`}
          subtitle={selected.question}
          actions={
            <button
              type="button"
              onClick={() => setSelected(null)}
              className="text-sm px-2 py-1 rounded bg-panel2 hover:bg-accent"
            >
              Close
            </button>
          }
        >
          <div className="p-4 space-y-4">
            <GoldComparison
              answer={selected.answer}
              gold={selected.gold_answer}
              correct={selected.correct}
              program={selected.program}
            />
            {selected.bundle?.prompts_version && (
              <div className="text-[11px] text-textMuted">
                answered by bundle <Mono>{selected.bundle_id}</Mono> · prompts{' '}
                <Mono>{selected.bundle.prompts_version}</Mono> · build{' '}
                <Mono>{selected.bundle.code_sha}</Mono>
              </div>
            )}
            {selected.error ? <ErrorNote error={selected.error} /> : null}
            <StageTimeline capture={selected.capture} />
          </div>
        </Panel>
      )}
    </div>
  );
}
