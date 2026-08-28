import { useEffect, useMemo, useState } from 'react';
import * as api from '../api';
import type { AnswerRow, SplitSummary, TraceDetail } from '../types';
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
} from './ui';

/**
 * Data & answers: the splits made visible, and every question with each
 * version's answer beside gold.
 *
 * The disagreements filter is the part worth using — it surfaces exactly the
 * turns where versions differ, which is where a regression actually lives. A
 * headline percentage cannot show you that v3_1 fixed some turns and broke
 * others; this can.
 */
export function DataPanel() {
  const [splits, setSplits] = useState<SplitSummary[]>([]);
  const [rows, setRows] = useState<AnswerRow[]>([]);
  const [reportFilter, setReportFilter] = useState('');
  const [onlyDisagreements, setOnlyDisagreements] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [trace, setTrace] = useState<TraceDetail | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([api.getSplits(), api.getAnswers('', onlyDisagreements)])
      .then(([s, a]) => {
        if (cancelled) return;
        setSplits(s);
        setRows(a);
        setError(null);
      })
      .catch((err) => !cancelled && setError(String(err)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [onlyDisagreements]);

  const splitOf = useMemo(() => {
    const map = new Map<string, string>();
    for (const split of splits) {
      if (split.name === 'sampled') continue;
      for (const rid of split.report_ids) map.set(rid, split.name);
    }
    return map;
  }, [splits]);

  const versions = useMemo(() => {
    const seen: string[] = [];
    for (const row of rows) {
      for (const v of row.versions) if (!seen.includes(v.version)) seen.push(v.version);
    }
    return seen;
  }, [rows]);

  const filtered = useMemo(
    () =>
      reportFilter
        ? rows.filter((r) => r.report_id.toLowerCase().includes(reportFilter.toLowerCase()))
        : rows,
    [rows, reportFilter],
  );

  async function openTrace(version: string, reportId: string, turnIndex: number) {
    try {
      setTrace(await api.getEvalTrace(version, reportId, turnIndex));
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <Panel
        title="Dataset splits"
        subtitle="The held-out discipline, as something you can check rather than take on trust."
      >
        <div className="grid gap-3 p-4 sm:grid-cols-3">
          {splits.map((split) => (
            <div key={split.name} className="bg-bg/40 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium capitalize">{split.name}</span>
                {split.name === 'holdout' && <Badge tone="good">reported on</Badge>}
                {split.name === 'train' && <Badge tone="warn">optimizer saw</Badge>}
              </div>
              <div className="text-xl font-semibold tabular-nums">{split.n_questions}</div>
              <div className="text-xs text-textMuted">
                questions across {split.n_conversations} conversations
              </div>
              <p className="text-[11px] text-textMuted mt-2 leading-snug">{split.description}</p>
            </div>
          ))}
        </div>
      </Panel>

      <Panel
        title="All the answers"
        subtitle={`Every question with gold and each version's answer beside it${
          versions.length ? ` (${versions.join(', ')})` : ''
        }.`}
        actions={
          <>
            <label className="flex items-center gap-1.5 text-xs text-textMuted">
              <input
                type="checkbox"
                checked={onlyDisagreements}
                onChange={(e) => setOnlyDisagreements(e.target.checked)}
              />
              only where versions disagree
            </label>
            <input
              value={reportFilter}
              onChange={(e) => setReportFilter(e.target.value)}
              placeholder="filter report…"
              className="bg-panel2 text-sm rounded px-2 py-1 border border-white/10 w-48"
            />
          </>
        }
      >
        {error && <ErrorNote error={error} />}
        {loading ? (
          <Spinner />
        ) : filtered.length === 0 ? (
          <EmptyState
            title={onlyDisagreements ? 'No disagreements in range' : 'No answers found'}
            hint="Prediction CSVs are committed to the repo; if this is empty, none were found."
          />
        ) : (
          <ScrollX>
            <table className="w-full text-sm">
              <thead className="text-xs text-textMuted border-b border-white/5">
                <tr className="text-left">
                  <th className="px-3 py-2 font-medium">Report</th>
                  <th className="px-3 py-2 font-medium">Split</th>
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Question</th>
                  <th className="px-3 py-2 font-medium">Gold</th>
                  {versions.map((v) => (
                    <th key={v} className="px-3 py-2 font-medium">
                      {v}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((row) => (
                  <tr
                    key={`${row.report_id}-${row.turn_index}`}
                    className="border-b border-white/5 hover:bg-panel2/60"
                  >
                    <td className="px-3 py-2">
                      <Mono>{row.report_id}</Mono>
                    </td>
                    <td className="px-3 py-2">
                      <Badge tone={splitOf.get(row.report_id) === 'holdout' ? 'good' : 'warn'}>
                        {splitOf.get(row.report_id) ?? '—'}
                      </Badge>
                    </td>
                    <td className="px-3 py-2 tabular-nums text-textMuted">{row.turn_index}</td>
                    <td className="px-3 py-2 max-w-sm">{row.question}</td>
                    <td className="px-3 py-2">
                      <Mono>{row.gold_answer}</Mono>
                    </td>
                    {versions.map((version) => {
                      const answer = row.versions.find((v) => v.version === version);
                      if (!answer)
                        return (
                          <td key={version} className="px-3 py-2 text-textMuted">
                            —
                          </td>
                        );
                      return (
                        <td key={version} className="px-3 py-2 whitespace-nowrap">
                          <button
                            type="button"
                            onClick={() =>
                              void openTrace(version, row.report_id, row.turn_index)
                            }
                            className="flex items-center gap-1.5 hover:text-accent2"
                            title="Open the stage trace for this scored turn"
                          >
                            <CorrectMark correct={answer.correct} />
                            <Mono>{answer.pred_answer || '—'}</Mono>
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollX>
        )}
      </Panel>

      {trace && (
        <Panel
          title={`Scored turn — ${trace.report_id} · turn ${trace.turn_index}`}
          subtitle={trace.question}
          actions={
            <button
              type="button"
              onClick={() => setTrace(null)}
              className="text-sm px-2 py-1 rounded bg-panel2 hover:bg-accent"
            >
              Close
            </button>
          }
        >
          <div className="p-4 space-y-4">
            <GoldComparison
              answer={trace.answer}
              gold={trace.gold_answer}
              correct={trace.correct}
              program={trace.program}
            />
            {trace.history_text ? (
              <details>
                <summary className="text-[11px] uppercase tracking-wide text-textMuted cursor-pointer">
                  Conversation history the agent saw
                </summary>
                <pre className="text-[11px] mt-1 bg-bg/60 rounded p-2 whitespace-pre-wrap">
                  {trace.history_text}
                </pre>
              </details>
            ) : null}
            <StageTimeline capture={trace.capture} />
          </div>
        </Panel>
      )}
    </div>
  );
}
