import { useEffect, useMemo, useState } from 'react';
import * as api from '../api';
import type { AccuracySlice, EvalSummary, ModelAccuracy, PredRow } from '../types';

// ---------------------------------------------------------------------------
// types
// ---------------------------------------------------------------------------

interface TurnComparison {
  q_order: number;
  turn_type: string;
  conv_type: string;
  question: string;
  gold_answer: string;
  gold_program: string;
  predictions: Record<string, { pred_answer: string; pred_program: string; correct: boolean }>;
}

interface ReportStat {
  report_id: string;
  modelAcc: Record<string, { correct: number; total: number }>;
}

const MODEL_LABEL: Record<string, string> = {
  dspy: 'DSPy',
  pydantic: 'Pydantic AI',
  api: 'Direct API',
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function mergePredsForReport(reportId: string, allPreds: Record<string, PredRow[]>): TurnComparison[] {
  const turns = new Map<number, TurnComparison>();
  for (const [model, preds] of Object.entries(allPreds)) {
    for (const row of preds) {
      if (row.report_id !== reportId) continue;
      if (!turns.has(row.q_order)) {
        turns.set(row.q_order, {
          q_order: row.q_order,
          turn_type: row.turn_type,
          conv_type: row.conv_type,
          question: row.question,
          gold_answer: row.gold_answer,
          gold_program: row.gold_program,
          predictions: {},
        });
      }
      turns.get(row.q_order)!.predictions[model] = {
        pred_answer: row.pred_answer,
        pred_program: row.pred_program,
        correct: row.correct,
      };
    }
  }
  return Array.from(turns.values()).sort((a, b) => a.q_order - b.q_order);
}

function buildReportStats(allPreds: Record<string, PredRow[]>, primaryModel: string): ReportStat[] {
  const reportIds = new Set<string>();
  for (const preds of Object.values(allPreds)) {
    for (const row of preds) reportIds.add(row.report_id);
  }
  return Array.from(reportIds)
    .map((rid) => {
      const modelAcc: ReportStat['modelAcc'] = {};
      for (const [model, preds] of Object.entries(allPreds)) {
        const rows = preds.filter((r) => r.report_id === rid);
        if (rows.length > 0)
          modelAcc[model] = { correct: rows.filter((r) => r.correct).length, total: rows.length };
      }
      return { report_id: rid, modelAcc };
    })
    .sort((a, b) => {
      const aAcc = a.modelAcc[primaryModel];
      const bAcc = b.modelAcc[primaryModel];
      if (!aAcc) return 1;
      if (!bAcc) return -1;
      return bAcc.correct / bAcc.total - aAcc.correct / aAcc.total;
    });
}

// ---------------------------------------------------------------------------
// Overview sub-components (accuracy cards)
// ---------------------------------------------------------------------------

function AccBar({ value }: { value: number }) {
  const color = value >= 0.7 ? '#00a884' : value >= 0.5 ? '#e9b03d' : '#f15c6d';
  return (
    <div className="h-1.5 rounded-full bg-panel overflow-hidden">
      <div className="h-full rounded-full" style={{ width: `${value * 100}%`, backgroundColor: color }} />
    </div>
  );
}

function SliceList({ slices }: { slices: AccuracySlice[] }) {
  return (
    <div className="flex flex-col gap-3">
      {slices.map((s) => (
        <div key={s.label} className="flex flex-col gap-0.5">
          <div className="flex justify-between text-xs">
            <span className="text-textMuted">{s.label}</span>
            <span className="font-mono text-textMain">{(s.accuracy * 100).toFixed(1)}%</span>
          </div>
          <AccBar value={s.accuracy} />
          <div className="text-[10px] text-textMuted">{s.n_correct}/{s.n_total}</div>
        </div>
      ))}
    </div>
  );
}

function AccCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-panel2 rounded-lg p-4 flex flex-col gap-3">
      <div className="text-xs font-medium text-textMuted uppercase tracking-wide">{title}</div>
      {children}
    </div>
  );
}

function OverallCard({ slice }: { slice: AccuracySlice }) {
  const pct = (slice.accuracy * 100).toFixed(1);
  const color = slice.accuracy >= 0.7 ? '#00a884' : slice.accuracy >= 0.5 ? '#e9b03d' : '#f15c6d';
  return (
    <AccCard title="Overall">
      <div className="flex flex-col items-center justify-center flex-1 py-2 gap-1">
        <div className="text-4xl font-bold tabular-nums" style={{ color }}>{pct}%</div>
        <div className="text-xs text-textMuted">{slice.n_correct}/{slice.n_total} correct</div>
      </div>
    </AccCard>
  );
}

function Insights({ acc }: { acc: ModelAccuracy }) {
  const numAcc = acc.by_turn_type.find((s) => s.label === 'Number')?.accuracy ?? 0;
  const progAcc = acc.by_turn_type.find((s) => s.label === 'Program')?.accuracy ?? 0;
  const t1 = acc.by_conv_type.find((s) => s.label === 'Type I')?.accuracy ?? 0;
  const t2 = acc.by_conv_type.find((s) => s.label === 'Type II')?.accuracy ?? 0;
  const q0 = acc.by_q_order.find((s) => s.label === '0')?.accuracy ?? 0;
  const late = acc.by_q_order.filter((s) => Number(s.label) >= 3);
  const lateAcc =
    late.length > 0
      ? late.reduce((a, s) => a + s.accuracy * s.n_total, 0) / late.reduce((a, s) => a + s.n_total, 0)
      : 0;
  const items = [
    numAcc > 0 && `Program questions (${(progAcc * 100).toFixed(0)}%) are ${((numAcc - progAcc) * 100).toFixed(0)}pp harder than number retrieval (${(numAcc * 100).toFixed(0)}%)`,
    t1 > 0 && `Type II conversations (${(t2 * 100).toFixed(0)}%) trail Type I by ${((t1 - t2) * 100).toFixed(0)}pp — longer cross-question dependency chains`,
    late.length > 0 && `Accuracy drops ${((q0 - lateAcc) * 100).toFixed(0)}pp by turn 3+ — earlier wrong answers propagate`,
  ].filter(Boolean) as string[];
  return (
    <div className="bg-panel2 rounded-lg p-4">
      <div className="text-xs font-medium text-textMuted uppercase tracking-wide mb-3">Key observations</div>
      <ul className="flex flex-col gap-2">
        {items.map((ins, i) => (
          <li key={i} className="text-sm text-textMuted flex gap-2 leading-snug">
            <span className="text-accent2 shrink-0">→</span>{ins}
          </li>
        ))}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Browse view
// ---------------------------------------------------------------------------

function MiniAccPill({ correct, total }: { correct: number; total: number }) {
  const pct = total > 0 ? correct / total : 0;
  const color = pct >= 0.7 ? 'text-accent2' : pct >= 0.5 ? 'text-yellow-400' : 'text-danger';
  return (
    <span className={`text-[10px] font-mono tabular-nums ${color}`}>
      {correct}/{total}
    </span>
  );
}

function ReportList({
  reports,
  selectedId,
  onSelect,
  availableModels,
}: {
  reports: ReportStat[];
  selectedId: string;
  onSelect: (id: string) => void;
  availableModels: string[];
}) {
  return (
    <div className="flex flex-col min-h-0 border-r border-black/30">
      <div className="px-3 py-2 border-b border-black/20 shrink-0">
        <div className="text-[10px] font-medium uppercase tracking-wide text-textMuted">
          {reports.length} documents
        </div>
      </div>
      <div className="overflow-y-auto flex-1">
        {reports.map((r) => {
          const isActive = r.report_id === selectedId;
          return (
            <button
              key={r.report_id}
              type="button"
              onClick={() => onSelect(r.report_id)}
              className={`w-full text-left px-3 py-2.5 border-b border-black/10 last:border-0 transition-colors ${
                isActive ? 'bg-accent/40' : 'hover:bg-panel2'
              }`}
            >
              <div className="text-xs font-mono text-textMain truncate leading-snug mb-1">
                {r.report_id.replace(/^(Double|Single)_/, '')}
              </div>
              <div className="flex gap-2">
                {availableModels.map((m) => {
                  const acc = r.modelAcc[m];
                  return acc ? (
                    <span key={m} className="flex items-center gap-0.5">
                      <span className="text-[9px] text-textMuted">{MODEL_LABEL[m]?.split(' ')[0]}</span>
                      <MiniAccPill correct={acc.correct} total={acc.total} />
                    </span>
                  ) : null;
                })}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ConversationView({
  reportId,
  turns,
  availableModels,
}: {
  reportId: string;
  turns: TurnComparison[];
  availableModels: string[];
}) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const toggle = (q: number) =>
    setExpanded((prev) => { const n = new Set(prev); n.has(q) ? n.delete(q) : n.add(q); return n; });

  const correct = (model: string) => turns.filter((t) => t.predictions[model]?.correct).length;

  return (
    <div className="flex flex-col min-h-0 flex-1">
      {/* report header */}
      <div className="px-5 py-3 border-b border-black/30 shrink-0">
        <div className="font-mono text-sm text-textMain truncate">{reportId}</div>
        <div className="flex gap-4 mt-1">
          {availableModels.map((m) =>
            turns.some((t) => m in t.predictions) ? (
              <span key={m} className="text-xs text-textMuted">
                {MODEL_LABEL[m]}:{' '}
                <span className={correct(m) / turns.length >= 0.7 ? 'text-accent2' : correct(m) / turns.length >= 0.5 ? 'text-yellow-400' : 'text-danger'}>
                  {correct(m)}/{turns.length}
                </span>
              </span>
            ) : null
          )}
        </div>
      </div>

      {/* column header */}
      <div
        className="grid gap-x-3 px-5 py-2 text-[10px] font-medium uppercase tracking-wide text-textMuted border-b border-black/20 shrink-0"
        style={{ gridTemplateColumns: `2rem 5rem 5rem 1fr 8rem ${availableModels.map(() => '9rem').join(' ')}` }}
      >
        <span>Q#</span>
        <span>Type</span>
        <span>Conv</span>
        <span>Question</span>
        <span>Gold answer</span>
        {availableModels.map((m) => <span key={m}>{MODEL_LABEL[m] ?? m}</span>)}
      </div>

      {/* rows */}
      <div className="overflow-y-auto flex-1">
        {turns.map((t) => {
          const isExp = expanded.has(t.q_order);
          const anyWrong = availableModels.some((m) => t.predictions[m] && !t.predictions[m].correct);
          return (
            <div key={t.q_order} className={`border-b border-black/10 last:border-0 ${anyWrong ? '' : 'opacity-80'}`}>
              <button
                type="button"
                className="w-full text-left"
                onClick={() => toggle(t.q_order)}
              >
                <div
                  className="grid gap-x-3 px-5 py-2.5 hover:bg-panel/50 transition-colors items-start"
                  style={{ gridTemplateColumns: `2rem 5rem 5rem 1fr 8rem ${availableModels.map(() => '9rem').join(' ')}` }}
                >
                  <span className="text-xs font-mono text-textMuted pt-0.5">{t.q_order}</span>
                  <span className={`text-xs pt-0.5 ${t.turn_type === 'Program' ? 'text-accent2' : 'text-textMuted'}`}>
                    {t.turn_type}
                  </span>
                  <span className={`text-xs pt-0.5 ${t.conv_type === 'Type II' ? 'text-yellow-400' : 'text-textMuted'}`}>
                    {t.conv_type}
                  </span>
                  <span className="text-xs text-textMain leading-snug">{t.question}</span>
                  <span className="text-xs font-mono text-textMain pt-0.5">{t.gold_answer}</span>
                  {availableModels.map((m) => {
                    const pred = t.predictions[m];
                    if (!pred) return <span key={m} className="text-xs text-textMuted pt-0.5">—</span>;
                    return (
                      <span key={m} className={`text-xs font-mono pt-0.5 ${pred.correct ? 'text-accent2' : 'text-danger'}`}>
                        {pred.correct ? '✓' : '✗'} {pred.pred_answer}
                      </span>
                    );
                  })}
                </div>
              </button>
              {isExp && (
                <div className="px-5 pb-3 bg-panel/30">
                  <div className="grid gap-x-4 gap-y-2 items-start"
                    style={{ gridTemplateColumns: `7rem 1fr${availableModels.map(() => ' 1fr').join('')}` }}>
                    {/* header row */}
                    <span />
                    <span className="text-[10px] font-medium uppercase tracking-wide text-accent2">Gold</span>
                    {availableModels.map((m) => (
                      <span key={m} className="text-[10px] font-medium uppercase tracking-wide text-textMuted">
                        {MODEL_LABEL[m] ?? m}
                      </span>
                    ))}
                    {/* program row */}
                    <span className="text-[10px] text-textMuted pt-1">Program</span>
                    <code className="text-[10px] font-mono text-accent2 bg-panel rounded px-2 py-1 break-all leading-snug">
                      {t.gold_program || '—'}
                    </code>
                    {availableModels.map((m) => {
                      const pred = t.predictions[m];
                      return (
                        <code key={m} className={`text-[10px] font-mono rounded px-2 py-1 break-all leading-snug ${
                          !pred ? 'text-textMuted bg-panel' :
                          pred.correct ? 'text-accent2 bg-panel' : 'text-danger bg-panel'
                        }`}>
                          {pred?.pred_program || '—'}
                        </code>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Wrong answers table (overview)
// ---------------------------------------------------------------------------

function roundAnswer(s: string): string {
  const trimmed = s.trim();
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
    return parseFloat(trimmed).toFixed(1);
  }
  return s;
}

function WrongAnswersTable({ preds }: { preds: PredRow[] }) {
  const [showAll, setShowAll] = useState(false);
  const [search, setSearch] = useState('');

  const rows = preds.filter((r) => {
    if (!showAll && r.correct) return false;
    if (search && !r.question.toLowerCase().includes(search.toLowerCase()) &&
        !r.report_id.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const selectCls = 'bg-panel text-textMain text-xs rounded px-2 py-1 border border-black/30';

  return (
    <div className="bg-panel2 rounded-lg flex flex-col">
      {/* header */}
      <div className="px-4 py-3 border-b border-black/30 flex flex-wrap items-center gap-3">
        <div className="text-xs font-medium text-textMuted uppercase tracking-wide">
          Predictions
        </div>
        <select className={selectCls} value={showAll ? 'all' : 'wrong'} onChange={(e) => setShowAll(e.target.value === 'all')}>
          <option value="wrong">✗ Wrong only</option>
          <option value="all">All</option>
        </select>
        <input
          className="bg-panel text-textMain text-xs rounded px-2 py-1 border border-black/30 flex-1 min-w-[160px] placeholder:text-textMuted"
          placeholder="Search question or document…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <span className="text-xs text-textMuted ml-auto shrink-0">{rows.length} rows</span>
      </div>

      {/* col headers */}
      <div
        className="grid gap-x-3 px-4 py-2 text-[10px] font-medium uppercase tracking-wide text-textMuted border-b border-black/20 shrink-0"
        style={{ gridTemplateColumns: '180px 2rem 5rem 5rem 1fr 7rem 7rem 1fr 1fr' }}
      >
        <span>Document</span>
        <span>Q#</span>
        <span>Type</span>
        <span>Conv</span>
        <span>Question</span>
        <span>Gold ans</span>
        <span>Pred ans</span>
        <span>Gold program</span>
        <span>Pred program</span>
      </div>

      {/* rows */}
      <div className="divide-y divide-black/10">
        {rows.length === 0 && (
          <div className="py-8 text-center text-sm text-textMuted">No rows match the current filters.</div>
        )}
        {rows.map((r) => (
          <div
            key={`${r.report_id}__${r.turn_index}`}
            className="grid gap-x-3 px-4 py-2.5 text-xs items-start hover:bg-panel/40"
            style={{ gridTemplateColumns: '180px 2rem 5rem 5rem 1fr 7rem 7rem 1fr 1fr' }}
          >
            <span className="font-mono text-textMuted truncate leading-snug" title={r.report_id}>
              {r.report_id.replace(/^(Double|Single)_/, '')}
            </span>
            <span className="font-mono text-textMuted">{r.q_order}</span>
            <span className={r.turn_type === 'Program' ? 'text-accent2' : 'text-textMuted'}>{r.turn_type}</span>
            <span className={r.conv_type === 'Type II' ? 'text-yellow-400' : 'text-textMuted'}>{r.conv_type}</span>
            <span className="text-textMain leading-snug">{r.question}</span>
            <span className="font-mono text-accent2">{roundAnswer(r.gold_answer)}</span>
            <span className={`font-mono ${r.correct ? 'text-accent2' : 'text-danger'}`}>
              {roundAnswer(r.pred_answer)}
            </span>
            <span className="font-mono text-textMuted text-[10px] leading-snug break-all">
              {r.gold_program || '—'}
            </span>
            <span className="font-mono text-textMuted text-[10px] leading-snug break-all">
              {r.pred_program || '—'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function EvalPanel() {
  const [runs, setRuns] = useState<string[]>([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [allPreds, setAllPreds] = useState<Record<string, PredRow[]>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [view, setView] = useState<'overview' | 'eval_sets'>('overview');
  const [selectedReport, setSelectedReport] = useState('');
  const [overviewModel, setOverviewModel] = useState('pydantic');

  useEffect(() => {
    api.listEvalRuns()
      .then((r) => {
        setRuns(r);
        if (r.length > 0) {
          const preferred = r.find((run) => run.includes('real')) ?? r[r.length - 1];
          setSelectedRun(preferred);
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selectedRun) return;
    setLoading(true);
    setError('');
    setAllPreds({});
    setSummary(null);
    setSelectedReport('');

    api.getEvalSummary(selectedRun)
      .then((sum) => {
        setSummary(sum);
        const primary = sum.available_models.includes('pydantic') ? 'pydantic' : sum.available_models[0];
        setOverviewModel(primary ?? 'pydantic');
        return Promise.all(
          sum.available_models.map((m) =>
            api.getEvalPredictions(selectedRun, m).then((preds) => [m, preds] as [string, PredRow[]])
          )
        );
      })
      .then((entries) => {
        const merged: Record<string, PredRow[]> = {};
        for (const [m, preds] of entries) merged[m] = preds;
        setAllPreds(merged);
        // auto-select first report
        const first = entries[0]?.[1]?.[0]?.report_id ?? '';
        setSelectedReport(first);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [selectedRun]);

  const availableModels = summary?.available_models ?? [];

  const reportStats = useMemo(
    () => buildReportStats(allPreds, overviewModel),
    [allPreds, overviewModel]
  );

  const selectedTurns = useMemo(
    () => (selectedReport ? mergePredsForReport(selectedReport, allPreds) : []),
    [selectedReport, allPreds]
  );

  const overviewAcc = summary?.models[overviewModel];

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* toolbar */}
      <div className="shrink-0 flex items-center gap-4 px-5 py-2.5 border-b border-black/30 bg-panel">
        <div className="flex items-center gap-2">
          <span className="text-xs text-textMuted">Run</span>
          <select
            className="bg-panel2 text-textMain text-xs rounded px-2 py-1.5 border border-black/30 max-w-[260px]"
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
          >
            {runs.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
        <div className="flex rounded-md overflow-hidden border border-black/30">
          {([['overview', 'Overview'], ['eval_sets', 'Evaluation Sets']] as const).map(([v, label]) => (
            <button
              key={v}
              type="button"
              onClick={() => setView(v)}
              className={`px-3 py-1 text-xs transition-colors ${
                view === v ? 'bg-accent2 text-bg font-medium' : 'bg-panel2 text-textMuted hover:text-textMain'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        {loading && <span className="text-xs text-textMuted">Loading…</span>}
        {error && <span className="text-xs text-danger">{error}</span>}
        {!loading && summary && (
          <span className="text-xs text-textMuted ml-auto">
            {reportStats.length} documents · {Object.values(allPreds)[0]?.length ?? 0} turns
          </span>
        )}
      </div>

      {/* empty state */}
      {!summary && !loading && (
        <div className="flex-1 flex items-center justify-center text-sm text-textMuted">
          {runs.length === 0
            ? 'No eval runs found. Run the agent evaluation to generate predictions.'
            : 'Select a run to view results.'}
        </div>
      )}

      {/* evaluation sets view */}
      {summary && view === 'eval_sets' && (
        <div className="flex-1 grid min-h-0 overflow-hidden" style={{ gridTemplateColumns: '260px 1fr' }}>
          <ReportList
            reports={reportStats}
            selectedId={selectedReport}
            onSelect={setSelectedReport}
            availableModels={availableModels}
          />
          {selectedReport ? (
            <ConversationView
              reportId={selectedReport}
              turns={selectedTurns}
              availableModels={availableModels}
            />
          ) : (
            <div className="flex items-center justify-center text-sm text-textMuted">
              Select a document
            </div>
          )}
        </div>
      )}

      {/* overview view  */}
      {summary && view === 'overview' && (
        <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-textMuted">Model</span>
            <div className="flex rounded-md overflow-hidden border border-black/30">
              {availableModels.map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setOverviewModel(m)}
                  className={`px-2.5 py-1 text-xs transition-colors ${
                    overviewModel === m
                      ? 'bg-accent2 text-bg font-medium'
                      : 'bg-panel2 text-textMuted hover:text-textMain'
                  }`}
                >
                  {MODEL_LABEL[m] ?? m}
                </button>
              ))}
            </div>
          </div>
          {overviewAcc && (
            <>
              <div className="grid grid-cols-4 gap-4">
                <OverallCard slice={overviewAcc.overall} />
                <AccCard title="By turn type"><SliceList slices={overviewAcc.by_turn_type} /></AccCard>
                <AccCard title="By conversation type"><SliceList slices={overviewAcc.by_conv_type} /></AccCard>
                <AccCard title="By question number"><SliceList slices={overviewAcc.by_q_order} /></AccCard>
              </div>
              <Insights acc={overviewAcc} />
              <WrongAnswersTable preds={allPreds[overviewModel] ?? []} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
