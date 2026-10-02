import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import { fmtPct } from '@/kit/format';
import { Empty, Kpi, Loading } from '@/kit/ui';
import { encPath, useLens } from '@/kit/url';
import { getDataset } from './api';
import type { DatasetRow } from './api';
import { InstrumentTable } from './InstrumentTable';
import { AdminPage } from './ui';
import './filings.css';

/**
 * Slot 2 · Filings & questions — what the benchmark is: gold, not predictions.
 *
 * Every optimisation verdict ultimately leans on these rows being right, and
 * the teacher occasionally says they are not (`gold_suspect`, settled on the
 * Gold review slot). This page is each split's every question beside its gold
 * answer and gold program, filterable, one click from asking the agent itself.
 *
 * The `derived gold` column is the second argument this page settles. The loop
 * attributes each failure to a subagent by walking four checks read straight
 * out of the gold program — what triage should have said, the operation
 * skeleton preprocess should have planned, the document values the retriever
 * owed, the answer the calculator owed. That derivation decides which agent an
 * experiment targets, so when the teacher disputes it, this is where a human
 * can see what the rule actually computed and judge which of them is right.
 */

const SPLITS = ['train', 'test', 'holdout'] as const;
type Split = (typeof SPLITS)[number];

const SPLIT_BLURB: Record<Split, string> = {
  train: 'what eval runs score and the teacher diagnoses — optimisation data',
  test: 'unseen by optimisation — the only split promotion evidence may come from',
  holdout: 'sealed for release gates; the gold is public, the model evidence is not',
};

const isSplit = (s: string | null): s is Split => (SPLITS as readonly string[]).includes(s ?? '');

const chatPath = (reportId: string) => `/agent/chat/${encPath(reportId)}`;

function programCell(program: string, turnType: string) {
  if (!program || program === '—') return <span className="muted">—</span>;
  const isSelection = turnType.toLowerCase() === 'number';
  return (
    <code className={`prog${isSelection ? ' sel' : ''}`} title={program}>
      {program}
    </code>
  );
}

export default function DatasetReview() {
  const [lens, setLens] = useLens();
  const rawSplit = lens.get('split');
  const split: Split = isSplit(rawSplit) ? rawSplit : 'train';
  const [needle, setNeedle] = useState('');
  const query = useQuery({
    queryKey: ['eval-dataset', split],
    queryFn: () => getDataset(split),
    staleTime: Infinity, // committed gold — it does not change under the page
  });

  const all = useMemo(() => query.data ?? [], [query.data]);

  const rows = useMemo(() => {
    const q = needle.trim().toLowerCase();
    if (!q) return all;
    return all.filter(
      (r) =>
        r.report_id.toLowerCase().includes(q) ||
        r.question.toLowerCase().includes(q) ||
        r.gold_answer.toLowerCase().includes(q) ||
        r.gold_program.toLowerCase().includes(q),
    );
  }, [all, needle]);

  const nConversations = useMemo(() => new Set(rows.map((r) => r.report_id)).size, [rows]);

  // The headline counts describe the whole split, not the filtered view.
  const facts = useMemo(() => {
    const n = all.length;
    const filings = new Set(all.map((r) => r.report_id)).size;
    const programs = all.filter((r) => r.turn_type && r.turn_type.toLowerCase() !== 'number').length;
    const hybrid = all.filter((r) => r.conv_type === 'Type II').length;
    return { n, filings, programs, hybrid };
  }, [all]);

  // Fixed layout, percentage widths summing to 100: every column that can hold
  // a long string wraps, so the table lays itself out to its box rather than
  // to the longest gold program in the split.
  const columns = useMemo<Array<ColumnDef<DatasetRow, unknown>>>(
    () => [
      {
        header: 'filing · turn',
        accessorKey: 'report_id',
        meta: { align: 'left', mono: false, wrap: true, width: '18%' },
        cell: ({ row }) => {
          const r = row.original;
          return (
            <>
              <span className="rid">{r.report_id}</span>
              <span className="path">
                q{r.turn_index} · <Link to={chatPath(r.report_id)}>ask it →</Link>
              </span>
            </>
          );
        },
      },
      {
        header: 'question',
        accessorKey: 'question',
        meta: { align: 'left', mono: false, wrap: true, width: '28%' },
        cell: ({ row }) => row.original.question,
      },
      {
        header: 'gold answer',
        accessorKey: 'gold_answer',
        meta: { align: 'right', wrap: true, width: '10%' },
        cell: ({ row }) => <span className="ans">{row.original.gold_answer}</span>,
      },
      {
        header: 'gold program',
        accessorKey: 'gold_program',
        meta: { align: 'left', mono: true, wrap: true, width: '18%' },
        cell: ({ row }) => programCell(row.original.gold_program, row.original.turn_type),
      },
      {
        header: 'derived gold',
        id: 'derived',
        meta: { align: 'left', mono: false, wrap: true, width: '16%' },
        cell: ({ row }) => {
          const r = row.original;
          if (!r.expected_skeleton.length && !r.expected_operands.length) {
            return (
              <span className="derived">
                <span className="none">triage {r.expected_triage || '—'} · no program to derive from</span>
              </span>
            );
          }
          return (
            <span className="derived">
              <span>
                <span className="k">skeleton</span>
                <code>{r.expected_skeleton.join(' → ') || '—'}</code>
              </span>
              <span>
                <span className="k">retrieve</span>
                <code>{r.expected_operands.join(', ') || 'nothing — all from history'}</code>
              </span>
            </span>
          );
        },
      },
      {
        header: 'type',
        accessorKey: 'turn_type',
        meta: { align: 'right', mono: false, wrap: true, width: '10%' },
        cell: ({ row }) => (
          <>
            <span className="chip">{row.original.turn_type || '—'}</span>
            {row.original.conv_type && <span className="path">{row.original.conv_type}</span>}
          </>
        ),
      },
    ],
    [],
  );

  const title = query.data ? (
    <>
      {facts.n} questions in {facts.filings} filings make the <em>{split}</em> split; {fmtPct(facts.programs / (facts.n || 1))} of
      them need a program, not a lookup
    </>
  ) : (
    <>
      The <em>{split}</em> split: every question beside its gold
    </>
  );

  return (
    <AdminPage
      eyebrow={`filings & questions · ${split} split`}
      title={title}
      sub={
        <>
          The {split} split is {SPLIT_BLURB[split]}. Each row is one turn of a conversation about one filing, beside
          its gold answer, its gold program and the per-subagent gold the attribution rule derives from them — the
          rows every verdict leans on.
        </>
      }
      testId="dataset-review"
    >
      {query.data && (
        <div className="kpis">
          <Kpi
            n={facts.n}
            b={
              <>
                questions over {facts.filings} filings · {(facts.n / (facts.filings || 1)).toFixed(1)} per conversation
              </>
            }
          />
          <Kpi
            n={`${facts.programs}/${facts.n}`}
            b={<>turns need a gold program ({fmtPct(facts.programs / (facts.n || 1))}); the rest select one number</>}
            tone="acc"
          />
          <Kpi
            n={`${facts.hybrid}/${facts.n}`}
            b={<>turns sit in hybrid (Type II) conversations, where two questions share one history</>}
          />
        </div>
      )}

      <section>
        <h2>Every question beside the gold it is scored against</h2>
        <div className="filters" role="group" aria-label="split and filter">
          {SPLITS.map((s) => (
            <button
              key={s}
              type="button"
              className={`tog${s === split ? ' on' : ''}`}
              aria-pressed={s === split}
              data-testid={`dataset-split-${s}`}
              onClick={() => setLens({ split: s === 'train' ? null : s })}
            >
              {s}
            </button>
          ))}
          <input
            type="search"
            value={needle}
            onChange={(e) => setNeedle(e.target.value)}
            placeholder="filter by filing, question, answer, or program"
            aria-label="filter questions"
          />
          <span className="count">
            {rows.length} questions · {nConversations} filings
          </span>
        </div>
        {query.isLoading ? (
          <Loading what="the split" />
        ) : query.error ? (
          <Loading what={`/eval/dataset?split=${split}`} error={query.error instanceof Error ? query.error.message : String(query.error)} />
        ) : rows.length === 0 ? (
          <Empty>no questions match “{needle}” in the {split} split</Empty>
        ) : (
          <div className="fl-table">
            <InstrumentTable
              data={rows}
              columns={columns}
              rowKey={(r) => `${r.report_id}#${r.turn_index}`}
              minWidth={860}
              layout="fixed"
              maxHeight={720}
              emptyLabel="no rows"
              testId="dataset-table"
            />
          </div>
        )}
        <p className="fl-source">
          Source: <code>/eval/dataset?split={split}</code>, from the committed split manifest{' '}
          <code>evaluation/splits/</code> and the ConvFinQA dataset. Gold only; no model output on this page.
        </p>
      </section>
    </AdminPage>
  );
}
