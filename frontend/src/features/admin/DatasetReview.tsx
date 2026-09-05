import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import { cn } from '@/lib/utils';
import { getDataset } from './api';
import type { DatasetRow } from './api';
import { InstrumentTable } from './InstrumentTable';
import { AdminPage, EmptyState, ErrorNote, LoadingRows, Panel } from './ui';

/**
 * The evaluation set itself — gold, not predictions.
 *
 * Every optimisation verdict ultimately leans on these rows being right, and
 * the teacher occasionally says they are not (`gold_suspect`). This page is
 * where a human settles that argument: each split's every question beside its
 * gold answer and gold program, filterable.
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

function programCell(program: string, turnType: string) {
  if (!program || program === '—') return <span className="text-faint">—</span>;
  const isSelection = turnType.toLowerCase() === 'number';
  // `break-words` only does anything because the column opts into `wrap` below:
  // the table's default is `whitespace-nowrap`, under which a program of
  // `subtract(206588, 3000000), divide(#0, 3000000)` sets the table's width on
  // its own and squeezes every other column to one word per line. It is
  // `break-words` rather than `break-all` deliberately — `break-all` wraps
  // mid-token and renders `divide(#0, 15704)` as `1` then `5704`, splitting an
  // operand across lines on the one page whose whole job is reading operands.
  return (
    <code
      className={cn('block font-mono text-[11px] break-words', isSelection && 'text-muted')}
      title={program}
    >
      {program}
    </code>
  );
}

export default function DatasetReview() {
  const [split, setSplit] = useState<Split>('train');
  const [needle, setNeedle] = useState('');
  const query = useQuery({
    queryKey: ['eval-dataset', split],
    queryFn: () => getDataset(split),
    staleTime: Infinity, // committed gold — it does not change under the page
  });

  const rows = useMemo(() => {
    const all = query.data ?? [];
    const q = needle.trim().toLowerCase();
    if (!q) return all;
    return all.filter(
      (r) =>
        r.report_id.toLowerCase().includes(q) ||
        r.question.toLowerCase().includes(q) ||
        r.gold_answer.toLowerCase().includes(q) ||
        r.gold_program.toLowerCase().includes(q),
    );
  }, [query.data, needle]);

  const nConversations = useMemo(
    () => new Set(rows.map((r) => r.report_id)).size,
    [rows],
  );

  // Fixed layout, percentage widths summing to 100: every column that can hold
  // a long string wraps, so the table lays itself out to the panel rather than
  // to the longest gold program in the split. Auto layout is not enough here —
  // it grows the report column to avoid breaking an unbreakable id and pushes
  // `type` off the right edge.
  const columns = useMemo<Array<ColumnDef<DatasetRow, unknown>>>(
    () => [
      {
        header: 'report',
        accessorKey: 'report_id',
        meta: { align: 'left', mono: true, wrap: true, width: '15%' },
        cell: ({ row }) => (
          <span className="block font-mono text-[11px] break-words">
            {row.original.report_id}
          </span>
        ),
      },
      {
        header: 'q#',
        accessorKey: 'turn_index',
        meta: { align: 'right', width: '4%' },
        cell: ({ row }) => <span className="font-mono">q{row.original.turn_index}</span>,
      },
      {
        header: 'question',
        accessorKey: 'question',
        meta: { align: 'left', mono: false, wrap: true, width: '27%' },
        cell: ({ row }) => <span className="block">{row.original.question}</span>,
      },
      {
        header: 'gold answer',
        accessorKey: 'gold_answer',
        meta: { align: 'right', wrap: true, width: '8%' },
        cell: ({ row }) => (
          <span className="block font-mono text-[12px] break-words">
            {row.original.gold_answer}
          </span>
        ),
      },
      {
        header: 'gold program',
        accessorKey: 'gold_program',
        meta: { align: 'left', mono: true, wrap: true, width: '20%' },
        cell: ({ row }) =>
          programCell(row.original.gold_program, row.original.turn_type),
      },
      {
        header: 'derived gold',
        id: 'derived',
        meta: { align: 'left', mono: true, wrap: true, width: '18%' },
        cell: ({ row }) => {
          const r = row.original;
          if (!r.expected_skeleton.length && !r.expected_operands.length) {
            return (
              <span className="type-small text-faint">
                triage {r.expected_triage || '—'} · no program to derive from
              </span>
            );
          }
          return (
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="type-small block text-muted">
                <span className="text-faint">skeleton</span>{' '}
                <code className="font-mono text-[11px] break-words text-violet">
                  {r.expected_skeleton.join(' → ') || '—'}
                </code>
              </span>
              <span className="type-small block text-muted">
                <span className="text-faint">retrieve</span>{' '}
                <code className="font-mono text-[11px] break-words text-amber">
                  {r.expected_operands.join(', ') || 'nothing — all from history'}
                </code>
              </span>
            </div>
          );
        },
      },
      {
        header: 'type',
        accessorKey: 'turn_type',
        meta: { align: 'right', wrap: true, width: '8%' },
        cell: ({ row }) => (
          <span className="mono-caps block text-faint">
            {row.original.turn_type || '—'}
            {row.original.conv_type ? ` · ${row.original.conv_type}` : ''}
          </span>
        ),
      },
    ],
    [],
  );

  return (
    <AdminPage
      eyebrow="evaluation set"
      title="Dataset review"
      sub="Every question of each eval-loop split beside its gold answer, its gold program, and the per-subagent gold derived from them — the rows every verdict leans on."
      testId="dataset-review"
    >
      <Panel
        title={`${split} split`}
        endpoint={`/eval/dataset?split=${split}`}
        note={SPLIT_BLURB[split]}
        right={
          <span className="type-small text-faint">
            {rows.length} questions · {nConversations} conversations
          </span>
        }
      >
        <div className="mb-3 flex flex-wrap items-center gap-2">
          {SPLITS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSplit(s)}
              className={cn(
                'rounded-[4px] border px-2.5 py-1 mono-caps transition-colors',
                s === split
                  ? 'border-accent text-accent'
                  : 'border-line text-muted hover:border-line-2',
              )}
            >
              {s}
            </button>
          ))}
          <input
            value={needle}
            onChange={(e) => setNeedle(e.target.value)}
            placeholder="filter by report, question, answer, or program"
            className="min-w-[16rem] flex-1 rounded-[4px] border border-line bg-transparent px-2.5 py-1 type-small outline-none placeholder:text-faint focus:border-line-2"
          />
        </div>
        {query.isLoading ? (
          <LoadingRows rows={8} />
        ) : query.error ? (
          <ErrorNote error={query.error} />
        ) : rows.length === 0 ? (
          <EmptyState>no rows match the filter</EmptyState>
        ) : (
          <InstrumentTable
            data={rows}
            columns={columns}
            rowKey={(r) => `${r.report_id}#${r.turn_index}`}
            minWidth={760}
            layout="fixed"
            maxHeight={640}
            emptyLabel="no rows"
            testId="dataset-table"
          />
        )}
      </Panel>
    </AdminPage>
  );
}
