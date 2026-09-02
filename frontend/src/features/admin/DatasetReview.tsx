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
 * gold answer and gold program, filterable, nothing derived.
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
  return (
    <code
      className={cn('font-mono text-[11px] break-all', isSelection && 'text-muted')}
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

  const columns = useMemo<Array<ColumnDef<DatasetRow, unknown>>>(
    () => [
      {
        header: 'report',
        accessorKey: 'report_id',
        cell: ({ row }) => (
          <span className="font-mono text-[11px]">{row.original.report_id}</span>
        ),
      },
      {
        header: 'q#',
        accessorKey: 'turn_index',
        cell: ({ row }) => <span className="font-mono">q{row.original.turn_index}</span>,
      },
      {
        header: 'question',
        accessorKey: 'question',
        cell: ({ row }) => (
          <span className="block max-w-[34rem] whitespace-normal">
            {row.original.question}
          </span>
        ),
      },
      {
        header: 'gold answer',
        accessorKey: 'gold_answer',
        cell: ({ row }) => (
          <span className="font-mono text-[12px]">{row.original.gold_answer}</span>
        ),
      },
      {
        header: 'gold program',
        accessorKey: 'gold_program',
        cell: ({ row }) =>
          programCell(row.original.gold_program, row.original.turn_type),
      },
      {
        header: 'type',
        accessorKey: 'turn_type',
        cell: ({ row }) => (
          <span className="mono-caps text-faint">
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
      sub="Every question of each eval-loop split beside its gold answer and gold program — the rows every verdict leans on."
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
            minWidth={900}
            maxHeight={640}
            emptyLabel="no rows"
            testId="dataset-table"
          />
        )}
      </Panel>
    </AdminPage>
  );
}
