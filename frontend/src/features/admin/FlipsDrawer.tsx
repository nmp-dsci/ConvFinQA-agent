import { useMemo, useState } from 'react';
import type { ColumnDef } from '@tanstack/react-table';
import { Link } from 'react-router-dom';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { cn } from '@/lib/utils';
import { formatFilingId, formatPercent, formatPointsDelta } from '../landing/format';
import { InstrumentTable } from './InstrumentTable';
import { clip, formatCount } from './lib';
import { Verdict } from './ui';
import type { ComparisonResult, Flip } from '../../types';

/**
 * The flips between two versions.
 *
 * This is the promotion contract made visible, and it is deliberately not a
 * footnote. "Beats the champion on average" is the exact claim the comparator
 * refuses: a version that fixes twenty number lookups and breaks twenty-four
 * programs can post a higher mean while being strictly worse to ship. So the
 * pass→fail column is the headline, the fail→pass column sits beside it for
 * fairness, and the verdict repeats the comparator's own sentence rather than
 * paraphrasing it.
 *
 * The rows are the point. Each flip links to the scored turn it came from, so
 * "−24" is a claim a reader can audit rather than a number to be trusted.
 */

function flipColumns(
  candidate: string,
  baseline: string,
  tone: 'good' | 'bad',
): Array<ColumnDef<Flip, unknown>> {
  return [
    {
      id: 'filing',
      header: 'filing · turn',
      accessorFn: (r) => r.report_id,
      meta: { align: 'left', mono: false, width: '150px' },
      cell: ({ row }) => (
        <Link
          to={`/admin/traces/eval?version=${candidate}&report_id=${encodeURIComponent(
            row.original.report_id,
          )}&turn_index=${row.original.q_order}`}
          className="hover:text-amber"
          title={row.original.report_id}
        >
          {formatFilingId(row.original.report_id)}
          <span className="text-faint"> · {row.original.q_order}</span>
        </Link>
      ),
    },
    {
      id: 'question',
      header: 'question',
      accessorFn: (r) => r.question,
      meta: { align: 'left', mono: false, wrap: true, width: '220px' },
      cell: ({ row }) => <span className="text-muted">{clip(row.original.question, 120)}</span>,
    },
    {
      id: 'gold',
      header: 'gold',
      accessorFn: (r) => r.gold_answer,
      meta: { width: '80px' },
      cell: ({ row }) => <span className="text-text">{row.original.gold_answer}</span>,
    },
    {
      id: 'baseline',
      header: baseline,
      accessorFn: (r) => r.baseline_answer,
      meta: { width: '80px' },
      cell: ({ row }) => (
        <span className={tone === 'bad' ? 'text-good' : 'text-bad'}>
          {row.original.baseline_answer || '—'}
        </span>
      ),
    },
    {
      id: 'candidate',
      header: candidate,
      accessorFn: (r) => r.candidate_answer,
      meta: { width: '80px' },
      cell: ({ row }) => (
        <span className={tone === 'bad' ? 'text-bad' : 'text-good'}>
          {row.original.candidate_answer || '—'}
        </span>
      ),
    },
  ];
}

// ---------------------------------------------------------------------------
// The always-visible summary
// ---------------------------------------------------------------------------

export function FlipsSummary({
  comparison,
  onOpen,
}: {
  comparison: ComparisonResult;
  onOpen: () => void;
}) {
  const { regressions, improvements, promotable, reason } = comparison;

  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Verdict ok={promotable}>{promotable ? 'promotable' : 'refused'}</Verdict>
        <span className="type-small text-muted">{reason}</span>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <button
          type="button"
          onClick={onOpen}
          data-testid="flips-open-regressions"
          className={cn(
            'group min-w-0 rounded-md border border-line bg-panel-2 p-2.5 text-left',
            'transition-colors hover:border-bad focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber',
          )}
        >
          <div className="mono-caps">pass → fail</div>
          <div className="type-hud mt-1 text-bad">{formatCount(regressions.length)}</div>
          <div className="type-meta mt-1 text-faint group-hover:text-amber">
            open the rows &rarr;
          </div>
        </button>

        <button
          type="button"
          onClick={onOpen}
          data-testid="flips-open-improvements"
          className={cn(
            'group min-w-0 rounded-md border border-line bg-panel-2 p-2.5 text-left',
            'transition-colors hover:border-good-line focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber',
          )}
        >
          <div className="mono-caps">fail → pass</div>
          <div className="type-hud mt-1 text-good">{formatCount(improvements.length)}</div>
          <div className="type-meta mt-1 text-faint group-hover:text-amber">
            open the rows &rarr;
          </div>
        </button>

        <div className="min-w-0 rounded-md border border-line bg-panel-2 p-2.5">
          <div className="mono-caps">accuracy delta</div>
          <div
            className={cn(
              'type-hud mt-1',
              comparison.accuracy_delta >= 0 ? 'text-good' : 'text-bad',
            )}
          >
            {formatPointsDelta(comparison.accuracy_delta)}
          </div>
          <div className="type-meta mt-1 text-faint">
            {formatPercent(comparison.baseline_accuracy)} →{' '}
            {formatPercent(comparison.candidate_accuracy)}
          </div>
        </div>

        <div className="min-w-0 rounded-md border border-line bg-panel-2 p-2.5">
          <div className="mono-caps">compared on</div>
          <div className="type-hud mt-1 text-text">{formatCount(comparison.n_compared)}</div>
          <div className="type-meta mt-1 text-faint">questions both versions answered</div>
        </div>
      </div>

      <p className="type-meta rounded-[4px] border border-line border-dashed px-2 py-1.5 text-faint">
        Promotion needs accuracy ≥ champion <em>and</em> zero pass→fail flips. Accuracy alone would
        let &ldquo;fixed the numbers, broke the programs&rdquo; through — which is precisely the
        shape of the {formatCount(improvements.length)}/{formatCount(regressions.length)} split
        above. Enforced in <code>tracking/comparator.py</code>, gated in CI by{' '}
        <code>tracking/gate.py</code>.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The drawer
// ---------------------------------------------------------------------------

export function FlipsDrawer({
  comparison,
  open,
  onOpenChange,
}: {
  comparison: ComparisonResult;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [tab, setTab] = useState<'regressions' | 'improvements'>('regressions');
  const { baseline_version: baseline, candidate_version: candidate } = comparison;

  const regressionColumns = useMemo(
    () => flipColumns(candidate, baseline, 'bad'),
    [candidate, baseline],
  );
  const improvementColumns = useMemo(
    () => flipColumns(candidate, baseline, 'good'),
    [candidate, baseline],
  );

  const rows = tab === 'regressions' ? comparison.regressions : comparison.improvements;
  const sliceKeys = Object.keys(comparison.slice_deltas ?? {});

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        data-testid="flips-drawer"
        className="w-full gap-0 overflow-y-auto border-line bg-ground p-0 sm:max-w-[min(96vw,880px)]"
      >
        <SheetHeader className="border-b border-line px-4 py-3">
          <SheetTitle className="type-h2">
            {baseline} → {candidate}
          </SheetTitle>
          <SheetDescription className="type-small text-muted">
            Every question whose result changed between the two versions.{' '}
            {comparison.promotable
              ? 'This pair clears the promotion contract.'
              : 'The gate refused this candidate — the reason is below.'}
          </SheetDescription>
        </SheetHeader>

        <div className="flex min-w-0 flex-col gap-3 px-4 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <Verdict ok={comparison.promotable}>
              {comparison.promotable ? 'promotable' : 'refused'}
            </Verdict>
            <span className="type-small text-muted">{comparison.reason}</span>
          </div>

          <div className="flex flex-wrap gap-1.5" role="tablist">
            {(['regressions', 'improvements'] as const).map((key) => {
              const count =
                key === 'regressions'
                  ? comparison.regressions.length
                  : comparison.improvements.length;
              const active = tab === key;
              return (
                <button
                  key={key}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setTab(key)}
                  className={cn(
                    'rounded-[4px] border px-2 py-1 font-mono text-[10px] tracking-[0.07em] uppercase',
                    active
                      ? 'border-amber-line bg-amber-soft text-amber'
                      : 'border-line text-faint hover:text-text',
                  )}
                >
                  {key === 'regressions' ? 'pass → fail' : 'fail → pass'} · {count}
                </button>
              );
            })}
          </div>

          <InstrumentTable
            key={tab}
            testId={`flips-table-${tab}`}
            data={rows}
            columns={tab === 'regressions' ? regressionColumns : improvementColumns}
            rowKey={(r, i) => `${r.report_id}:${r.q_order}:${i}`}
            minWidth={640}
            emptyLabel={
              tab === 'regressions'
                ? 'no pass→fail flips — this candidate broke nothing it used to get right'
                : 'no fail→pass flips'
            }
          />

          {sliceKeys.length > 0 && (
            <div className="min-w-0 rounded-md border border-line bg-panel p-3">
              <h3 className="type-body mb-2 font-medium text-text">Slice deltas</h3>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {sliceKeys.map((key) => (
                  <div key={key} className="min-w-0">
                    <div className="mono-caps mb-1">{key.replace(/^gold_/, '')}</div>
                    <ul className="flex flex-col gap-0.5">
                      {Object.entries(comparison.slice_deltas[key]).map(([label, delta]) => (
                        <li
                          key={label}
                          className="flex items-baseline justify-between gap-3 border-b border-line py-0.5 last:border-0"
                        >
                          <span className="type-small text-muted">{label}</span>
                          <span
                            className={cn(
                              'type-num text-[11px]',
                              delta > 0 ? 'text-good' : delta < 0 ? 'text-bad' : 'text-faint',
                            )}
                          >
                            {formatPointsDelta(delta)}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          )}

          {comparison.notes.length > 0 && (
            <ul className="type-meta flex flex-col gap-1 text-faint">
              {comparison.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
