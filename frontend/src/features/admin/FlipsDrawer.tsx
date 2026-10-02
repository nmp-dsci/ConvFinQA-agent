import { Fragment, useMemo, useState } from 'react';
import type { ColumnDef } from '@tanstack/react-table';
import { Link } from 'react-router-dom';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { formatFilingId, formatPercent, formatPointsDelta } from '../landing/format';
import { InstrumentTable } from './InstrumentTable';
import { clip, formatCount, versionLabel } from './lib';
import { Verdict } from './ui';
import type { ComparisonResult, Flip } from '../../types';
import './runs.css';

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
          to={`/runs/traces/eval?version=${candidate}&report_id=${encodeURIComponent(
            row.original.report_id,
          )}&turn_index=${row.original.q_order}`}
          title={row.original.report_id}
        >
          {formatFilingId(row.original.report_id)}
          <span className="muted"> · {row.original.q_order}</span>
        </Link>
      ),
    },
    {
      id: 'question',
      header: 'question',
      accessorFn: (r) => r.question,
      meta: { align: 'left', mono: false, wrap: true, width: '220px' },
      cell: ({ row }) => clip(row.original.question, 120),
    },
    {
      id: 'gold',
      header: 'gold',
      accessorFn: (r) => r.gold_answer,
      meta: { width: '80px' },
      cell: ({ row }) => row.original.gold_answer,
    },
    {
      id: 'baseline',
      header: versionLabel(baseline),
      accessorFn: (r) => r.baseline_answer,
      meta: { width: '80px' },
      cell: ({ row }) => (
        <span className={tone === 'bad' ? 'v-ok' : 'v-fail'}>
          {row.original.baseline_answer || '—'} {tone === 'bad' ? '✓' : '✗'}
        </span>
      ),
    },
    {
      id: 'candidate',
      header: versionLabel(candidate),
      accessorFn: (r) => r.candidate_answer,
      meta: { width: '80px' },
      cell: ({ row }) => (
        <span className={tone === 'bad' ? 'v-fail' : 'v-ok'}>
          {row.original.candidate_answer || '—'} {tone === 'bad' ? '✗' : '✓'}
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
  const delta = comparison.accuracy_delta;

  return (
    <>
      <p className="verdict-line">
        <Verdict ok={promotable}>{promotable ? 'promotable' : 'refused'}</Verdict>
        <span className="small">{reason}</span>
      </p>

      <div className="kpis flips">
        <button type="button" onClick={onOpen} data-testid="flips-open-regressions" className="kpi">
          <div className={`n ${regressions.length > 0 ? 'fail' : 'ok'}`}>
            {formatCount(regressions.length)}
          </div>
          <div className="b">
            pass → fail of {formatCount(comparison.n_compared)} compared
          </div>
          <div className="act">open the rows →</div>
        </button>

        <button type="button" onClick={onOpen} data-testid="flips-open-improvements" className="kpi">
          <div className="n ok">{formatCount(improvements.length)}</div>
          <div className="b">
            fail → pass of {formatCount(comparison.n_compared)} compared
          </div>
          <div className="act">open the rows →</div>
        </button>

        <div className="kpi">
          <div className={`n ${delta >= 0 ? 'ok' : 'fail'}`}>{formatPointsDelta(delta)}</div>
          <div className="b">
            accuracy, {formatPercent(comparison.baseline_accuracy)} →{' '}
            {formatPercent(comparison.candidate_accuracy)}
          </div>
        </div>

        <div className="kpi">
          <div className="n">{formatCount(comparison.n_compared)}</div>
          <div className="b">questions both versions answered</div>
        </div>
      </div>

      <p className="callout note">
        Promotion needs accuracy ≥ champion <em>and</em> zero pass→fail flips. Accuracy alone would
        let &ldquo;fixed the numbers, broke the programs&rdquo; through — which is precisely the
        shape of the {formatCount(improvements.length)}/{formatCount(regressions.length)} split
        above. Enforced in <code>tracking/comparator.py</code>, gated in CI by{' '}
        <code>tracking/gate.py</code>.
      </p>
    </>
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
      <SheetContent side="right" data-testid="flips-drawer" className="flips-sheet">
        <SheetHeader>
          <SheetTitle>
            {versionLabel(baseline)} → {versionLabel(candidate)}
          </SheetTitle>
          <SheetDescription>
            Every question whose result changed between the two versions.{' '}
            {comparison.promotable
              ? 'This pair clears the promotion contract.'
              : 'The gate refused this candidate — the reason is below.'}
          </SheetDescription>
        </SheetHeader>

        <div className="sheet-body">
          <p className="verdict-line">
            <Verdict ok={comparison.promotable}>
              {comparison.promotable ? 'promotable' : 'refused'}
            </Verdict>
            <span className="small">{comparison.reason}</span>
          </p>

          <div className="seg" role="tablist" aria-label="Which flips">
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
                  className={active ? 'on' : undefined}
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
            <section>
              <h3>Slice deltas — where the accuracy moved</h3>
              <div className="grid2">
                {sliceKeys.map((key) => (
                  <div key={key}>
                    <p className="label">{key.replace(/^gold_/, '')}</p>
                    <dl className="kv">
                      {Object.entries(comparison.slice_deltas[key]).map(([label, delta]) => (
                        <Fragment key={label}>
                          <dt>{label}</dt>
                          <dd className={`num ${delta > 0 ? 'v-ok' : delta < 0 ? 'v-fail' : 'muted'}`}>
                            {delta > 0 ? '▲ ' : delta < 0 ? '▼ ' : ''}
                            {formatPointsDelta(delta)}
                          </dd>
                        </Fragment>
                      ))}
                    </dl>
                  </div>
                ))}
              </div>
            </section>
          )}

          {comparison.notes.length > 0 && (
            <ul className="notes">
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
