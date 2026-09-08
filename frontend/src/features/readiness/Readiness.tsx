import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { shippedCount, useReadiness } from './api';
import type { Readiness, ReadinessRow, ReadinessStatus } from './api';

/**
 * The scorecard, in three sizes from one file.
 *
 *  - `ReadinessStrip`: nine cells — ref, glyph, one word — for the landing's
 *    star. Each cell links to the row's proof page.
 *  - `ReadinessBlock`: the two halves the rubric defines, each row with its
 *    status word, the 40-word "how" and its proof links. Architecture §13.
 *  - `/admin/readiness` (`ReadinessRoute`) adds the question each dimension
 *    asks and the proof paths behind a disclosure.
 *
 * Status is a word and a glyph, never a colour alone; the glyphs ● ◐ ○ —
 * survive greyscale, which is the portfolio's own rule.
 */

const STATUS_CLASS: Record<ReadinessStatus, string> = {
  shipped: 'border-good-line bg-good/10 text-good',
  partial: 'border-amber-line bg-amber-soft text-amber',
  designed: 'border-dashed border-line-2 text-muted',
  na: 'border-line text-faint',
};

function StatusChip({ data, status }: { data: Readiness; status: ReadinessStatus }) {
  const s = data.statuses[status];
  return (
    <span
      data-status={status}
      title={s.blurb}
      className={cn(
        'type-num type-meta inline-flex items-center gap-1 rounded-[4px] border px-1.5 py-0.5',
        STATUS_CLASS[status],
      )}
    >
      <span aria-hidden>{s.glyph}</span>
      {s.label}
    </span>
  );
}

function ScoreLine({ data }: { data: Readiness }) {
  const { shipped, total } = shippedCount(data.rows);
  return (
    <>
      <span className="type-num">{shipped}</span> / <span className="type-num">{total}</span>{' '}
      shipped · rung <span className="type-num">{data.rung}</span> · measured{' '}
      <span className="type-num">{data.measured}</span>
    </>
  );
}

// ---------------------------------------------------------------------------
// Strip
// ---------------------------------------------------------------------------

export function ReadinessStrip({ className }: { className?: string }) {
  const query = useReadiness();
  const data = query.data;

  if (query.isLoading) {
    return (
      <div className={cn('grid grid-cols-9 gap-1', className)} aria-label="loading">
        {Array.from({ length: 9 }, (_, i) => (
          <div key={i} className="h-11 animate-pulse rounded bg-panel-2" />
        ))}
      </div>
    );
  }
  if (!data) {
    return (
      <p className={cn('type-meta text-faint', className)} data-testid="readiness-strip-absent">
        The readiness scorecard did not load — /eval/readiness returned nothing for this deployment.
      </p>
    );
  }

  return (
    <div className={className} data-testid="readiness-strip">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="mono-caps">
          production readiness · <ScoreLine data={data} />
        </span>
        <Link
          to="/admin/readiness"
          className="type-meta text-amber underline decoration-amber-line underline-offset-4 hover:decoration-amber"
        >
          the nine dimensions →
        </Link>
      </div>
      <ol className="mt-2 grid grid-cols-9 gap-1">
        {data.rows.map((row) => {
          const s = data.statuses[row.status];
          return (
            <li key={row.ref} className="min-w-0">
              <Link
                to={row.app[0] ?? '/admin/readiness'}
                data-ref={row.ref}
                data-status={row.status}
                title={`${row.ref} ${row.label} — ${s.label}: ${s.blurb}`}
                className={cn(
                  'flex h-11 flex-col items-center justify-center rounded-[4px] border transition-colors',
                  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber',
                  STATUS_CLASS[row.status],
                  'hover:border-amber',
                )}
              >
                <span aria-hidden className="type-num type-small leading-none">
                  {s.glyph}
                </span>
                <span className="type-num type-meta mt-0.5 leading-none">{row.ref}</span>
                <span className="sr-only">
                  {row.label}: {s.label}
                </span>
              </Link>
            </li>
          );
        })}
      </ol>
      <p className="type-num type-meta mt-1.5 text-faint">
        {data.rows.map((r) => r.short.replace(/­/g, '').toLowerCase()).join(' · ')}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

function RowCard({
  data,
  row,
  detailed,
}: {
  data: Readiness;
  row: ReadinessRow;
  detailed: boolean;
}) {
  return (
    <li
      data-testid={`readiness-${row.ref}`}
      data-status={row.status}
      className="min-w-0 rounded-md border border-line bg-panel p-3.5"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="type-body font-medium text-text">
          <span className="type-num mr-2 text-faint">{row.ref}</span>
          {row.label}
        </span>
        <StatusChip data={data} status={row.status} />
      </div>
      {detailed && <p className="type-small mt-1 text-faint italic">{row.question}</p>}
      <p className="type-small mt-1.5 text-muted">{row.how}</p>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
        {row.app.map((to) => (
          <Link
            key={to}
            to={to}
            className="type-num type-meta text-amber underline decoration-amber-line underline-offset-4 hover:decoration-amber"
          >
            {to} →
          </Link>
        ))}
      </div>
      {detailed && row.proof.length > 0 && (
        <details className="mt-2">
          <summary className="mono-caps cursor-pointer select-none">
            proof · {row.proof.length} path{row.proof.length === 1 ? '' : 's'} in the repo
          </summary>
          <ul className="mt-1 space-y-0.5">
            {row.proof.map((p) => (
              <li key={p} className="type-num type-meta break-all text-muted">
                {p}
              </li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}

export function ReadinessBlock({ detailed = false }: { detailed?: boolean }) {
  const query = useReadiness();
  const data = query.data;

  if (query.isLoading) {
    return (
      <div className="grid gap-2" aria-label="loading">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="h-20 animate-pulse rounded-md bg-panel" />
        ))}
      </div>
    );
  }
  if (!data) {
    return (
      <p className="type-small rounded-md border border-dashed border-line-2 p-3 text-faint">
        The readiness scorecard did not load — /eval/readiness returned nothing for this
        deployment. It is a committed file, so a fresh checkout carries it.
      </p>
    );
  }

  return (
    <div data-testid="readiness-block" className="grid gap-5">
      <p className="type-small text-muted">
        <ScoreLine data={data} />. {' '}
        {Object.values(data.statuses).map((s, i) => (
          <span key={s.label} className="type-num">
            {i > 0 ? ' · ' : ''}
            {s.glyph} {s.label} — {s.blurb}
          </span>
        ))}
      </p>
      {data.halves.map((half, i) => {
        const from = i === 0 ? 0 : data.halves[i - 1].upto;
        const rows = data.rows.filter((r) => r.order > from && r.order <= half.upto);
        return (
          <section key={half.label} className="min-w-0">
            <div className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span className="type-body font-medium text-text">{half.label}</span>
              <span className="mono-caps">{half.range}</span>
              <span className="type-meta">{half.blurb}</span>
            </div>
            <ul className="grid gap-2 lg:grid-cols-2">
              {rows.map((row) => (
                <RowCard key={row.ref} data={data} row={row} detailed={detailed} />
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
