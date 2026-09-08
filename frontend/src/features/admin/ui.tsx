import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { NextFooter, PageHeader } from '@/components/console';
import { NO_VALUE } from '../landing/format';

export { Lamp, LampRow } from '@/components/console';
export type { LampTone } from '@/components/console';

/**
 * The admin console's chrome, built once.
 *
 * The mock's admin frame is a rail plus a page of panels: an `h2`, a one-line
 * `sub` that says what the page is for, a lamp strip, tiles, then two columns
 * of `.pan` panels holding `table.tb` instrument tables. The shell already owns
 * the rail, so what is reproduced here is the page: same hierarchy, same
 * densities, expressed as components rather than as the mock's literal HTML.
 */

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function AdminPage({
  eyebrow,
  title,
  sub,
  actions,
  children,
  testId,
}: {
  /** Kept for callers; the header now derives its eyebrow from the nav. */
  eyebrow?: string;
  title: string;
  /** The sentence the page settles — rendered at the lede step. */
  sub: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  testId: string;
}) {
  void eyebrow;
  return (
    <div className="h-full overflow-y-auto bg-ground" data-testid={testId}>
      <div className="mx-auto flex max-w-[1560px] min-w-0 flex-col gap-4 px-3 py-5 sm:px-5">
        <PageHeader title={title} verdict={sub} chips={actions} />
        {children}
        <NextFooter />
      </div>
    </div>
  );
}

/** A two-column band that becomes one column below `lg`. */
export function TwoUp({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('grid min-w-0 grid-cols-1 gap-3 lg:grid-cols-2', className)}>{children}</div>
  );
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

/**
 * `.pan` from the mock: a lit surface at `--panel` on the `--ground` page, a
 * 12px semibold title, and — on the right of the title — the path where this
 * panel's rows live.
 */
export function Panel({
  title,
  endpoint,
  to,
  note,
  right,
  children,
  className,
  testId,
}: {
  title: ReactNode;
  /** The API path or page path this panel is a view of. Printed in mono. */
  endpoint?: string;
  /** When set, the endpoint chip becomes a link to that in-app route. */
  to?: string;
  /** A line under the title: provenance, a caveat, a source note. */
  note?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  testId?: string;
}) {
  return (
    <section
      data-testid={testId}
      className={cn('min-w-0 rounded-md border border-line bg-panel p-3', className)}
    >
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h2 className="type-body min-w-0 font-medium text-text">{title}</h2>
        <div className="flex shrink-0 items-center gap-2">
          {right}
          {endpoint && <EndpointChip path={endpoint} to={to} />}
        </div>
      </div>
      {note && <p className="type-meta mb-2 text-faint">{note}</p>}
      {children}
    </section>
  );
}

/** The mock's `.kbd` chip: where this panel's rows come from, or lead to. */
export function EndpointChip({ path, to }: { path: string; to?: string }) {
  const body = (
    <span
      className={cn(
        'inline-block rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5',
        'font-mono text-[10px] whitespace-nowrap text-faint',
        to && 'transition-colors hover:border-amber-line hover:text-amber',
      )}
    >
      {path}
    </span>
  );
  return to ? <Link to={to}>{body}</Link> : body;
}

// ---------------------------------------------------------------------------
// Stat cells — the mock's `.tot`
// ---------------------------------------------------------------------------

export interface StatCell {
  label: string;
  value: string;
  /** Required whenever `value` is an em dash. Never a bare blank. */
  reason?: string;
  tone?: 'good' | 'bad' | 'plain';
}

/** A tight row of small figures — the inspector's totals strip in the mock. */
export function StatCells({ cells, columns = 3 }: { cells: StatCell[]; columns?: number }) {
  return (
    <div
      className="grid gap-1.5"
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {cells.map((cell) => {
        const absent = cell.value === NO_VALUE;
        return (
          <div
            key={cell.label}
            data-absent={absent ? 'true' : 'false'}
            className="min-w-0 rounded-[5px] border border-line-2 bg-panel-2 px-2 py-1.5"
          >
            <div
              className={cn(
                'type-num truncate text-[12.5px] font-medium',
                absent
                  ? 'text-faint'
                  : cell.tone === 'good'
                    ? 'text-good'
                    : cell.tone === 'bad'
                      ? 'text-bad'
                      : 'text-text',
              )}
            >
              {cell.value}
            </div>
            <div className="mono-caps mt-0.5 truncate" title={absent ? cell.reason : cell.label}>
              {cell.label}
            </div>
            {absent && cell.reason && (
              <div className="type-meta mt-0.5 text-faint">{cell.reason}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Verdicts and small badges
// ---------------------------------------------------------------------------

export function Verdict({
  ok,
  children,
  className,
}: {
  ok: boolean | null;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      data-verdict={ok === null ? 'unknown' : ok ? 'pass' : 'fail'}
      className={cn(
        'inline-flex items-center gap-1 rounded-[4px] border px-1.5 py-0.5 font-mono text-[10px] tracking-[0.06em] uppercase',
        ok === null && 'border-line-2 text-faint',
        ok === true && 'border-good-line text-good',
        ok === false && 'border-bad text-bad',
        className,
      )}
    >
      <span aria-hidden>{ok === null ? '·' : ok ? '✓' : '✗'}</span>
      {children}
    </span>
  );
}

/** An explanatory line that has to travel with a number. */
export function Caveat({ children }: { children: ReactNode }) {
  return (
    <p className="type-meta mt-2 rounded-[4px] border border-line border-dashed px-2 py-1.5 text-faint">
      {children}
    </p>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-[5px] border border-line border-dashed px-3 py-6 text-center">
      <p className="type-small text-faint">{children}</p>
    </div>
  );
}

export function LoadingRows({ rows = 4 }: { rows?: number }) {
  return (
    <div className="flex flex-col gap-1.5" aria-label="loading">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="h-5 animate-pulse rounded-[3px] bg-panel-2" />
      ))}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-[5px] border border-bad px-3 py-2">
      <div className="mono-caps text-bad">read failed</div>
      <p className="type-small mt-1 break-words text-muted">{message}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The demo gate, layer two
// ---------------------------------------------------------------------------

/**
 * A real `<fieldset disabled>` around every write control, with the reason
 * printed next to it.
 *
 * `pointer-events: none` would look the same and be a lie — the control would
 * still be focusable, still submit on Enter, and still be enabled to a screen
 * reader. `disabled` on a fieldset disables every form control inside it at the
 * platform level, which is the only version of this that is true.
 *
 * This is one of three layers, not the whole gate: the server refuses the same
 * write with a 403 or 501 even if a client forges the request, and the write
 * routes are the only place that decision is actually made.
 */
export function WriteGate({
  enabled,
  reason,
  children,
  testId,
}: {
  enabled: boolean;
  /** Why writes are refused. Shown whenever `enabled` is false. */
  reason: string;
  children: ReactNode;
  testId: string;
}) {
  return (
    <fieldset
      disabled={!enabled}
      data-testid={testId}
      data-write-enabled={enabled ? 'true' : 'false'}
      className={cn('min-w-0 border-0 p-0', !enabled && 'opacity-70')}
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2">{children}</div>
      {!enabled && (
        <p className="type-meta mt-1.5 flex items-start gap-1.5 text-faint">
          <span aria-hidden className="mt-[3px] size-1.5 shrink-0 rounded-full border border-amber border-dashed" />
          <span>{reason}</span>
        </p>
      )}
    </fieldset>
  );
}
