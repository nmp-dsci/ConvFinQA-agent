import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { PageHead } from '@/kit/ui';
import { NO_VALUE } from '../landing/format';

export { Lamp, LampRow } from '@/components/console';
export type { LampTone } from '@/components/console';

/**
 * The evidence pages' furniture, on the house kit (ui/DESIGN.md): a page opens as a slide
 * (eyebrow → claim → lead), then panels — titled cards naming the path their rows come from —
 * holding house tables, lamps and stat cells.
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
  /** The mono label above the claim. */
  eyebrow?: string;
  title: ReactNode;
  /** The sentence the page settles — rendered at the lede step. */
  sub: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  testId: string;
}) {
  return (
    <div className="stack" data-testid={testId}>
      <PageHead eyebrow={eyebrow} title={title} lead={sub}>
        {actions && <div className="chips">{actions}</div>}
      </PageHead>
      {children}
    </div>
  );
}

/** Two columns that become one below ~700px. */
export function TwoUp({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={`grid2${className ? ` ${className}` : ''}`}>{children}</div>;
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

/** A titled card. On its right, the path its rows come from — provenance printed, not implied. */
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
    <section data-testid={testId} className={`panel${className ? ` ${className}` : ''}`}>
      <div className="panel-head">
        <h3>{title}</h3>
        {(right || endpoint) && (
          <span className="right">
            {right}
            {endpoint && <EndpointChip path={endpoint} to={to} />}
          </span>
        )}
      </div>
      {note && <p className="note">{note}</p>}
      {children}
    </section>
  );
}

/** Where this panel's rows come from, or lead to. */
export function EndpointChip({ path, to }: { path: string; to?: string }) {
  return to ? (
    <Link to={to} className="endpoint">
      {path}
    </Link>
  ) : (
    <span className="endpoint">{path}</span>
  );
}

// ---------------------------------------------------------------------------
// Stat cells
// ---------------------------------------------------------------------------

export interface StatCell {
  label: string;
  value: string;
  /** Required whenever `value` is an em dash. Never a bare blank. */
  reason?: string;
  tone?: 'good' | 'bad' | 'plain';
}

/** A tight row of small figures. */
export function StatCells({ cells, columns = 3 }: { cells: StatCell[]; columns?: number }) {
  return (
    <div className="stats" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
      {cells.map((cell) => {
        const absent = cell.value === NO_VALUE;
        const tone = absent ? 'absent' : cell.tone === 'good' ? 'good' : cell.tone === 'bad' ? 'bad' : '';
        return (
          <div key={cell.label} className="stat" data-absent={absent ? 'true' : 'false'}>
            <div className={`n ${tone}`}>{cell.value}</div>
            <span className="label" title={absent ? cell.reason : cell.label}>
              {cell.label}
            </span>
            {absent && cell.reason && <div className="why">{cell.reason}</div>}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Verdicts, notes, states
// ---------------------------------------------------------------------------

/** Passed / failed as a word with its glyph — never colour alone. */
export function Verdict({ ok, children, className }: { ok: boolean | null; children: ReactNode; className?: string }) {
  const tone = ok === null ? '' : ok ? 'ok' : 'fail';
  return (
    <span data-verdict={ok === null ? 'unknown' : ok ? 'pass' : 'fail'} className={`tag ${tone}${className ? ` ${className}` : ''}`}>
      <span aria-hidden>{ok === null ? '·' : ok ? '✓' : '✗'}</span> {children}
    </span>
  );
}

/** An explanatory line that has to travel with a number. */
export function Caveat({ children }: { children: ReactNode }) {
  return <p className="callout">{children}</p>;
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>;
}

export function LoadingRows({ rows = 4 }: { rows?: number }) {
  void rows;
  return (
    <p className="empty" aria-busy="true">
      Loading…
    </p>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="callout err">
      <span className="ttl">Read failed</span>
      <span className="wrap-any">{message}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The demo gate, layer two
// ---------------------------------------------------------------------------

/**
 * A real `<fieldset disabled>` around every write control, with the reason printed next to it.
 * `disabled` on a fieldset disables every control inside it at the platform level — focus,
 * Enter, screen readers — which `pointer-events: none` would only pretend to. One of three
 * layers: the server refuses the same write with a 403 or 501 even if a client forges it.
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
    <fieldset disabled={!enabled} data-testid={testId} data-write-enabled={enabled ? 'true' : 'false'}>
      <div className="row">{children}</div>
      {!enabled && <p className="small muted">{reason}</p>}
    </fieldset>
  );
}
