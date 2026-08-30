import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { NO_VALUE } from '../landing/format';

/**
 * The debrief's furniture.
 *
 * One idea runs through all of it: every claim on this page is either cited to
 * the paper, read live from this deployment, or read from a file committed in
 * the repo — and the reader can tell which at a glance, without hunting for a
 * footnote. `Cite` and `Provenance` are the two components that make that a
 * property of the layout rather than a discipline someone has to remember.
 */

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

export interface SectionProps {
  id: string;
  /** Printed as `01`, `02`… so the twelve sections read as one document. */
  index: number;
  eyebrow: string;
  title: string;
  lede?: ReactNode;
  children: ReactNode;
}

export function Section({ id, index, eyebrow, title, lede, children }: SectionProps) {
  const n = String(index).padStart(2, '0');
  return (
    <section
      id={id}
      data-testid={`system-section-${id}`}
      // Anchor links land under the sticky top bar rather than behind it.
      className="scroll-mt-6 border-t border-line pt-8 first:border-t-0 first:pt-0"
    >
      <p className="mono-caps">
        {n} · {eyebrow}
      </p>
      <h2 className="type-h2 mt-2 max-w-[46ch]">{title}</h2>
      {lede && <div className="type-lede mt-3 max-w-[62ch]">{lede}</div>}
      <div className="mt-5 space-y-5">{children}</div>
    </section>
  );
}

export function Panel({
  children,
  className,
  tone = 'panel',
}: {
  children: ReactNode;
  className?: string;
  tone?: 'panel' | 'dashed';
}) {
  return (
    <div
      className={cn(
        'min-w-0 rounded-md border p-4',
        tone === 'dashed' ? 'border-dashed border-line-2 bg-panel/60' : 'border-line bg-panel',
        className,
      )}
    >
      {children}
    </div>
  );
}

export function PanelTitle({ children }: { children: ReactNode }) {
  return <div className="mono-caps mb-2">{children}</div>;
}

/** Body prose at the reading size, held to a comfortable measure. */
export function Prose({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn('type-body max-w-[68ch] text-muted', className)}>{children}</p>;
}

/**
 * A callout. `broken` is a distinct tone from `warn` on purpose — "this does
 * not work today" and "read this carefully" are different facts and a reader
 * skimming must be able to tell them apart without reading the words.
 */
export function Callout({
  tone = 'note',
  title,
  children,
}: {
  tone?: 'note' | 'warn' | 'broken';
  title?: ReactNode;
  children: ReactNode;
}) {
  const border =
    tone === 'broken' ? 'border-bad/55' : tone === 'warn' ? 'border-amber-line' : 'border-line-2';
  const dot = tone === 'broken' ? 'bg-bad' : tone === 'warn' ? 'bg-amber' : 'bg-info';
  return (
    <div className={cn('min-w-0 rounded-md border border-dashed bg-panel/60 p-3.5', border)}>
      {title && (
        <div className="mb-1.5 flex items-center gap-2">
          <span aria-hidden className={cn('size-1.5 shrink-0 rounded-full', dot)} />
          <span className="type-small font-medium text-text">{title}</span>
        </div>
      )}
      <div className="type-small text-muted [&>p+p]:mt-2">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/** `Table 5` — where in the paper a figure came from. */
export function Cite({ children }: { children: ReactNode }) {
  return (
    <span className="type-num whitespace-nowrap rounded-sm border border-line-2 px-1 py-px text-[9.5px] tracking-[0.04em] text-faint">
      {children}
    </span>
  );
}

export type Origin = 'paper' | 'live' | 'committed' | 'code';

const ORIGIN_LABEL: Record<Origin, string> = {
  paper: 'from the paper',
  live: 'read live from this deployment',
  committed: 'recomputed from a committed artefact',
  code: 'read from the source',
};

const ORIGIN_CLASS: Record<Origin, string> = {
  paper: 'border-line-2 text-faint',
  live: 'border-good-line text-good',
  committed: 'border-info-line text-info',
  code: 'border-amber-line text-amber',
};

/**
 * Where a block of numbers came from. Printed on every table and chart.
 *
 * The four origins are not decoration: "77.1% from the paper" and "77.1% read
 * live" are different claims, and a page that mixes them without saying so is
 * exactly the kind of thing this project exists to argue against.
 */
export function Provenance({ origin, children }: { origin: Origin; children?: ReactNode }) {
  return (
    // Deliberately NOT a flex row: the badge sits inline so the sentence after
    // it flows and wraps as one paragraph. As a flex container each text node
    // became its own item, which stranded punctuation at the start of a line.
    <p className="type-meta mt-2 text-faint">
      <span
        className={cn(
          'mono-caps mr-1.5 inline-block rounded-sm border px-1 py-px align-baseline text-[9.5px]',
          ORIGIN_CLASS[origin],
        )}
      >
        {ORIGIN_LABEL[origin]}
      </span>
      {children}
    </p>
  );
}

// ---------------------------------------------------------------------------
// Live values
// ---------------------------------------------------------------------------

/**
 * A value read from the backend, or an em dash and the reason it is absent.
 *
 * Never renders a placeholder number and never renders zero for "unknown". The
 * page is required to stay readable with the API stopped, and this is the
 * component that makes that a rendering rather than a blank.
 */
export function Live({
  value,
  reason = 'backend not reachable',
  className,
}: {
  value: string | null | undefined;
  reason?: string;
  className?: string;
}) {
  if (value === null || value === undefined || value === '' || value === NO_VALUE) {
    return (
      <span className={cn('type-num text-faint', className)} title={reason}>
        {NO_VALUE}
      </span>
    );
  }
  return <span className={cn('type-num text-text', className)}>{value}</span>;
}

/**
 * A monospaced inline literal — a file path, a command, an identifier.
 *
 * `break-words` is load-bearing, not cosmetic: several of these are long
 * unbroken paths, and at a 420px column one of them pushes the whole section
 * sideways. There is no space in `evaluation/predictions/…_joined.csv` for the
 * browser to break at unless it is told it may break anywhere.
 */
export function Mono({ children }: { children: ReactNode }) {
  return (
    <span className="type-num rounded-sm bg-panel-2 px-1 py-px text-[11px] break-words text-text">
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/**
 * Every table on this page scrolls inside its own box.
 *
 * The page itself must never scroll sideways — at 420px a nine-column
 * benchmark table would otherwise push the whole document off screen.
 */
export function ScrollX({ children }: { children: ReactNode }) {
  return (
    <div className="min-w-0 overflow-x-auto pb-1">
      <div className="min-w-[36rem]">{children}</div>
    </div>
  );
}

export function Table({ children }: { children: ReactNode }) {
  return <table className="w-full border-collapse text-left">{children}</table>;
}

export function Th({
  children,
  numeric = false,
  className,
}: {
  children: ReactNode;
  numeric?: boolean;
  className?: string;
}) {
  return (
    <th
      className={cn(
        'mono-caps border-b border-line-2 pb-1.5 pr-3 align-bottom font-normal',
        numeric && 'text-right',
        className,
      )}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  numeric = false,
  className,
}: {
  /** Optional: an empty cell keeps a colspan-free table aligned when a row
   *  carries a single explanatory message instead of values. */
  children?: ReactNode;
  numeric?: boolean;
  className?: string;
}) {
  return (
    <td
      className={cn(
        'border-b border-line py-2 pr-3 align-top type-small text-muted',
        numeric && 'type-num text-right text-text',
        className,
      )}
    >
      {children}
    </td>
  );
}

// ---------------------------------------------------------------------------
// Small pieces
// ---------------------------------------------------------------------------

/** A labelled fact in a definition grid. */
export function Field({
  label,
  children,
  note,
}: {
  label: string;
  children: ReactNode;
  note?: ReactNode;
}) {
  return (
    <div className="min-w-0">
      <div className="mono-caps">{label}</div>
      <div className="type-body mt-0.5 break-words text-text">{children}</div>
      {note && <div className="type-meta mt-0.5">{note}</div>}
    </div>
  );
}

/**
 * A status lamp. Shape carries the state as well as colour: a solid ring is
 * good, a dashed ring is a replay or a deferral, a filled dot is a failure.
 */
export function Lamp({ state }: { state: 'good' | 'replay' | 'bad' | 'idle' }) {
  const base = 'inline-block size-2 shrink-0 rounded-full border';
  if (state === 'good') return <span aria-hidden className={cn(base, 'border-good bg-good/35')} />;
  if (state === 'replay')
    return <span aria-hidden className={cn(base, 'border-dashed border-amber bg-transparent')} />;
  if (state === 'bad') return <span aria-hidden className={cn(base, 'border-bad bg-bad')} />;
  return <span aria-hidden className={cn(base, 'border-line-2 bg-transparent')} />;
}
