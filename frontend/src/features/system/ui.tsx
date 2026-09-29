import type { ReactNode } from 'react';
import { NO_VALUE } from '../landing/format';

/**
 * The long-form pages' furniture (Agent, Scoring, Leaderboard), on the house kit.
 *
 * One idea runs through all of it: every claim is either cited to the paper, read live from this
 * deployment, recomputed from a committed file, or read from the source — and the reader can tell
 * which at a glance. `Cite` and `Provenance` make that a property of the layout.
 */

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

export interface SectionProps {
  id: string;
  /** Printed as `01`, `02`… so the sections read as one document. */
  index: number;
  eyebrow: string;
  title: string;
  lede?: ReactNode;
  children: ReactNode;
}

export function Section({ id, index, eyebrow, title, lede, children }: SectionProps) {
  const n = String(index).padStart(2, '0');
  return (
    <section id={id} data-testid={`system-section-${id}`} className="doc-section">
      <p className="label">
        {n} · {eyebrow}
      </p>
      <h2>{title}</h2>
      {lede && <div className="lead">{lede}</div>}
      <div className="body">{children}</div>
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
  return <div className={`${tone === 'dashed' ? 'empty' : 'panel'}${className ? ` ${className}` : ''}`}>{children}</div>;
}

export function PanelTitle({ children }: { children: ReactNode }) {
  return <p className="label fig-title">{children}</p>;
}

/** Body prose at the reading measure. */
export function Prose({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={className}>{children}</p>;
}

/**
 * A callout. `broken` is a distinct tone from `warn` on purpose — "this does not work today" and
 * "read this carefully" are different facts, told apart by the left rule and the title's word.
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
  const cls = tone === 'broken' ? 'callout broken' : tone === 'warn' ? 'callout' : 'callout note';
  return (
    <div className={cls} data-tone={tone}>
      {title && <span className="ttl">{title}</span>}
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/** `Table 5` — where in the paper a figure came from. */
export function Cite({ children }: { children: ReactNode }) {
  return <span className="cite">{children}</span>;
}

export type Origin = 'paper' | 'live' | 'committed' | 'code';

const ORIGIN_LABEL: Record<Origin, string> = {
  paper: 'from the paper',
  live: 'read live from this deployment',
  committed: 'recomputed from a committed artefact',
  code: 'read from the source',
};

/**
 * Where a block of numbers came from. "77.1% from the paper" and "77.1% read live" are different
 * claims, and a page that mixes them without saying so is what this project argues against.
 */
export function Provenance({ origin, children }: { origin: Origin; children?: ReactNode }) {
  return (
    <p className="provenance">
      <span className={`origin ${origin}`}>{ORIGIN_LABEL[origin]}</span>
      {children}
    </p>
  );
}

// ---------------------------------------------------------------------------
// Live values
// ---------------------------------------------------------------------------

/** A value read from the backend, or `—` and the reason it is absent. Never a placeholder. */
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
      <span className={`live absent${className ? ` ${className}` : ''}`} title={reason}>
        {NO_VALUE}
      </span>
    );
  }
  return <span className={`live${className ? ` ${className}` : ''}`}>{value}</span>;
}

/** A monospaced inline literal — a path, a command, an identifier. Breaks anywhere. */
export function Mono({ children }: { children: ReactNode }) {
  return <code className="lit">{children}</code>;
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/** Every table scrolls inside its own box; the page never scrolls sideways. */
export function ScrollX({ children }: { children: ReactNode }) {
  return <div className="tw">{children}</div>;
}

export function Table({ children }: { children: ReactNode }) {
  return <table style={{ minWidth: '36rem' }}>{children}</table>;
}

export function Th({ children, numeric = false, className }: { children: ReactNode; numeric?: boolean; className?: string }) {
  return <th className={`${numeric ? 'num' : ''}${className ? ` ${className}` : ''}`}>{children}</th>;
}

export function Td({
  children,
  numeric = false,
  className,
}: {
  /** Optional: an empty cell keeps a row aligned when it carries a single message. */
  children?: ReactNode;
  numeric?: boolean;
  className?: string;
}) {
  return <td className={`${numeric ? 'num' : ''}${className ? ` ${className}` : ''}`}>{children}</td>;
}

// ---------------------------------------------------------------------------
// Small pieces
// ---------------------------------------------------------------------------

/** A labelled fact in a definition grid. */
export function Field({ label, children, note }: { label: string; children: ReactNode; note?: ReactNode }) {
  return (
    <div className="field">
      <span className="label">{label}</span>
      <div className="v">{children}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

/** A status mark: ● good · ◐ replay or deferred · ✕ broken · ○ idle — glyph, not colour alone. */
export function Lamp({ state }: { state: 'good' | 'replay' | 'bad' | 'idle' }) {
  const tone = state === 'good' ? 'ok' : state === 'replay' ? 'warn' : state === 'bad' ? 'err' : 'no';
  return <span aria-hidden className={`status ${tone}`} />;
}
