// nmp-ui-kit 1.0.0 · vendored from nmp-central-ai/ui/kit — do not edit here; change the kit and run `make ui-sync`.
/**
 * The pieces every page shares, so DESIGN.md's number and status rules live in one place.
 * Carved from DataAgentBench's and tau2-loop's `lib/ui.tsx`.
 */
import { type ReactNode, useState } from 'react';
import { Link } from 'react-router-dom';
import { fmtPct } from './format';

/** The page opening, as a slide: mono eyebrow → serif h1 claim (one <em>) → serif lead. */
export function PageHead({ eyebrow, title, lead, children }: { eyebrow?: ReactNode; title: ReactNode; lead?: ReactNode; children?: ReactNode }) {
  return (
    <header className="page-head" data-testid="page-header">
      {eyebrow && <p className="label">{eyebrow}</p>}
      <h1 data-testid="page-title">{title}</h1>
      {lead && (
        <p className="lead" data-testid="page-verdict">
          {lead}
        </p>
      )}
      {children}
    </header>
  );
}

/** A rate with its denominator in the same cell: never a bare percentage. */
export function Rate({ passed, n, digits = 0, unit = 'questions', warnBelow = 0.1 }: { passed: number | null | undefined; n: number | null | undefined; digits?: number; unit?: string; warnBelow?: number }) {
  if (passed == null || !n) return <span className="muted">not scored</span>;
  const r = passed / n;
  return (
    <span className="ratecell" title={`${passed} of ${n} ${unit}`}>
      <span className="track">
        <i className={r < warnBelow ? 'warn' : ''} style={{ width: `${Math.max(1, r * 100)}%` }} />
      </span>
      <span className="mono">
        {fmtPct(r, digits)} · {passed}/{n}
      </span>
    </span>
  );
}

/** A stat tile: the number, and underneath it what it measures and against what. */
export function Kpi({ n, b, tone, testId }: { n: ReactNode; b: ReactNode; tone?: 'ok' | 'warn' | 'acc'; testId?: string }) {
  return (
    <div className="kpi" data-testid={testId}>
      <div className={`n ${tone ?? ''}`}>{n}</div>
      <div className="b">{b}</div>
    </div>
  );
}

export function Loading({ error, what }: { error?: string | null; what?: string }) {
  return <p className="empty">{error ? `Could not load${what ? ` ${what}` : ''}: ${error}` : 'Loading…'}</p>;
}

/** An absence, said out loud: what is missing and why. */
export function Empty({ children, testId }: { children: ReactNode; testId?: string }) {
  return (
    <p className="empty" data-testid={testId}>
      {children}
    </p>
  );
}

export type StatusTone = 'ok' | 'no' | 'warn' | 'err' | 'acc';
/** A status word with its glyph (● ○ ◐ ✕): status is never colour alone. */
export function Status({ tone, children, title }: { tone: StatusTone; children: ReactNode; title?: string }) {
  return (
    <span className={`status ${tone}`} title={title}>
      {children}
    </span>
  );
}

/** Every item of one kind as a lozenge, one click into it. `current` is unset on an index page. */
export function ChipBar({ items, current, label }: { items: { key: string; to: string; label: ReactNode; n?: number | null }[]; current?: string; label: string }) {
  return (
    <nav className="chips bar" aria-label={label}>
      {items.map((it) => (
        <Link key={it.key} to={it.to} className={`chip nav ${it.key === current ? 'on' : ''}`} aria-current={it.key === current ? 'page' : undefined}>
          {it.label}
          {it.n != null && <span className="n">{it.n}</span>}
        </Link>
      ))}
    </nav>
  );
}

/** Long text clipped at `at` characters, with a toggle that opens the rest in place. */
export function Clip({ text, at }: { text: string; at: number }) {
  const [open, setOpen] = useState(false);
  if (text.length <= at) return <>{text}</>;
  return (
    <>
      {open ? text : `${text.slice(0, at).trimEnd()}…`}
      <button type="button" className="more" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        {open ? 'less' : 'more'}
      </button>
    </>
  );
}

/** A figure, introduced before it is read: mono title, the frame, then the takeaway and its source. */
export function Figure({ title, caption, source, children, testId }: { title: ReactNode; caption?: ReactNode; source?: ReactNode; children: ReactNode; testId?: string }) {
  return (
    <figure data-testid={testId}>
      <p className="label fig-title">{title}</p>
      {children}
      {(caption || source) && (
        <figcaption>
          {caption}
          {source && <> Source: {source}.</>}
        </figcaption>
      )}
    </figure>
  );
}

/** A write control in a read-only deployment: a real disabled fieldset, and the reason beside it. */
export function WriteGate({ readOnly, reason, children }: { readOnly: boolean; reason?: string; children: ReactNode }) {
  return (
    <fieldset disabled={readOnly} aria-disabled={readOnly}>
      {children}
      {readOnly && <p className="small muted">{reason ?? 'Read-only here: this deployment serves committed records and never writes.'}</p>}
    </fieldset>
  );
}
