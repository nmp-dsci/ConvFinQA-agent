import type { ReactNode } from 'react';
import { useIsDemo } from '../modeStore';

/** Percentage with one decimal, and tabular figures so columns line up. */
export function Pct({ value, className = '' }: { value: number | null | undefined; className?: string }) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return <span className="text-textMuted">—</span>;
  }
  return <span className={`tabular-nums ${className}`}>{(value * 100).toFixed(1)}%</span>;
}

/** A signed delta, coloured by direction. Zero reads as neutral, not as a win. */
export function Delta({ value }: { value: number }) {
  const tone = value > 0 ? 'text-accent2' : value < 0 ? 'text-danger' : 'text-textMuted';
  const sign = value > 0 ? '+' : '';
  return (
    <span className={`tabular-nums ${tone}`}>
      {sign}
      {(value * 100).toFixed(2)}%
    </span>
  );
}

export function Badge({
  children,
  tone = 'neutral',
  title,
}: {
  children: ReactNode;
  tone?: 'neutral' | 'good' | 'bad' | 'warn' | 'accent';
  title?: string;
}) {
  const tones: Record<string, string> = {
    neutral: 'bg-panel2 text-textMuted border-white/10',
    good: 'bg-accent/25 text-accent2 border-accent2/40',
    bad: 'bg-danger/15 text-danger border-danger/40',
    warn: 'bg-amber-500/15 text-amber-300 border-amber-400/40',
    accent: 'bg-accent2/20 text-accent2 border-accent2/50',
  };
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[11px] font-medium whitespace-nowrap ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function CorrectMark({ correct }: { correct: boolean | null | undefined }) {
  if (correct === null || correct === undefined) return <span className="text-textMuted">—</span>;
  return correct ? (
    <span className="text-accent2" title="matches gold">
      ✓
    </span>
  ) : (
    <span className="text-danger" title="does not match gold">
      ✗
    </span>
  );
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`bg-panel border border-white/5 rounded-lg overflow-hidden ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 px-4 py-3 border-b border-white/5">
          <div className="min-w-0">
            {title && <h2 className="font-medium text-sm">{title}</h2>}
            {subtitle && <div className="text-xs text-textMuted mt-0.5">{subtitle}</div>}
          </div>
          {actions && <div className="shrink-0 flex items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-textMuted p-4">
      <span className="inline-block size-3 rounded-full border-2 border-textMuted/40 border-t-accent2 animate-spin" />
      {label}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: ReactNode }) {
  return (
    <div className="p-8 text-center">
      <div className="text-sm text-textMain">{title}</div>
      {hint && <div className="text-xs text-textMuted mt-1 max-w-prose mx-auto">{hint}</div>}
    </div>
  );
}

export function ErrorNote({ error }: { error: string }) {
  return (
    <div className="m-4 p-3 rounded border border-danger/40 bg-danger/10 text-sm text-danger">
      {error}
    </div>
  );
}

/**
 * Wraps controls that write. In demo mode it renders a real `<fieldset disabled>`
 * — not a click handler that silently does nothing — so the control is
 * genuinely inert for keyboard and screen-reader users too, and *visible*, which
 * is the point: the demo shows the whole product, with the writes turned off.
 */
export function DemoGate({
  children,
  reason = 'Read-only in the demo',
  className = '',
}: {
  children: ReactNode;
  reason?: string;
  className?: string;
}) {
  const isDemo = useIsDemo();
  if (!isDemo) return <>{children}</>;
  return (
    <div className={`relative ${className}`} title={reason}>
      <fieldset disabled className="contents opacity-50 pointer-events-none">
        {children}
      </fieldset>
    </div>
  );
}

/** A short label explaining what the demo replaces, shown beside gated controls. */
export function DemoChip({ children = 'demo replay' }: { children?: ReactNode }) {
  return <Badge tone="warn">{children}</Badge>;
}

export function Mono({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <span className={`font-mono text-xs ${className}`}>{children}</span>;
}

/** Horizontal-scroll container. Wide tables must scroll, never the page body. */
export function ScrollX({ children }: { children: ReactNode }) {
  return <div className="overflow-x-auto">{children}</div>;
}

export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

export function formatTime(value: string | number | null | undefined): string {
  if (!value) return '—';
  const date = typeof value === 'number' ? new Date(value) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}
