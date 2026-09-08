import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

export type LampTone = 'good' | 'amber' | 'bad' | 'info' | 'idle';

const DOT_TONE: Record<LampTone, string> = {
  good: 'border-good bg-good shadow-[0_0_6px_var(--good-glow)]',
  amber: 'border-amber',
  bad: 'border-bad bg-bad',
  info: 'border-info bg-info',
  idle: 'border-line-2',
};

const TEXT_TONE: Record<LampTone, string> = {
  good: 'text-good',
  amber: 'text-amber',
  bad: 'text-bad',
  info: 'text-info',
  idle: 'text-faint',
};

export interface LampProps {
  label: string;
  value: string;
  tone: LampTone;
  /**
   * Dashed means replayed or unverified, solid means measured. The shape, not
   * the colour, is what a colour-blind reader and a greyscale screenshot are
   * left with — so no lamp may distinguish itself by colour alone.
   */
  dashed?: boolean;
  /** A plain-text title when there is no tooltip. */
  title?: string;
  /** A rich tooltip; wins over `title`. */
  tooltip?: ReactNode;
  to?: string;
}

/**
 * One lamp — the same component on the landing strip and every admin page.
 * It used to exist twice with identical tone tables; this is the one copy.
 */
export function Lamp({ label, value, tone, dashed = false, title, tooltip, to }: LampProps) {
  const body = (
    <span
      data-testid={`lamp-${label}`}
      data-tone={tone}
      data-shape={dashed ? 'dashed' : 'solid'}
      title={tooltip ? undefined : title}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-line bg-panel py-1 pr-2.5 pl-2',
        to && 'transition-colors hover:border-amber-line hover:bg-panel-2',
      )}
    >
      <span
        aria-hidden
        className={cn(
          'size-2 shrink-0 rounded-full border',
          DOT_TONE[tone],
          dashed && 'border-dashed bg-transparent shadow-none',
        )}
      />
      <span className="mono-caps text-faint">{label}</span>
      <span className={cn('type-num type-meta', TEXT_TONE[tone])}>{value}</span>
    </span>
  );

  const linked = to ? (
    <Link to={to} className="cursor-pointer">
      {body}
    </Link>
  ) : tooltip ? (
    <span className="cursor-default">{body}</span>
  ) : (
    body
  );

  if (!tooltip) return linked;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{linked}</TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-xs">
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

export function LampRow({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap items-center gap-1.5">{children}</div>;
}
