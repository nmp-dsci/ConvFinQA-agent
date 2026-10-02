import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

export type LampTone = 'good' | 'amber' | 'bad' | 'info' | 'idle';

export interface LampProps {
  label: string;
  value: string;
  /** good = passed (--ok) · bad = failed (--red) · amber = partial or replayed (--amber) · info/idle = neutral */
  tone: LampTone;
  /** Dashed means replayed or unverified, solid means measured: shape, not colour alone. */
  dashed?: boolean;
  title?: string;
  /** A rich tooltip; wins over `title`. */
  tooltip?: ReactNode;
  to?: string;
}

/** One labelled status — the same component on the Overview strip and every evidence page. */
export function Lamp({ label, value, tone, dashed = false, title, tooltip, to }: LampProps) {
  const props = {
    'data-testid': `lamp-${label}`,
    'data-tone': tone,
    'data-shape': dashed ? 'dashed' : 'solid',
    className: `lamp${dashed ? ' dashed' : ''}`,
    title: tooltip ? undefined : title,
  };
  const inner = (
    <>
      <span className="dot" aria-hidden />
      <span className="k">{label}</span>
      <span className="v">{value}</span>
    </>
  );
  const body = to ? (
    <Link to={to} {...props}>
      {inner}
    </Link>
  ) : (
    <span {...props}>{inner}</span>
  );
  if (!tooltip) return body;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{body}</TooltipTrigger>
      <TooltipContent side="bottom">{tooltip}</TooltipContent>
    </Tooltip>
  );
}

export function LampRow({ children }: { children: ReactNode }) {
  return <div className="lamps">{children}</div>;
}
