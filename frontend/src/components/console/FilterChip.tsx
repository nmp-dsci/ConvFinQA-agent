import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

/**
 * A toggle chip in a filter row. Active is amber on amber-soft with an amber
 * edge — the same active grammar as the rail — and the state is also on
 * `aria-pressed`, so it is not colour-only.
 */
export function FilterChip({
  active,
  onClick,
  children,
  className,
  testId,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
  className?: string;
  testId?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      data-testid={testId}
      className={cn(
        'mono-caps rounded-[4px] border px-2.5 py-1 transition-colors',
        active
          ? 'border-amber-line bg-amber-soft text-amber'
          : 'border-line text-muted hover:border-line-2 hover:text-text',
        className,
      )}
    >
      {children}
    </button>
  );
}
