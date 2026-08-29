import type { ReactNode } from 'react';

/**
 * A route that exists but is not built yet.
 *
 * Phase 0's job is that every path in the route table resolves — a 404 during
 * the redesign would be indistinguishable from a routing bug. Each placeholder
 * says which phase owns it and what will land there, so an operator who clicks
 * through gets an answer rather than an empty frame.
 */
export function Placeholder({
  title,
  phase,
  children,
  testId,
}: {
  title: string;
  phase: number;
  children: ReactNode;
  testId: string;
}) {
  return (
    <div className="h-full overflow-y-auto p-6" data-testid={testId}>
      <div className="mx-auto max-w-2xl">
        <div className="mono-caps mb-2">phase {phase}</div>
        <h1 className="mb-3 text-xl font-medium tracking-tight text-text">{title}</h1>
        <div className="rounded-lg border border-line bg-panel p-5 text-sm leading-relaxed text-muted">
          {children}
          <p className="mt-3 text-xs text-faint">
            Phase {phase} builds this. The route, the shell and the data client are in place; only
            the page is pending.
          </p>
        </div>
      </div>
    </div>
  );
}
