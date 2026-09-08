import { Suspense, useEffect } from 'react';
import { Link, Outlet } from 'react-router-dom';
import { TooltipProvider } from '@/components/ui/tooltip';
import { ThemeToggle } from '../components/ThemeToggle';
import { useMode } from '../modeStore';
import { useStore } from '../store';
import { ModeLamp } from './ModeLamp';
import { NavRail } from './NavRail';
import { TabBar } from './TabBar';

function BrandMark() {
  return (
    <Link
      to="/"
      className="flex items-center gap-2 text-text transition-colors hover:text-amber"
      title="ConvFinQA console"
    >
      <span
        aria-hidden
        className="grid size-6 place-items-center rounded-[4px] border border-amber-line bg-amber-soft type-num type-meta font-semibold text-amber"
      >
        CF
      </span>
      <span className="type-body font-medium tracking-tight">ConvFinQA</span>
    </Link>
  );
}

function TopBar() {
  const health = useMode((s) => s.health);
  const serving = health?.serving_champion ?? health?.champion;

  return (
    <header
      data-testid="shell-topbar"
      className="flex h-11 shrink-0 items-center justify-between gap-4 border-b border-line bg-ground px-3"
    >
      <BrandMark />
      <div className="flex items-center gap-3">
        {serving && (
          <span className="hidden items-baseline gap-1.5 lg:inline-flex">
            <span className="mono-caps">serving</span>
            <span className="type-num type-meta text-muted">
              {serving}
              {health?.runtime === 'agent_sdk' ? ' · agent sdk' : ''}
            </span>
          </span>
        )}
        <ModeLamp />
        <ThemeToggle />
      </div>
    </header>
  );
}

/** A skeleton in the page-header shape, so a lazy route does not flash blank. */
function RouteFallback() {
  return (
    <div className="mx-auto max-w-[1560px] px-3 py-4 sm:px-5" aria-label="loading" aria-busy>
      <div className="h-2.5 w-28 animate-pulse rounded bg-panel-2" />
      <div className="mt-3 h-6 w-56 animate-pulse rounded bg-panel-2" />
      <div className="mt-3 h-4 w-[min(70ch,100%)] animate-pulse rounded bg-panel-2" />
      <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="h-24 animate-pulse rounded-md bg-panel" />
        ))}
      </div>
    </div>
  );
}

/**
 * The app shell: rail on the left, top bar across, routed content below, and
 * on a phone a tab bar along the bottom instead of the rail.
 *
 * The rail and the top bar sit at `--ground`; whatever the route renders is
 * responsible for its own lit surface. That is the elevation rule made
 * structural rather than a convention each page has to remember.
 */
export function Shell() {
  const loadReports = useStore((s) => s.loadReports);
  const loadMode = useMode((s) => s.load);

  // Both are read once for the whole session. They live here rather than in a
  // route so switching tabs never re-asks the server what deployment this is.
  useEffect(() => {
    void loadMode();
    void loadReports();
  }, [loadMode, loadReports]);

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex h-full min-h-0 bg-ground text-text">
        <NavRail />
        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar />
          <main className="min-h-0 flex-1 overflow-hidden">
            <Suspense fallback={<RouteFallback />}>
              <Outlet />
            </Suspense>
          </main>
          <TabBar />
        </div>
      </div>
    </TooltipProvider>
  );
}
