import { Suspense, useEffect } from 'react';
import { Link, Outlet } from 'react-router-dom';
import { TooltipProvider } from '@/components/ui/tooltip';
import { ThemeToggle } from '../components/ThemeToggle';
import { useMode } from '../modeStore';
import { useStore } from '../store';
import { ModeLamp } from './ModeLamp';
import { NavRail } from './NavRail';

function BrandMark() {
  return (
    <Link
      to="/"
      className="flex items-center gap-2 text-text hover:text-amber transition-colors"
      title="ConvFinQA console"
    >
      <span
        aria-hidden
        className="grid size-6 place-items-center rounded-[4px] border border-amber-line bg-amber-soft font-mono text-[10px] font-semibold text-amber"
      >
        CF
      </span>
      <span className="text-[13px] font-medium tracking-tight">ConvFinQA</span>
    </Link>
  );
}

function TopBar() {
  const champion = useMode((s) => s.health?.champion);

  return (
    <header
      data-testid="shell-topbar"
      className="flex h-11 shrink-0 items-center justify-between gap-4 border-b border-line bg-ground px-3"
    >
      <BrandMark />
      <div className="flex items-center gap-3">
        {champion && (
          <span className="hidden items-baseline gap-1.5 lg:inline-flex">
            <span className="mono-caps">champion</span>
            <span className="font-mono text-[11px] text-muted">{champion}</span>
          </span>
        )}
        <ModeLamp />
        <ThemeToggle />
      </div>
    </header>
  );
}

function RouteFallback() {
  return (
    <div className="flex h-full items-center justify-center">
      <span className="mono-caps animate-pulse">loading…</span>
    </div>
  );
}

/**
 * The app shell: nav rail on the left, top bar across, routed content below.
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
        </div>
      </div>
    </TooltipProvider>
  );
}
