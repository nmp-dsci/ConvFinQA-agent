// nmp-ui-kit 0.1.0 · vendored from nmp-central-ai/ui/kit — do not edit here; change the kit and run `make ui-sync`.
/**
 * The house shell: a sticky header (brand, the tab pills, the mode pill), the page, and a
 * provenance footer. Every project renders the same markup; only the props differ.
 *
 * Tabs are data, passed in: eight slots in story order (what the benchmark is → how it is judged →
 * who else has tried → what we ran → how we improve → what the agent is → what we know by hand).
 * A route whose `handle` carries `{ wide: true }` renders full-width and full-height (a chat, a
 * playground) with no footer.
 */
import { type ReactNode, Suspense } from 'react';
import { NavLink, Outlet, useMatches } from 'react-router-dom';

export type Tab = {
  /** where the tab goes */
  to: string;
  /** the visible label; never an icon alone */
  label: string;
  /** one line said on hover: what the page answers */
  hint?: string;
  /** other path prefixes that should light this tab */
  matches?: string[];
};

export type Mode = { label: string; tone?: 'ok' | 'idle'; title?: string };

export type ShellProps = {
  /** e.g. <>convfinqa<b>-agent</b></> — the second word takes the accent */
  brand: ReactNode;
  /** 22px mark beside the brand; the favicon, usually */
  logo?: string;
  tabs: Tab[];
  mode?: Mode | null;
  footer?: ReactNode;
  /** passed to <Outlet context>, e.g. /healthz */
  context?: unknown;
  /** rendered after <main> and before the footer (a toaster, a palette) */
  children?: ReactNode;
  /** shown while a lazy route loads; the header never blanks */
  fallback?: ReactNode;
};

function isWide(matches: ReturnType<typeof useMatches>): boolean {
  return matches.some((m) => (m.handle as { wide?: boolean } | undefined)?.wide === true);
}

export function Shell({ brand, logo = '/favicon.svg', tabs, mode, footer, context, children, fallback }: ShellProps) {
  const matches = useMatches();
  const wide = isWide(matches);
  const path = matches.length ? matches[matches.length - 1].pathname : '/';
  return (
    <div className={`shell${wide ? ' app' : ''}`}>
      <header className="top">
        <div className="in">
          <NavLink to="/" className="brand">
            {logo && <img src={logo} alt="" width="22" height="22" />}
            <span>{brand}</span>
          </NavLink>
          <nav aria-label="Pages">
            {tabs.map((t) => (
              <NavLink
                key={t.to}
                to={t.to}
                end={t.to === '/'}
                title={t.hint}
                className={({ isActive }) =>
                  isActive || (t.matches ?? []).some((m) => path === m || path.startsWith(`${m}/`)) ? 'on' : ''
                }
              >
                {t.label}
              </NavLink>
            ))}
          </nav>
          {mode !== undefined && (
            <span className={`mode ${mode?.tone === 'ok' ? 'ok' : ''}`} title={mode?.title} data-testid="mode-pill">
              {mode ? mode.label : '…'}
            </span>
          )}
        </div>
      </header>
      <main className={wide ? 'app' : undefined}>
        <Suspense fallback={fallback ?? <p className="empty">Loading…</p>}>
          <Outlet context={context} />
        </Suspense>
      </main>
      {children}
      {!wide && footer && <footer className="site">{footer}</footer>}
    </div>
  );
}
