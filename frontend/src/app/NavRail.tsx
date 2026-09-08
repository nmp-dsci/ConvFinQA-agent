import { NavLink, useLocation } from 'react-router-dom';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { NAV, isActive } from './nav';
import type { NavItem } from './nav';

/**
 * The rail, in two widths.
 *
 * At `lg` and above it is 176px with the label beside every icon and a
 * mono-caps header over each of the four groups: a reader arriving cold can
 * see the whole story in one column without hovering anything. Between `md`
 * and `lg` it is the 52px icon rail with the label in a tooltip. Below `md` it
 * is not rendered at all — `TabBar` takes over at the bottom of the screen.
 *
 * Active state is amber-soft fill plus a solid amber edge: colour *and* shape,
 * so it survives a colour-blind reader and a greyscale screenshot. The focus
 * ring is the global one from tokens.css.
 */
export function NavRail() {
  const { pathname } = useLocation();

  return (
    <nav
      aria-label="Primary"
      data-testid="nav-rail"
      className={cn(
        'hidden shrink-0 flex-col gap-3 overflow-y-auto border-r border-line bg-ground py-2',
        'md:flex md:w-[var(--rail-w)] md:items-center',
        'lg:w-[var(--rail-w-wide)] lg:items-stretch lg:px-2',
      )}
    >
      {NAV.map((group) => (
        <div key={group.key} className="flex flex-col gap-0.5 lg:gap-px">
          <div className="mono-caps hidden px-2 pt-1 pb-1 lg:block" aria-hidden>
            {group.label}
          </div>
          {group.items.map((item) => (
            <RailItem key={item.to} item={item} active={isActive(item, pathname)} />
          ))}
        </div>
      ))}
    </nav>
  );
}

function RailItem({ item, active }: { item: NavItem; active: boolean }) {
  const Icon = item.icon;
  const link = (
    <NavLink
      to={item.to}
      aria-label={item.label}
      aria-current={active ? 'page' : undefined}
      data-testid={`nav-${item.label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/-$/, '')}`}
      className={cn(
        'relative flex items-center rounded-md transition-colors',
        'size-9 justify-center lg:h-8 lg:w-auto lg:justify-start lg:gap-2.5 lg:px-2',
        'text-faint hover:bg-panel-2 hover:text-text',
        active && 'bg-amber-soft text-amber hover:bg-amber-soft hover:text-amber',
      )}
    >
      {active && (
        <span
          aria-hidden
          className="absolute top-1.5 bottom-1.5 left-[-8px] w-[2px] rounded-full bg-amber lg:left-[-9px]"
        />
      )}
      <Icon className="size-4 shrink-0" aria-hidden />
      <span className="type-small hidden truncate lg:inline">{item.label}</span>
    </NavLink>
  );

  return (
    <>
      {/* Icon-only widths get the label as a tooltip; the wide rail shows it. */}
      <div className="lg:hidden">
        <Tooltip>
          <TooltipTrigger asChild>{link}</TooltipTrigger>
          <TooltipContent side="right">
            <div className="font-medium">{item.label}</div>
            <div className="type-meta opacity-80">{item.hint}</div>
          </TooltipContent>
        </Tooltip>
      </div>
      <div className="hidden lg:block">{link}</div>
    </>
  );
}
