import { Menu } from 'lucide-react';
import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { cn } from '@/lib/utils';
import { NAV, NAV_ITEMS, isActive } from './nav';

/** The four items that get a tab of their own; everything else is in the sheet. */
const DIRECT = ['/', '/chat', '/admin/system', '/admin/runtimes'];

/**
 * Phone navigation: a bottom tab bar with four direct tabs and a menu that
 * opens the full grouped list as a sheet.
 *
 * Rendered only below `md`. The rail is `hidden` there rather than shrunk,
 * because a 52px column on a 390px phone is an eighth of the screen spent on
 * icons nobody can read, and a tab bar is what a phone user's thumb expects.
 */
export function TabBar() {
  const { pathname } = useLocation();
  const [open, setOpen] = useState(false);
  const direct = DIRECT.map((to) => NAV_ITEMS.find((item) => item.to === to)).filter(
    (item): item is NonNullable<typeof item> => Boolean(item),
  );
  const menuActive = !direct.some((item) => isActive(item, pathname));

  return (
    <>
      <nav
        aria-label="Primary"
        data-testid="tab-bar"
        className="flex shrink-0 items-stretch justify-around border-t border-line bg-ground md:hidden"
      >
        {direct.map((item) => {
          const Icon = item.icon;
          const active = isActive(item, pathname);
          return (
            <NavLink
              key={item.to}
              to={item.to}
              aria-current={active ? 'page' : undefined}
              className={cn(
                'flex min-w-0 flex-1 flex-col items-center gap-0.5 px-1 pt-2 pb-2.5',
                'mono-caps transition-colors',
                active ? 'text-amber' : 'text-faint hover:text-text',
              )}
            >
              <Icon className="size-4" aria-hidden />
              <span className="truncate">{item.label}</span>
            </NavLink>
          );
        })}
        <button
          type="button"
          onClick={() => setOpen(true)}
          aria-label="All pages"
          aria-expanded={open}
          className={cn(
            'flex min-w-0 flex-1 flex-col items-center gap-0.5 px-1 pt-2 pb-2.5 mono-caps transition-colors',
            menuActive ? 'text-amber' : 'text-faint hover:text-text',
          )}
        >
          <Menu className="size-4" aria-hidden />
          <span>Menu</span>
        </button>
      </nav>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="bottom" className="max-h-[80vh] overflow-y-auto bg-ground">
          <SheetHeader className="p-0 pb-3">
            <SheetTitle className="type-body font-medium text-text">All pages</SheetTitle>
            <SheetDescription className="type-meta">
              In the order the story is told: the product, how it was built, the evidence, the
              operations behind it.
            </SheetDescription>
          </SheetHeader>
          <div className="grid gap-4">
            {NAV.map((group) => (
              <div key={group.key}>
                <div className="mono-caps mb-1">{group.label}</div>
                <ul className="grid gap-0.5">
                  {group.items.map((item) => {
                    const Icon = item.icon;
                    const active = isActive(item, pathname);
                    return (
                      <li key={item.to}>
                        <NavLink
                          to={item.to}
                          onClick={() => setOpen(false)}
                          aria-current={active ? 'page' : undefined}
                          className={cn(
                            'flex items-start gap-2.5 rounded-md px-2 py-1.5 transition-colors',
                            active ? 'bg-amber-soft text-amber' : 'text-text hover:bg-panel-2',
                          )}
                        >
                          <Icon className="mt-0.5 size-4 shrink-0" aria-hidden />
                          <span className="min-w-0">
                            <span className="type-body block font-medium">{item.label}</span>
                            <span className="type-meta block">{item.hint}</span>
                          </span>
                        </NavLink>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
}
