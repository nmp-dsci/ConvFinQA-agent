import {
  Activity,
  BarChart3,
  FlaskConical,
  LayoutDashboard,
  Microscope,
  MessageSquare,
  Server,
  Sparkles,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { NavLink, useLocation } from 'react-router-dom';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

interface RailItem {
  to: string;
  label: string;
  hint: string;
  icon: LucideIcon;
  /** Extra paths that should light this item up (child routes, aliases). */
  matches?: string[];
}

const ITEMS: RailItem[] = [
  {
    to: '/welcome',
    label: 'Overview',
    hint: 'What this system is and how it is measured',
    icon: Sparkles,
  },
  {
    to: '/chat',
    label: 'Chat',
    hint: 'Ask a question about a filing',
    icon: MessageSquare,
    matches: ['/'],
  },
  {
    to: '/admin',
    label: 'Admin',
    hint: 'Operator overview',
    icon: LayoutDashboard,
  },
  {
    to: '/admin/evaluations',
    label: 'Evaluations',
    hint: 'Accuracy slices per version, gold beside every answer',
    icon: BarChart3,
  },
  {
    to: '/admin/experiments',
    label: 'Experiments',
    hint: 'Runs, registry, promotion history',
    icon: FlaskConical,
  },
  {
    to: '/admin/traces',
    label: 'Traces',
    hint: 'Stage-by-stage record of every turn',
    icon: Activity,
  },
  {
    to: '/admin/research',
    label: 'Research',
    hint: 'Automated prompt-research rounds',
    icon: Microscope,
  },
  {
    to: '/admin/system',
    label: 'System',
    hint: 'Deployment, limits, demo gate',
    icon: Server,
  },
];

/**
 * The 52px icon rail.
 *
 * It sits at `--ground` with the top bar, so the lit surfaces (the chat
 * thread, panels) read as raised out of it rather than pasted onto it.
 * Active state is amber-soft fill plus a solid amber edge — colour *and*
 * shape, so it survives a colour-blind reader and a greyscale screenshot.
 */
export function NavRail() {
  const { pathname } = useLocation();

  function isActive(item: RailItem): boolean {
    if (item.matches?.includes(pathname)) return true;
    if (item.to === '/admin') return pathname === '/admin';
    return pathname === item.to || pathname.startsWith(`${item.to}/`);
  }

  return (
    <nav
      aria-label="Primary"
      data-testid="nav-rail"
      className="w-[var(--rail-w)] shrink-0 bg-ground border-r border-line flex flex-col items-center gap-1 py-2"
    >
      {ITEMS.map((item) => {
        const Icon = item.icon;
        const active = isActive(item);
        return (
          <Tooltip key={item.to}>
            <TooltipTrigger asChild>
              <NavLink
                to={item.to}
                aria-label={item.label}
                aria-current={active ? 'page' : undefined}
                data-testid={`nav-${item.label.toLowerCase()}`}
                className={cn(
                  'relative flex size-9 items-center justify-center rounded-md transition-colors',
                  'text-faint hover:text-text hover:bg-panel-2',
                  active && 'bg-amber-soft text-amber hover:bg-amber-soft hover:text-amber',
                )}
              >
                {active && (
                  <span
                    aria-hidden
                    className="absolute left-[-8px] top-1.5 bottom-1.5 w-[2px] rounded-full bg-amber"
                  />
                )}
                <Icon className="size-4" aria-hidden />
              </NavLink>
            </TooltipTrigger>
            <TooltipContent side="right">
              <div className="font-medium">{item.label}</div>
              <div className="text-[11px] opacity-80">{item.hint}</div>
            </TooltipContent>
          </Tooltip>
        );
      })}
    </nav>
  );
}
