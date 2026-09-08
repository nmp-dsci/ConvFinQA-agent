import {
  Activity,
  BarChart3,
  BookOpenCheck,
  ClipboardCheck,
  Cpu,
  FlaskConical,
  LayoutDashboard,
  MessageSquare,
  Microscope,
  Server,
  Sparkles,
  TrendingUp,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

/**
 * The navigation, as data — and the story order, as navigation.
 *
 * Eleven routes in four labelled groups, in the order the write-up tells it:
 * the product, how it was built, the evidence, and the operations behind it.
 * Every surface that walks the story — the rail, the phone tab bar, the page
 * headers' eyebrows and the "next" footers — reads this one list, so the order
 * cannot be told three different ways.
 */

export interface NavItem {
  to: string;
  label: string;
  /** One line under the label in tooltips and in the next-step footer. */
  hint: string;
  icon: LucideIcon;
  /** Extra paths that should light this item up (child routes, aliases). */
  matches?: string[];
}

export interface NavGroup {
  key: 'product' | 'built' | 'evidence' | 'operations';
  label: string;
  items: NavItem[];
}

export const NAV: NavGroup[] = [
  {
    key: 'product',
    label: 'Product',
    items: [
      {
        to: '/',
        label: 'Overview',
        hint: 'What this system does, the result, and the nine-dimension score',
        icon: Sparkles,
      },
      {
        to: '/chat',
        label: 'Chat',
        hint: 'Ask a filing a chain of questions and watch each stage resolve',
        icon: MessageSquare,
      },
    ],
  },
  {
    key: 'built',
    label: 'Built',
    items: [
      {
        to: '/admin/system',
        label: 'Architecture',
        hint: 'Four agents, one choke point, one contract — and what is still broken',
        icon: Server,
        matches: ['/debrief'],
      },
      {
        to: '/admin/readiness',
        label: 'Readiness',
        hint: 'The production gen-AI rubric, R1–R9, scored against this system with its proof',
        icon: ClipboardCheck,
      },
    ],
  },
  {
    key: 'evidence',
    label: 'Evidence',
    items: [
      {
        to: '/admin',
        label: 'Scoreboard',
        hint: 'What serves now and what production has been doing',
        icon: LayoutDashboard,
      },
      {
        to: '/admin/evaluations',
        label: 'Evaluations',
        hint: 'Accuracy per version and slice, gold beside every answer',
        icon: BarChart3,
      },
      {
        to: '/admin/dataset',
        label: 'Dataset',
        hint: 'Every split, every question, its gold answer and gold program',
        icon: BookOpenCheck,
      },
      {
        to: '/admin/campaigns',
        label: 'Campaigns',
        hint: 'The optimisation loop: one prompt per experiment, and what moved',
        icon: TrendingUp,
      },
      {
        to: '/admin/runtimes',
        label: 'Runtimes',
        hint: 'One Claude session against four agents on one sealed split — and the judge tried after',
        icon: Cpu,
      },
      {
        to: '/admin/experiments',
        label: 'Experiments',
        hint: 'Runs, the registry, and the append-only promotion history',
        icon: FlaskConical,
      },
    ],
  },
  {
    key: 'operations',
    label: 'Operations',
    items: [
      {
        to: '/admin/traces',
        label: 'Traces',
        hint: 'Every served turn, by source, with the four stage captures behind it',
        icon: Activity,
      },
      {
        to: '/admin/research',
        label: 'Research (s7)',
        hint: 'The retired per-case harness that wrote v3_1, kept because its stores are committed',
        icon: Microscope,
      },
    ],
  },
];

/** Every item, in story order. */
export const NAV_ITEMS: NavItem[] = NAV.flatMap((g) => g.items);

/** True when `pathname` is this item or one of its children. */
export function isActive(item: NavItem, pathname: string): boolean {
  if (item.matches?.includes(pathname)) return true;
  // `/` is a prefix of every path and `/admin` of every admin page, so both
  // can only ever match exactly.
  if (item.to === '/' || item.to === '/admin') return pathname === item.to;
  return pathname === item.to || pathname.startsWith(`${item.to}/`);
}

export function activeItem(pathname: string): NavItem | undefined {
  return NAV_ITEMS.find((item) => isActive(item, pathname));
}

export function groupOf(item: NavItem): NavGroup | undefined {
  return NAV.find((g) => g.items.includes(item));
}

/** `evidence · 03 of 06` — the eyebrow every page header prints. */
export function positionOf(pathname: string): string | null {
  const item = activeItem(pathname);
  const group = item && groupOf(item);
  if (!item || !group) return null;
  const n = group.items.indexOf(item) + 1;
  return `${group.label} · ${String(n).padStart(2, '0')} of ${String(group.items.length).padStart(2, '0')}`;
}

/** The item after this one in story order, or null at the end. */
export function nextItem(pathname: string): NavItem | null {
  const item = activeItem(pathname);
  if (!item) return null;
  const i = NAV_ITEMS.indexOf(item);
  return NAV_ITEMS[i + 1] ?? null;
}

export function prevItem(pathname: string): NavItem | null {
  const item = activeItem(pathname);
  if (!item) return null;
  const i = NAV_ITEMS.indexOf(item);
  return i > 0 ? NAV_ITEMS[i - 1] : null;
}
