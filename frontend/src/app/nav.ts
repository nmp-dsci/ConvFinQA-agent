/**
 * Navigation is data, told once: the house's eight slots (ui/DESIGN.md §2) in story order —
 * what the benchmark is → how it is judged → who else has tried → what we ran → how we improve
 * → what the agent is → what we know by hand. The header's tabs, each slot's chip bar and the
 * command palette all read from here.
 */
import type { Tab } from '@/kit/Shell';

export type SlotKey =
  | 'overview'
  | 'filings'
  | 'scoring'
  | 'leaderboard'
  | 'runs'
  | 'optimise'
  | 'agent'
  | 'gold';

export interface Slot extends Tab {
  key: SlotKey;
}

export const SLOTS: Slot[] = [
  { key: 'overview', to: '/', label: 'Overview', hint: 'The result, with its baseline and its caveat' },
  {
    key: 'filings',
    to: '/filings',
    label: 'Filings & questions',
    hint: 'Every filing and question in every split, beside its gold answer and gold program',
  },
  {
    key: 'scoring',
    to: '/scoring',
    label: 'Scoring',
    hint: 'How a turn is judged: execution accuracy, attribution, the promotion gate, the judge',
  },
  {
    key: 'leaderboard',
    to: '/leaderboard',
    label: 'Leaderboard',
    hint: "The paper's baselines and the human expert beside our two runtimes, on one axis",
  },
  {
    key: 'runs',
    to: '/runs',
    label: 'Runs',
    hint: 'Every test pass of every agent, every served turn, and the legacy 770-question versions',
  },
  {
    key: 'optimise',
    to: '/optimise',
    label: 'Optimise',
    hint: 'The whole story on the gate split, the campaigns, the runtime comparison, and rule research',
  },
  {
    key: 'agent',
    to: '/agent',
    label: 'Agent',
    hint: 'Four agents, one choke point, one contract — and a conversation with it',
  },
  {
    key: 'gold',
    to: '/gold',
    label: 'Gold review',
    hint: 'Gold labels the loop flagged as suspect or could not settle, and how each was settled',
  },
];

/** The pages inside a slot, as its chip bar. A slot with one page has no bar. */
export const SUBPAGES: Partial<Record<SlotKey, { key: string; to: string; label: string }[]>> = {
  runs: [
    { key: 'passes', to: '/runs', label: 'Every test pass' },
    { key: 'traces', to: '/runs/traces', label: 'Served turns' },
    { key: 'legacy', to: '/runs/legacy', label: 'Legacy v1–v3_1' },
  ],
  optimise: [
    { key: 'story', to: '/optimise', label: 'Story' },
    { key: 'campaigns', to: '/optimise/campaigns', label: 'Campaigns' },
    { key: 'runtimes', to: '/optimise/runtimes', label: 'Runtimes' },
    { key: 'research', to: '/optimise/research', label: 'Rule research' },
  ],
  agent: [
    { key: 'architecture', to: '/agent', label: 'Architecture' },
    { key: 'readiness', to: '/agent/readiness', label: 'Readiness R1–R9' },
    { key: 'chat', to: '/agent/chat', label: 'Ask a filing' },
  ],
};

/** Every addressable page, flat: the command palette's list. */
export const NAV_ITEMS: { to: string; label: string; hint?: string }[] = SLOTS.flatMap((s) => {
  const subs = SUBPAGES[s.key];
  if (!subs) return [{ to: s.to, label: s.label, hint: s.hint }];
  return subs.map((p) => ({ to: p.to, label: `${s.label} · ${p.label}`, hint: s.hint }));
});

/** The slot a path belongs to. `/` only matches itself. */
export function slotOf(pathname: string): Slot | null {
  if (pathname === '/') return SLOTS[0];
  return SLOTS.slice(1).find((s) => pathname === s.to || pathname.startsWith(`${s.to}/`)) ?? null;
}

/** The sub-page a path is on, most specific first (`/runs/traces/x` → traces, not passes). */
export function subpageOf(pathname: string): string | undefined {
  const slot = slotOf(pathname);
  const subs = slot ? SUBPAGES[slot.key] : undefined;
  if (!subs) return undefined;
  const hit = [...subs]
    .sort((a, b) => b.to.length - a.to.length)
    .find((p) => pathname === p.to || pathname.startsWith(`${p.to}/`));
  return hit?.key;
}
