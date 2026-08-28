import { useMode } from '../modeStore';
import { useStore } from '../store';
import { Badge, Mono } from './ui';

export type AppTab = 'chat' | 'data' | 'traces' | 'experiments' | 'research' | 'eval';

const TABS: Array<{ key: AppTab; label: string; hint: string }> = [
  { key: 'chat', label: 'Chat', hint: 'Ask a question about a filing' },
  { key: 'data', label: 'Data & answers', hint: 'Splits, gold, and every version’s answer' },
  { key: 'traces', label: 'Traces', hint: 'Stage-by-stage record of every turn' },
  { key: 'experiments', label: 'Experiments', hint: 'Runs, registry, promotion history' },
  { key: 'research', label: 'Research', hint: 'Launch and inspect auto-research rounds' },
  { key: 'eval', label: 'Eval', hint: 'Accuracy slices per version' },
];

export function TopBar({
  activeTab,
  onTabChange,
  onHome,
}: {
  activeTab: AppTab;
  onTabChange: (t: AppTab) => void;
  onHome?: () => void;
}) {
  const activeReportId = useStore((s) => s.activeReportId);
  const openPicker = useStore((s) => s.openPicker);
  const health = useMode((s) => s.health);
  const isDemo = health?.mode === 'demo';

  return (
    <header className="flex items-center justify-between gap-4 px-4 py-2.5 bg-panel border-b border-black/40 shrink-0">
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onHome}
          title="Back to the overview"
          className="size-9 rounded-full bg-accent flex items-center justify-center font-semibold shrink-0 hover:brightness-110"
        >
          CF
        </button>

        <nav className="flex rounded-md overflow-hidden border border-black/40 shrink-0">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              title={tab.hint}
              onClick={() => onTabChange(tab.key)}
              className={`px-3 py-1 text-sm whitespace-nowrap transition-colors ${
                activeTab === tab.key
                  ? 'bg-accent2 text-bg font-medium'
                  : 'bg-panel2 text-textMuted hover:text-textMain'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {activeTab === 'chat' && (
          <div className="min-w-0">
            <div className="text-[11px] text-textMuted leading-none mb-1">Active conversation</div>
            {activeReportId ? (
              <div className="font-mono text-sm truncate" data-testid="active-report-id">
                {activeReportId}
              </div>
            ) : (
              <div className="text-sm text-textMuted italic">No conversation selected</div>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center gap-3 shrink-0">
        {health && (
          <span
            title={
              isDemo
                ? 'This deployment holds no API keys. Chat replays recorded conversations; everything else is live.'
                : `Live against ${health.bundle.lm_mini}`
            }
          >
            <Badge tone={isDemo ? 'warn' : 'good'}>
              {isDemo ? 'demo · read only' : 'dev · live'}
            </Badge>
          </span>
        )}
        {health?.champion && (
          <span className="text-[11px] text-textMuted hidden lg:inline">
            champion <Mono>{health.champion}</Mono>
          </span>
        )}
        {activeTab === 'chat' && (
          <button
            type="button"
            onClick={openPicker}
            className="px-3 py-1.5 text-sm rounded-md bg-panel2 hover:bg-accent text-textMain"
            data-testid="topbar-change-report"
          >
            {activeReportId ? 'Change report' : '+ New conversation'}
          </button>
        )}
      </div>
    </header>
  );
}
