import { useStore } from '../store';

export type AppTab = 'chat' | 'eval';

export function TopBar({
  activeTab,
  onTabChange,
}: {
  activeTab: AppTab;
  onTabChange: (t: AppTab) => void;
}) {
  const activeReportId = useStore((s) => s.activeReportId);
  const openPicker = useStore((s) => s.openPicker);

  return (
    <header className="flex items-center justify-between px-4 py-3 bg-panel border-b border-black/40 shrink-0">
      <div className="flex items-center gap-3 min-w-0">
        <div className="size-9 rounded-full bg-accent flex items-center justify-center font-semibold shrink-0">
          CF
        </div>
        <div className="flex rounded-md overflow-hidden border border-black/40 shrink-0">
          {(['chat', 'eval'] as AppTab[]).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => onTabChange(tab)}
              className={`px-3 py-1 text-sm capitalize transition-colors ${
                activeTab === tab
                  ? 'bg-accent2 text-bg font-medium'
                  : 'bg-panel2 text-textMuted hover:text-textMain'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
        {activeTab === 'chat' && (
          <div className="min-w-0">
            <div className="text-sm text-textMuted leading-none mb-1">Active conversation</div>
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
      {activeTab === 'chat' && (
        <button
          type="button"
          onClick={openPicker}
          className="px-3 py-1.5 text-sm rounded-md bg-panel2 hover:bg-accent text-textMain shrink-0"
          data-testid="topbar-change-report"
        >
          {activeReportId ? 'Change report' : '+ New conversation'}
        </button>
      )}
    </header>
  );
}
