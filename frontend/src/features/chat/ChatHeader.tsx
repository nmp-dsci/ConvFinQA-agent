import { useStore } from '../../store';

/**
 * The chat route's own header strip.
 *
 * What used to be `components/TopBar.tsx` was two things at once: app-wide
 * navigation and the chat's active-conversation controls. Navigation moved to
 * the nav rail, so only the conversation half survives here — same test ids,
 * same behaviour, one owner.
 */
export function ChatHeader() {
  const activeReportId = useStore((s) => s.activeReportId);
  const openPicker = useStore((s) => s.openPicker);

  return (
    <div className="flex h-10 shrink-0 items-center justify-between gap-3 border-b border-line bg-ground px-3">
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="mono-caps shrink-0">conversation</span>
        {activeReportId ? (
          <span className="truncate font-mono text-xs text-text" data-testid="active-report-id">
            {activeReportId}
          </span>
        ) : (
          <span className="text-xs text-faint italic">none selected</span>
        )}
      </div>

      <button
        type="button"
        onClick={openPicker}
        className="shrink-0 rounded-md border border-line-2 px-2.5 py-1 text-xs text-muted transition-colors hover:border-amber-line hover:text-amber"
        data-testid="topbar-change-report"
      >
        {activeReportId ? 'Change report' : '+ New conversation'}
      </button>
    </div>
  );
}
