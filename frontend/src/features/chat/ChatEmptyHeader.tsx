import { PanelLeft } from 'lucide-react';
import { useStore } from '../../store';

/**
 * The header strip the chat shows before a filing is chosen.
 *
 * It exists so `topbar-change-report` is reachable in *both* states. The
 * pre-redesign `ChatHeader` sat above the panes and was always mounted; the
 * Console moves those controls into the thread's own header, which would leave
 * the empty state with no way to open a conversation from the top bar. Three
 * e2e specs open the chat cold and click that control first.
 */
export function ChatEmptyHeader({ onShowSessions }: { onShowSessions: () => void }) {
  const openPicker = useStore((s) => s.openPicker);

  return (
    <div className="flex h-9 shrink-0 items-center gap-2 border-b border-line bg-panel px-3">
      <button
        type="button"
        onClick={onShowSessions}
        aria-label="Show sessions"
        className="rounded-[4px] border border-line-2 p-1 text-muted transition-colors hover:border-amber-line hover:text-amber md:hidden"
      >
        <PanelLeft className="size-3" aria-hidden />
      </button>
      <span className="mono-caps shrink-0">filing</span>
      <span className="min-w-0 flex-1 truncate text-[11.5px] text-faint italic">
        none selected
      </span>
      <button
        type="button"
        onClick={openPicker}
        data-testid="topbar-change-report"
        className="shrink-0 rounded-[4px] border border-line-2 px-1.5 py-0.5 font-mono text-[10px] text-muted transition-colors hover:border-amber-line hover:text-amber"
      >
        + new conversation
      </button>
    </div>
  );
}
