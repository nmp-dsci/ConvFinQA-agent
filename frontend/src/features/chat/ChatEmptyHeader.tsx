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
    <div className="ch-head">
      <button
        type="button"
        onClick={onShowSessions}
        aria-label="Show sessions"
        className="tog ch-narrow-md"
      >
        ☰ sessions
      </button>
      <span className="label">filing</span>
      <span className="ch-fill ch-trunc small muted">— none selected</span>
      <button
        type="button"
        onClick={openPicker}
        data-testid="topbar-change-report"
        className="tog"
      >
        + new conversation
      </button>
    </div>
  );
}
