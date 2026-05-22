import { useStore } from '../store';
import type { Conversation } from '../types';

function formatRelative(ms: number): string {
  const diff = Date.now() - ms;
  const sec = Math.floor(diff / 1000);
  if (sec < 5) return 'Just now';
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  return `${day}d ago`;
}

function previewOf(conv: Conversation): string {
  if (conv.isStreaming) return 'typing…';
  if (conv.messages.length === 0) return '(no messages yet)';
  const last = conv.messages[conv.messages.length - 1];
  if (!last) return '(no messages yet)';
  if (last.role === 'assistant' && !last.text) return 'thinking…';
  const text = last.text || '(empty)';
  return text.length > 60 ? `${text.slice(0, 59)}…` : text;
}

function Spinner() {
  return (
    <svg
      className="size-3 animate-spin shrink-0"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="3"
        strokeDasharray="40 60"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function Sidebar() {
  const conversations = useStore((s) => s.conversations);
  const activeReportId = useStore((s) => s.activeReportId);
  const openPicker = useStore((s) => s.openPicker);
  const selectReport = useStore((s) => s.selectReport);

  const ordered = Object.values(conversations).sort((a, b) => b.lastUsedAt - a.lastUsedAt);

  return (
    <aside className="bg-panel border-r border-black/40 flex flex-col min-h-0 overflow-hidden">
      <div className="px-4 py-3 flex items-center justify-between border-b border-black/40">
        <h2 className="font-semibold text-sm tracking-wide">Conversations</h2>
        <button
          type="button"
          onClick={openPicker}
          className="text-xs px-2 py-1 rounded-md bg-panel2 hover:bg-accent"
          data-testid="sidebar-new-conversation"
        >
          + New
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {ordered.length === 0 ? (
          <div className="p-4 text-sm text-textMuted italic">
            Your conversations will appear here. Click + New to start.
          </div>
        ) : (
          <ul>
            {ordered.map((conv) => (
              <li key={conv.reportId}>
                <button
                  type="button"
                  onClick={() => void selectReport(conv.reportId)}
                  data-testid="sidebar-row"
                  data-rid={conv.reportId}
                  data-active={conv.reportId === activeReportId ? 'true' : 'false'}
                  className={`w-full text-left px-4 py-3 border-b border-black/30 hover:bg-panel2 ${
                    conv.reportId === activeReportId ? 'bg-panel2' : ''
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    {conv.isStreaming && <Spinner />}
                    <span className="font-mono text-sm truncate flex-1">{conv.reportId}</span>
                    <span className="text-[10px] text-textMuted shrink-0">
                      {formatRelative(conv.lastUsedAt)}
                    </span>
                    {conv.unreadCount > 0 && (
                      <span
                        data-testid="unread-badge"
                        className="ml-1 inline-flex items-center justify-center min-w-4 h-4 px-1 rounded-full text-[10px] font-semibold bg-accent2 text-bg"
                      >
                        {conv.unreadCount === 1 ? '' : conv.unreadCount}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-textMuted truncate mt-0.5">
                    {previewOf(conv)}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
