import { useEffect, useRef, useState } from 'react';
import { useStore } from '../store';
import { Composer } from './Composer';
import { DocumentViewer } from './DocumentViewer';
import { LandingScreen } from './LandingScreen';
import { MessageBubble } from './MessageBubble';

function TrashIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="size-4"
      aria-hidden="true"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
      <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" />
    </svg>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`size-4 transition-transform ${open ? 'rotate-180' : ''}`}
      aria-hidden="true"
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

export function ChatPanel() {
  const activeReportId = useStore((s) => s.activeReportId);
  const conversation = useStore((s) =>
    s.activeReportId ? s.conversations[s.activeReportId] : undefined
  );
  const resetConversation = useStore((s) => s.resetConversation);

  const [docOpen, setDocOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Collapse the document viewer when switching reports.
  useEffect(() => {
    setDocOpen(false);
  }, [activeReportId]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [conversation?.messages.length, activeReportId]);

  useEffect(() => {
    // Smooth follow while streaming: nudge scroll on every event tick.
    if (!conversation?.isStreaming) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  });

  if (!activeReportId || !conversation) {
    return <LandingScreen />;
  }

  const canReset = !conversation.isStreaming && conversation.messages.length > 0;

  return (
    <section className="flex-1 flex flex-col min-h-0 min-w-0 bg-bg overflow-x-hidden">
      <div className="px-4 py-2 bg-panel2/40 border-b border-black/40 flex items-center justify-between gap-2 shrink-0">
        <div className="font-mono text-xs text-textMuted truncate">
          {activeReportId}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setDocOpen((v) => !v)}
            aria-expanded={docOpen}
            aria-controls="document-viewer"
            className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-panel2 hover:bg-accent"
            data-testid="toggle-document"
            title="Show the underlying financial document (pre-text, table, post-text)"
          >
            <ChevronIcon open={docOpen} />
            <span>{docOpen ? 'Hide document' : 'Show document'}</span>
          </button>
          <button
            type="button"
            onClick={() => {
              if (!canReset) return;
              const ok = window.confirm(
                'Reset this conversation? The agent will answer the next question with no prior history.'
              );
              if (ok) void resetConversation(activeReportId);
            }}
            disabled={!canReset}
            className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-panel2 hover:bg-danger disabled:opacity-30 disabled:hover:bg-panel2"
            data-testid="reset-conversation"
            title="Clear conversation history (next question goes to agent with no prior context)"
          >
            <TrashIcon />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {docOpen && <DocumentViewer reportId={activeReportId} />}

      <div ref={scrollRef} className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-3 min-w-0">
        {conversation.messages.length === 0 ? (
          <div className="text-textMuted text-sm italic text-center mt-12">
            No messages yet. Ask a question or click "Run all gold" below.
          </div>
        ) : (
          conversation.messages.map((m) => <MessageBubble key={m.id} message={m} />)
        )}
      </div>

      <Composer reportId={activeReportId} isStreaming={conversation.isStreaming} />
    </section>
  );
}
