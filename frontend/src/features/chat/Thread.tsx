import { PanelRight, Table2, Trash2 } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';
import { useIsDemo } from '../../modeStore';
import { useStore } from '../../store';
import type { Conversation, Message } from '../../types';
import { Composer } from './Composer';
import { FilingDrawer } from './FilingDrawer';
import { Turn } from './Turn';

function ModeChip() {
  const isDemo = useIsDemo();
  return (
    <span
      title={
        isDemo
          ? 'This deployment holds no API key. Every turn is a recording made in development, replayed.'
          : 'Live: every turn is a real model call'
      }
      className={cn(
        'shrink-0 rounded-full px-1.5 py-0.5 font-mono text-[9.5px] leading-tight',
        isDemo
          ? 'border border-dashed border-amber-line text-amber'
          : 'border border-good-line text-good'
      )}
    >
      {isDemo ? 'replay' : 'live'}
    </span>
  );
}

interface Props {
  conversation: Conversation;
  selectedId: string | null;
  onSelect: (id: string) => void;
  docOpen: boolean;
  onToggleDoc: () => void;
  selectedMessage: Message | null;
  /** Only rendered below the inspector's breakpoint, where it is a drawer. */
  onToggleInspector?: () => void;
}

export function Thread({
  conversation,
  selectedId,
  onSelect,
  docOpen,
  onToggleDoc,
  selectedMessage,
  onToggleInspector,
}: Props) {
  const openPicker = useStore((s) => s.openPicker);
  const resetConversation = useStore((s) => s.resetConversation);
  const scrollRef = useRef<HTMLDivElement>(null);
  const rid = conversation.reportId;

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [conversation.messages.length, rid]);

  useEffect(() => {
    // Follow the stream: every event tick nudges the view to the bottom.
    if (!conversation.isStreaming) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  });

  const canReset = !conversation.isStreaming && conversation.messages.length > 0;

  return (
    <div className="relative flex min-h-0 min-w-0 flex-1 flex-col bg-panel">
      {/*
        Wraps rather than clips below ~520 px. A header that scrolls its own
        controls off the right edge hides the reset and inspector buttons on a
        phone with nothing on screen to say they exist.
      */}
      <div className="flex min-h-9 shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b border-line px-3 py-1">
        <span className="mono-caps shrink-0">filing</span>
        {/*
          An <h1>, not a <span>: the thread is about exactly one filing, and
          that filing's id is the page's subject. Without it the chat has no
          level-one heading at all, which is what axe flagged and what leaves a
          screen-reader user with no way to answer "what am I looking at" other
          than reading the transcript. The text is unchanged — `smoke.spec.ts`
          asserts this element's text equals the report id exactly.
        */}
        <h1
          data-testid="active-report-id"
          title={rid}
          className="min-w-0 flex-1 truncate font-mono text-[11.5px] font-normal text-text"
        >
          {rid}
        </h1>
        <ModeChip />

        <button
          type="button"
          onClick={onToggleDoc}
          aria-expanded={docOpen}
          data-testid="toggle-document"
          title="Show the filing, with the cells this turn's retriever returned highlighted"
          className={cn(
            'flex shrink-0 items-center gap-1 rounded-[4px] border px-1.5 py-0.5 font-mono text-[10px] transition-colors',
            docOpen
              ? 'border-amber-line bg-amber-soft text-amber'
              : 'border-line-2 text-muted hover:border-amber-line hover:text-amber'
          )}
        >
          <Table2 className="size-3" aria-hidden />
          filing
        </button>

        <button
          type="button"
          onClick={openPicker}
          data-testid="topbar-change-report"
          title="Open a different filing"
          className="shrink-0 rounded-[4px] border border-line-2 px-1.5 py-0.5 font-mono text-[10px] text-muted transition-colors hover:border-amber-line hover:text-amber"
        >
          change
        </button>

        <button
          type="button"
          disabled={!canReset}
          onClick={() => {
            if (!canReset) return;
            const ok = window.confirm(
              'Reset this conversation? The agent will answer the next question with no prior history.'
            );
            if (ok) void resetConversation(rid);
          }}
          data-testid="reset-conversation"
          title="Clear the history so the next question is answered with no prior context"
          className="flex shrink-0 items-center gap-1 rounded-[4px] border border-line-2 px-1.5 py-0.5 font-mono text-[10px] text-muted transition-colors hover:border-bad hover:text-bad disabled:opacity-30 disabled:hover:border-line-2 disabled:hover:text-muted"
        >
          <Trash2 className="size-3" aria-hidden />
          reset
        </button>

        {onToggleInspector && (
          <button
            type="button"
            onClick={onToggleInspector}
            title="Show the trace inspector"
            className="shrink-0 rounded-[4px] border border-line-2 p-1 text-muted transition-colors hover:border-amber-line hover:text-amber xl:hidden"
          >
            <PanelRight className="size-3" aria-hidden />
          </button>
        )}
      </div>

      {/*
        The drawer covers the transcript, never the header or the composer:
        the control that opened it has to stay reachable to close it, and
        reading the table while typing the next question is the whole point.
      */}
      <div className="relative min-h-0 min-w-0 flex-1">
        <div
          ref={scrollRef}
          className="absolute inset-0 overflow-y-auto overflow-x-hidden px-3 py-2"
        >
          {conversation.messages.length === 0 ? (
            <p className="mt-10 text-center text-[12px] leading-relaxed text-faint">
              No turns yet. Ask a question below, or pick one of the suggested questions to watch
              all four stages run.
            </p>
          ) : (
            conversation.messages.map((message) => (
              <Turn
                key={message.id}
                message={message}
                selected={message.id === selectedId}
                onSelect={onSelect}
              />
            ))
          )}
        </div>

        {docOpen && (
          <FilingDrawer reportId={rid} message={selectedMessage} onClose={onToggleDoc} />
        )}
      </div>

      <Composer reportId={rid} isStreaming={conversation.isStreaming} />
    </div>
  );
}
