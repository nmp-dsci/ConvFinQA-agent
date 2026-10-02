import { useEffect, useRef } from 'react';
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
      className={isDemo ? 'tag warn ch-replay' : 'tag ok'}
    >
      {isDemo ? '◌ recorded replay' : '● live model'}
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
  /** Only shown below the sessions pane's breakpoint, where it is a drawer. */
  onShowSessions?: () => void;
  /** Only shown below the inspector's breakpoint, where it is a drawer. */
  onToggleInspector?: () => void;
}

export function Thread({
  conversation,
  selectedId,
  onSelect,
  docOpen,
  onToggleDoc,
  selectedMessage,
  onShowSessions,
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
  const isDemo = useIsDemo();

  return (
    <div className="ch-thread">
      {/*
        Wraps rather than clips below ~520 px. A header that scrolls its own
        controls off the right edge hides the reset and inspector buttons on a
        phone with nothing on screen to say they exist.
      */}
      <div className="ch-head">
        {onShowSessions && (
          <button
            type="button"
            onClick={onShowSessions}
            aria-label="Show sessions"
            className="tog ch-narrow-md"
          >
            ☰ sessions
          </button>
        )}
        <span className="label">filing</span>
        {/*
          An <h1>, not a <span>: the thread is about exactly one filing, and
          that filing's id is the page's subject. Without it the chat has no
          level-one heading at all, which is what axe flagged and what leaves a
          screen-reader user with no way to answer "what am I looking at" other
          than reading the transcript. The text is unchanged — `smoke.spec.ts`
          asserts this element's text equals the report id exactly.
        */}
        <h1 data-testid="active-report-id" title={rid} className="ch-title ch-fill ch-trunc">
          {rid}
        </h1>
        <ModeChip />

        <button
          type="button"
          onClick={onToggleDoc}
          aria-expanded={docOpen}
          data-testid="toggle-document"
          title="Show the filing, with the cells this turn's retriever returned highlighted"
          className={docOpen ? 'tog on' : 'tog'}
        >
          ▤ filing
        </button>

        <button
          type="button"
          onClick={openPicker}
          data-testid="topbar-change-report"
          title="Open a different filing"
          className="tog"
        >
          ⇄ change
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
          className="tog"
        >
          ↻ reset
        </button>

        {onToggleInspector && (
          <button
            type="button"
            onClick={onToggleInspector}
            aria-label="Show the trace inspector"
            title="Show the trace inspector"
            className="tog ch-narrow-xl"
          >
            ◧ trace
          </button>
        )}
      </div>

      {/*
        The drawer covers the transcript, never the header or the composer:
        the control that opened it has to stay reachable to close it, and
        reading the table while typing the next question is the whole point.
      */}
      {isDemo && (
        <p data-testid="demo-banner" className="ch-banner">
          <b className="ch-kicker">◐ demo</b>
          Replaying a recorded run at its real pacing — the stage events, tool calls and timings
          captured in development. No model is called on this deployment.
        </p>
      )}

      <div className="ch-body">
        <div ref={scrollRef} className="ch-scroll">
          <div className="ch-turns">
            {conversation.messages.length === 0 ? (
              <p className="empty ch-none">
                No turns yet. Ask a question below, or pick one of the suggested questions to
                watch all four stages run.
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
        </div>

        {docOpen && (
          <FilingDrawer reportId={rid} message={selectedMessage} onClose={onToggleDoc} />
        )}
      </div>

      <Composer reportId={rid} isStreaming={conversation.isStreaming} />
    </div>
  );
}
