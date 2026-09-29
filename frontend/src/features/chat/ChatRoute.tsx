import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useStore } from '../../store';
import type { Message } from '../../types';
import { ChatEmpty } from './ChatEmpty';
import { ChatEmptyHeader } from './ChatEmptyHeader';
import { CommandPalette } from './CommandPalette';
import { Inspector } from './Inspector';
import { ReportPicker } from './ReportPicker';
import { SessionsPane } from './SessionsPane';
import { Thread } from './Thread';
import { useMediaQuery } from './useMediaQuery';
import './chat.css';

const INSPECTOR_KEY = 'convfinqa.inspectorWidth';
const INSPECTOR_MIN = 240;
const INSPECTOR_MAX = 560;
const INSPECTOR_DEFAULT = 290;

function storedWidth(): number {
  try {
    const raw = Number.parseInt(window.localStorage.getItem(INSPECTOR_KEY) ?? '', 10);
    if (Number.isFinite(raw)) return Math.min(INSPECTOR_MAX, Math.max(INSPECTOR_MIN, raw));
  } catch {
    /* private mode — the default is fine */
  }
  return INSPECTOR_DEFAULT;
}

/**
 * The Console's chat: sessions · thread · docked inspector.
 *
 * The sessions pane and the inspector are `--panel` chrome split by `--line`;
 * the thread sits on `--bg` with each turn a card. The inspector is docked rather than
 * modal on purpose — a system whose selling point is that it shows its work
 * cannot put the work behind a click.
 *
 * Below 1280 px the inspector becomes a drawer and below 768 px the sessions
 * pane does too, but each is a single DOM node in both layouts: duplicating
 * them would duplicate `sidebar-row` and `stage-output-panel`, which several
 * specs address as unique.
 */
export function ChatRoute() {
  const params = useParams();
  const navigate = useNavigate();

  const activeReportId = useStore((s) => s.activeReportId);
  const conversation = useStore((s) =>
    s.activeReportId ? s.conversations[s.activeReportId] : undefined
  );
  const selectReport = useStore((s) => s.selectReport);
  const openPicker = useStore((s) => s.openPicker);
  const loadExamples = useStore((s) => s.loadExamples);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [docOpen, setDocOpen] = useState(false);
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [width, setWidth] = useState(storedWidth);
  const wide = useMediaQuery('(min-width: 1280px)');
  const dragging = useRef(false);

  const routeReportId = params['*'] ? decodeURIComponent(params['*']) : '';

  useEffect(() => {
    void loadExamples();
  }, [loadExamples]);

  useEffect(() => {
    if (!routeReportId || routeReportId === activeReportId) return;
    void selectReport(routeReportId);
  }, [routeReportId, activeReportId, selectReport]);

  // Keep the URL honest when the conversation changes from the pane, the
  // picker or the palette, so a copied link points at what is on screen.
  useEffect(() => {
    if (!routeReportId) return;
    if (activeReportId && activeReportId !== routeReportId) {
      navigate(`/agent/chat/${activeReportId}`, { replace: true });
    }
  }, [activeReportId, routeReportId, navigate]);

  // Switching filings resets the per-conversation panels rather than leaving
  // one filing's table open over another's thread.
  useEffect(() => {
    setDocOpen(false);
    setSelectedId(null);
    setSessionsOpen(false);
  }, [activeReportId]);

  const messages = conversation?.messages;
  const latestAssistant = useMemo(() => {
    if (!messages) return null;
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      if (message.role === 'assistant') return message;
    }
    return null;
  }, [messages]);

  // The inspector follows the newest turn unless the reader pinned an older
  // one — watching a stream and having the panel jump backwards is worse than
  // having to click once to go back.
  const selectedMessage: Message | null = useMemo(() => {
    if (!messages) return null;
    if (selectedId) {
      const pinned = messages.find((m) => m.id === selectedId);
      if (pinned) return pinned;
    }
    return latestAssistant;
  }, [messages, selectedId, latestAssistant]);

  const turnNumber = useMemo(() => {
    if (!messages || !selectedMessage) return 0;
    let n = 0;
    for (const message of messages) {
      if (message.role !== 'assistant') continue;
      n += 1;
      if (message.id === selectedMessage.id) return n;
    }
    return n;
  }, [messages, selectedMessage]);

  const onDrag = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
  }, []);

  const onDragMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return;
    const next = Math.min(
      INSPECTOR_MAX,
      Math.max(INSPECTOR_MIN, window.innerWidth - event.clientX)
    );
    setWidth(next);
  }, []);

  const onDragEnd = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      dragging.current = false;
      event.currentTarget.releasePointerCapture(event.pointerId);
      try {
        window.localStorage.setItem(INSPECTOR_KEY, String(width));
      } catch {
        /* the width holds for this session and no longer */
      }
    },
    [width]
  );

  return (
    <div className="ch-root">
      {/*
        Both panes are <aside>, so both expose the `complementary` landmark.
        Without names a screen-reader's landmark list reads "complementary,
        complementary" and the reader has to enter each one to find out which
        is which — axe's `landmark-unique`, and a real navigation cost on a
        three-pane layout whose whole point is moving between the panes.
      */}
      <aside aria-label="Sessions" className={`ch-sessions${sessionsOpen ? ' open' : ''}`}>
        <SessionsPane
          onNew={() => {
            setSessionsOpen(false);
            openPicker();
          }}
        />
      </aside>

      {sessionsOpen && (
        <button
          type="button"
          aria-label="Close the sessions pane"
          onClick={() => setSessionsOpen(false)}
          className="ch-scrim ch-narrow-md"
        />
      )}

      <div className="ch-main">
        {!activeReportId || !conversation ? (
          <>
            <ChatEmptyHeader onShowSessions={() => setSessionsOpen(true)} />
            <ChatEmpty />
          </>
        ) : (
          <Thread
            conversation={conversation}
            selectedId={selectedMessage?.id ?? null}
            onSelect={setSelectedId}
            docOpen={docOpen}
            onToggleDoc={() => setDocOpen((v) => !v)}
            selectedMessage={selectedMessage}
            onShowSessions={() => setSessionsOpen(true)}
            onToggleInspector={() => setInspectorOpen(true)}
          />
        )}
      </div>

      {wide && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the trace inspector"
          aria-valuemin={INSPECTOR_MIN}
          aria-valuemax={INSPECTOR_MAX}
          aria-valuenow={width}
          tabIndex={0}
          title="Drag, or use the arrow keys, to resize the trace inspector"
          onPointerDown={onDrag}
          onPointerMove={onDragMove}
          onPointerUp={onDragEnd}
          onKeyDown={(event) => {
            // The pointer path is the common one; this is the one a keyboard
            // reader has. Arrow keys step the width, Home/End snap to the ends.
            const step = event.shiftKey ? 48 : 16;
            let next: number | null = null;
            if (event.key === 'ArrowLeft') next = width + step;
            else if (event.key === 'ArrowRight') next = width - step;
            else if (event.key === 'Home') next = INSPECTOR_MAX;
            else if (event.key === 'End') next = INSPECTOR_MIN;
            if (next === null) return;
            event.preventDefault();
            const clamped = Math.min(INSPECTOR_MAX, Math.max(INSPECTOR_MIN, next));
            setWidth(clamped);
            try {
              window.localStorage.setItem(INSPECTOR_KEY, String(clamped));
            } catch {
              /* the width holds for this session and no longer */
            }
          }}
          className="ch-resize"
        />
      )}

      {inspectorOpen && (
        <button
          type="button"
          aria-label="Close the trace inspector"
          onClick={() => setInspectorOpen(false)}
          className="ch-scrim ch-narrow-xl"
        />
      )}

      <aside
        aria-label="Trace inspector"
        style={wide ? { width } : undefined}
        className={`ch-inspector${inspectorOpen ? ' open' : ''}`}
      >
        {inspectorOpen && (
          <div className="ch-head ch-narrow-xl">
            <span className="label ch-fill">trace inspector</span>
            <button type="button" className="tog" onClick={() => setInspectorOpen(false)}>
              ✕ close
            </button>
          </div>
        )}
        <Inspector message={selectedMessage} turnNumber={turnNumber} />
      </aside>

      <ReportPicker />
      <CommandPalette />
    </div>
  );
}
