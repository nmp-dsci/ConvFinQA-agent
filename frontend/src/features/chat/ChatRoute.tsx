import { PanelLeft } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { cn } from '@/lib/utils';
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
 * The rail, the sessions pane and the inspector sit at `--ground`; the thread
 * is the one lit surface at `--panel`. The inspector is docked rather than
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
      navigate(`/chat/${activeReportId}`, { replace: true });
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
    <div className="relative flex h-full min-h-0 w-full overflow-hidden bg-ground">
      {/*
        Both panes are <aside>, so both expose the `complementary` landmark.
        Without names a screen-reader's landmark list reads "complementary,
        complementary" and the reader has to enter each one to find out which
        is which — axe's `landmark-unique`, and a real navigation cost on a
        three-pane layout whose whole point is moving between the panes.
      */}
      <aside
        aria-label="Sessions"
        className={cn(
          'z-30 w-[210px] shrink-0 flex-col border-r border-line bg-ground',
          'max-md:absolute max-md:inset-y-0 max-md:left-0 max-md:shadow-xl',
          sessionsOpen ? 'flex' : 'hidden md:flex'
        )}
      >
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
          className="absolute inset-0 z-20 bg-black/40 md:hidden"
        />
      )}

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {!activeReportId || !conversation ? (
          <div className="flex min-h-0 flex-1 flex-col">
            <ChatEmptyHeader onShowSessions={() => setSessionsOpen(true)} />
            <ChatEmpty />
          </div>
        ) : (
          <div className="flex min-h-0 min-w-0 flex-1">
            <div className="flex w-8 shrink-0 flex-col items-center border-r border-line bg-ground pt-2 md:hidden">
              <button
                type="button"
                onClick={() => setSessionsOpen(true)}
                aria-label="Show sessions"
                className="rounded-[4px] border border-line-2 p-1 text-muted"
              >
                <PanelLeft className="size-3" aria-hidden />
              </button>
            </div>
            <Thread
              conversation={conversation}
              selectedId={selectedMessage?.id ?? null}
              onSelect={setSelectedId}
              docOpen={docOpen}
              onToggleDoc={() => setDocOpen((v) => !v)}
              selectedMessage={selectedMessage}
              onToggleInspector={() => setInspectorOpen(true)}
            />
          </div>
        )}
      </div>

      {wide && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the trace inspector"
          onPointerDown={onDrag}
          onPointerMove={onDragMove}
          onPointerUp={onDragEnd}
          className="w-1 shrink-0 cursor-col-resize bg-line hover:bg-amber-line"
        />
      )}

      <aside
        aria-label="Trace inspector"
        style={wide ? { width } : undefined}
        className={cn(
          'z-30 shrink-0 flex-col border-l border-line bg-ground',
          'max-xl:absolute max-xl:inset-y-0 max-xl:right-0 max-xl:w-[min(320px,88vw)] max-xl:shadow-xl',
          inspectorOpen ? 'flex' : 'hidden xl:flex'
        )}
      >
        {inspectorOpen && (
          <button
            type="button"
            onClick={() => setInspectorOpen(false)}
            className="flex shrink-0 items-center justify-between border-b border-line px-2.5 py-1.5 text-left font-mono text-[10px] text-muted transition-colors hover:text-amber xl:hidden"
          >
            <span className="mono-caps">trace inspector</span>
            <span aria-hidden>close ×</span>
          </button>
        )}
        <Inspector message={selectedMessage} turnNumber={turnNumber} />
      </aside>

      <ReportPicker />
      <CommandPalette />
    </div>
  );
}
