import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import { looseNumericMatch } from '../../numericMatch';
import { useStore } from '../../store';
import type { Conversation } from '../../types';
import { dayGroup, fmtRelative, shortRid } from './format';

/** Turns answered, and how many of those matched the gold the dataset carries. */
function tally(conv: Conversation): { turns: number; correct: number; scored: number } {
  let turns = 0;
  let correct = 0;
  let scored = 0;
  for (const message of conv.messages) {
    if (message.role !== 'assistant' || message.status !== 'done') continue;
    turns += 1;
    if (!message.goldAnswer) continue;
    scored += 1;
    if (looseNumericMatch(message.text, message.goldAnswer)) correct += 1;
  }
  return { turns, correct, scored };
}

function Spinner() {
  return (
    <svg className="size-3 shrink-0 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden>
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

interface RowProps {
  rid: string;
  meta: string;
  active: boolean;
  streaming?: boolean;
  unread?: number;
  example?: boolean;
  onSelect: () => void;
}

function Row({ rid, meta, active, streaming, unread, example, onSelect }: RowProps) {
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        title={rid}
        data-testid="sidebar-row"
        data-rid={rid}
        data-active={active ? 'true' : 'false'}
        data-example={example ? 'true' : 'false'}
        className={cn(
          'w-full border-l-2 px-2.5 py-2 text-left transition-colors',
          active
            ? 'border-l-amber bg-panel-2 text-text'
            : 'border-l-transparent text-muted hover:bg-panel-2/60 hover:text-text'
        )}
      >
        <div className="flex min-w-0 items-center gap-1.5">
          {streaming && <Spinner />}
          <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] leading-tight">
            {shortRid(rid)}
          </span>
          {!!unread && unread > 0 && (
            <span
              data-testid="unread-badge"
              title={`${unread} new answer${unread === 1 ? '' : 's'}`}
              className="inline-flex h-4 min-w-4 shrink-0 items-center justify-center rounded-full bg-amber px-1 font-mono text-[9px] font-semibold text-amber-ink"
            >
              {unread === 1 ? '' : unread}
            </span>
          )}
        </div>
        <div className="mt-0.5 truncate font-mono text-[10px] text-faint">{meta}</div>
      </button>
    </li>
  );
}

/**
 * The left pane: previous conversations grouped by day, then the recorded
 * examples.
 *
 * The examples exist so a first-time visitor never meets an empty list. They
 * are labelled `recorded` rather than presented as history, because they are
 * the demo pack's conversations — replayable on any deployment, including one
 * with no API key.
 */
export function SessionsPane({ onNew }: { onNew: () => void }) {
  const conversations = useStore((s) => s.conversations);
  const activeReportId = useStore((s) => s.activeReportId);
  const selectReport = useStore((s) => s.selectReport);
  const examples = useStore((s) => s.examples);

  const groups = useMemo(() => {
    const ordered = Object.values(conversations).sort((a, b) => b.lastUsedAt - a.lastUsedAt);
    const out: Array<{ label: string; items: Conversation[] }> = [];
    for (const conv of ordered) {
      const label = dayGroup(conv.lastUsedAt);
      const last = out[out.length - 1];
      if (last && last.label === label) last.items.push(conv);
      else out.push({ label, items: [conv] });
    }
    return out;
  }, [conversations]);

  // A recorded conversation the user has already opened is history now, not an
  // example — showing it in both places would double the row and the id.
  const unopened = examples.filter((e) => !conversations[e.reportId]);

  return (
    <>
      <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-b border-line px-2.5">
        <span className="mono-caps">sessions</span>
        <button
          type="button"
          onClick={onNew}
          data-testid="sidebar-new-conversation"
          className="rounded-[4px] border border-line-2 px-1.5 py-0.5 font-mono text-[10px] text-muted transition-colors hover:border-amber-line hover:text-amber"
        >
          + new
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden pb-3">
        {groups.length === 0 && (
          <p className="px-2.5 py-3 text-[11px] leading-relaxed text-faint">
            Your conversations will appear here. Start one with <em>+ new</em>, or open a
            recorded example below.
          </p>
        )}

        {groups.map((group) => (
          <section key={group.label}>
            <div className="sticky top-0 z-10 bg-ground px-2.5 pt-3 pb-1">
              <span className="mono-caps">{group.label}</span>
            </div>
            <ul>
              {group.items.map((conv) => {
                const { turns, correct, scored } = tally(conv);
                const meta = [
                  `${turns} turn${turns === 1 ? '' : 's'}`,
                  scored > 0 ? `${correct} ✓` : null,
                  fmtRelative(conv.lastUsedAt),
                ]
                  .filter(Boolean)
                  .join(' · ');
                return (
                  <Row
                    key={conv.reportId}
                    rid={conv.reportId}
                    meta={conv.isStreaming ? 'answering…' : meta}
                    active={conv.reportId === activeReportId}
                    streaming={conv.isStreaming}
                    unread={conv.unreadCount}
                    onSelect={() => void selectReport(conv.reportId)}
                  />
                );
              })}
            </ul>
          </section>
        ))}

        {unopened.length > 0 && (
          <section>
            <div className="sticky top-0 z-10 bg-ground px-2.5 pt-3 pb-1">
              <span className="mono-caps">examples</span>
            </div>
            <ul>
              {unopened.map((example) => (
                <Row
                  key={example.reportId}
                  rid={example.reportId}
                  meta={`${example.nQuestions} turns · recorded`}
                  active={example.reportId === activeReportId}
                  example
                  onSelect={() => void selectReport(example.reportId)}
                />
              ))}
            </ul>
          </section>
        )}
      </div>
    </>
  );
}
