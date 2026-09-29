import { useMemo } from 'react';
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

/** A turning glyph, not a colour: the row also says "answering…" in words. */
function Spinner() {
  return (
    <span className="ch-spin" aria-hidden>
      ◐
    </span>
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
        className={active ? 'ch-row on' : 'ch-row'}
      >
        <span className="ch-row-top">
          {streaming && <Spinner />}
          <span className="ch-row-id ch-trunc">{shortRid(rid)}</span>
          {!!unread && unread > 0 && (
            <span
              data-testid="unread-badge"
              title={`${unread} new answer${unread === 1 ? '' : 's'}`}
              className="ch-unread"
            >
              {unread} new
            </span>
          )}
        </span>
        <span className="ch-meta ch-trunc">{meta}</span>
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
      <div className="ch-head">
        <span className="label ch-fill">sessions</span>
        <button type="button" onClick={onNew} data-testid="sidebar-new-conversation" className="tog">
          + new
        </button>
      </div>

      <div className="ch-scroll-y">
        {groups.length === 0 && (
          <p className="ch-note">
            Your conversations will appear here. Start one with <b>+ new</b>, or open a recorded
            example below.
          </p>
        )}

        {groups.map((group) => (
          <section key={group.label}>
            <div className="ch-group">
              <span className="label">{group.label}</span>
            </div>
            <ul className="ch-list">
              {group.items.map((conv) => {
                const { turns, correct, scored } = tally(conv);
                const meta = [
                  `${turns} turn${turns === 1 ? '' : 's'}`,
                  scored > 0 ? `${correct}/${scored} ✓ gold` : null,
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
            <div className="ch-group">
              <span className="label">recorded · replayable without a key</span>
            </div>
            <ul className="ch-list">
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
