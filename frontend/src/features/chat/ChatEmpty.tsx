import { useStore } from '../../store';
import { shortRid } from './format';

const STEPS: Array<{ title: string; body: string }> = [
  {
    title: 'Pick a report',
    body: 'Every conversation is anchored to one filing — a single page of a 10-K or 10-Q, with its text and its table. 442 are available.',
  },
  {
    title: 'Ask a question',
    body: 'Type freely, or take one of the dataset’s questions. Later turns refer back to earlier answers — “that change”, “this sum” — and resolving them is the agent’s job.',
  },
  {
    title: 'Run all gold',
    body: 'Runs every dataset question for the filing in order and marks each answer against the gold the dataset carries.',
  },
];

/**
 * What the thread shows before a filing is chosen.
 *
 * Keeps the `landing-screen` / `landing-cta` ids and the `data-variant`
 * new/returning distinction: they name the chat's empty state, which still
 * exists after the marketing landing moves to `/`. Its "+ New conversation" is
 * the one solid button on this screen.
 */
export function ChatEmpty() {
  const conversations = useStore((s) => s.conversations);
  const examples = useStore((s) => s.examples);
  const openPicker = useStore((s) => s.openPicker);
  const selectReport = useStore((s) => s.selectReport);

  const known = Object.values(conversations).sort((a, b) => b.lastUsedAt - a.lastUsedAt);
  const variant = known.length === 0 ? 'new' : 'returning';

  return (
    <div data-testid="landing-screen" data-variant={variant} className="ch-empty">
      <div className="ch-empty-in">
        <p className="label">multi-turn financial QA · one session per conversation</p>
        <h1>
          {variant === 'new' ? (
            <>
              Ask a <em>filing</em> a question
            </>
          ) : (
            <>
              Pick up where you <em>left off</em>
            </>
          )}
        </h1>

        {variant === 'new' ? (
          <ol className="steps">
            {STEPS.map((step) => (
              <li key={step.title}>
                <b>{step.title}.</b> {step.body}
              </li>
            ))}
          </ol>
        ) : (
          <ul className="ch-recent">
            {known.slice(0, 5).map((conv) => (
              <li key={conv.reportId}>
                <button type="button" onClick={() => void selectReport(conv.reportId)}>
                  <span className="ch-row-id ch-trunc">{conv.reportId}</span>
                  <span className="ch-meta">
                    {conv.messages.filter((m) => m.role === 'assistant').length} turns
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="row">
          <button type="button" onClick={openPicker} data-testid="landing-cta" className="btn">
            + New conversation
          </button>
          <span className="small muted">
            or press <kbd className="mono">⌘K</kbd>
          </span>
        </div>

        {examples.length > 0 && (
          <div className="ch-examples">
            <p className="label">recorded conversations · replayable without a key</p>
            <div className="chips">
              {examples.slice(0, 6).map((example) => (
                <button
                  key={example.reportId}
                  type="button"
                  title={example.reportId}
                  onClick={() => void selectReport(example.reportId)}
                  className="chip"
                >
                  ▸ {shortRid(example.reportId)} · {example.nQuestions} turns
                </button>
              ))}
            </div>
            <p className="small muted">
              These replay stage-for-stage from recordings made in development, so they work on a
              deployment with no API key.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
