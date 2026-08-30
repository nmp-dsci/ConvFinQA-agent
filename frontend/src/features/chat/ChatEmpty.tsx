import { useStore } from '../../store';
import { shortRid } from './format';

const STEPS: Array<{ title: string; body: string }> = [
  {
    title: 'Pick a report',
    body: 'Every conversation is anchored to one filing — a single page of a 10-K or 10-Q, with its text and its table.',
  },
  {
    title: 'Ask a question',
    body: 'Type freely, or take one of the suggested questions. Later turns may refer back to earlier answers; resolving that is the pipeline’s job.',
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
 * exists after the marketing landing moves to `/`.
 */
export function ChatEmpty() {
  const conversations = useStore((s) => s.conversations);
  const examples = useStore((s) => s.examples);
  const openPicker = useStore((s) => s.openPicker);
  const selectReport = useStore((s) => s.selectReport);

  const known = Object.values(conversations).sort((a, b) => b.lastUsedAt - a.lastUsedAt);
  const variant = known.length === 0 ? 'new' : 'returning';

  return (
    <div
      data-testid="landing-screen"
      data-variant={variant}
      className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto bg-panel p-6"
    >
      <div className="w-full max-w-lg">
        <div className="mono-caps mb-1.5">multi-turn financial QA · four-agent pipeline</div>
        <h1 className="text-xl font-medium tracking-tight text-text">
          {variant === 'new' ? 'Ask a filing a question' : 'Pick up where you left off'}
        </h1>

        {variant === 'new' ? (
          <ol className="mt-4 space-y-3">
            {STEPS.map((step, i) => (
              <li key={step.title} className="flex gap-2.5">
                <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-[4px] border border-amber-line bg-amber-soft font-mono text-[10px] text-amber">
                  {i + 1}
                </span>
                <div className="min-w-0">
                  <div className="text-[13px] font-medium text-text">{step.title}</div>
                  <p className="text-[12px] leading-relaxed text-muted">{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
        ) : (
          <ul className="mt-4 space-y-1">
            {known.slice(0, 5).map((conv) => (
              <li key={conv.reportId}>
                <button
                  type="button"
                  onClick={() => void selectReport(conv.reportId)}
                  className="flex w-full items-baseline gap-2 rounded-md border border-line px-2.5 py-1.5 text-left transition-colors hover:border-amber-line"
                >
                  <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-text">
                    {conv.reportId}
                  </span>
                  <span className="shrink-0 font-mono text-[10px] text-faint">
                    {conv.messages.filter((m) => m.role === 'assistant').length} turns
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="mt-5 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={openPicker}
            data-testid="landing-cta"
            className="rounded-md bg-amber px-3 py-1.5 text-[12px] font-semibold text-amber-ink transition-opacity hover:opacity-90"
          >
            + New conversation
          </button>
          <span className="font-mono text-[10px] text-faint">or press ⌘K</span>
        </div>

        {examples.length > 0 && (
          <div className="mt-6 border-t border-line pt-3">
            <div className="mono-caps mb-1.5">recorded conversations</div>
            <div className="flex flex-wrap gap-1.5">
              {examples.slice(0, 6).map((example) => (
                <button
                  key={example.reportId}
                  type="button"
                  title={example.reportId}
                  onClick={() => void selectReport(example.reportId)}
                  className="rounded-full border border-line-2 px-2 py-0.5 font-mono text-[10.5px] text-muted transition-colors hover:border-amber-line hover:text-amber"
                >
                  {shortRid(example.reportId)} · {example.nQuestions} turns
                </button>
              ))}
            </div>
            <p className="mt-1.5 text-[10.5px] leading-relaxed text-faint">
              These replay stage-for-stage from recordings made in development, so they work on a
              deployment with no API key.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
