import { useStore } from '../store';

export function LandingScreen() {
  const conversationCount = useStore((s) => Object.keys(s.conversations).length);
  const openPicker = useStore((s) => s.openPicker);

  const variant = conversationCount === 0 ? 'new' : 'returning';

  return (
    <div
      data-testid="landing-screen"
      data-variant={variant}
      className="flex-1 flex items-center justify-center p-8"
    >
      <div className="max-w-xl w-full bg-panel rounded-xl p-8 shadow-lg border border-black/40">
        {variant === 'new' ? (
          <>
            <h1 className="text-3xl font-bold mb-2 text-textMain">ConvFinQA Agent</h1>
            <p className="text-textMuted mb-6">
              Multi-turn financial QA over SEC filings, streamed live.
            </p>
            <ol className="space-y-4 mb-8">
              <li className="flex gap-3">
                <span className="size-7 rounded-full bg-accent text-textMain shrink-0 flex items-center justify-center font-semibold">
                  1
                </span>
                <div>
                  <div className="font-semibold">Pick a report</div>
                  <div className="text-sm text-textMuted">
                    Every conversation is anchored to one report (a single page from a 10-K /
                    10-Q).
                  </div>
                </div>
              </li>
              <li className="flex gap-3">
                <span className="size-7 rounded-full bg-accent text-textMain shrink-0 flex items-center justify-center font-semibold">
                  2
                </span>
                <div>
                  <div className="font-semibold">Ask a question</div>
                  <div className="text-sm text-textMuted">
                    Type freely or click one of the gold questions for that report. Answers
                    stream stage-by-stage.
                  </div>
                </div>
              </li>
              <li className="flex gap-3">
                <span className="size-7 rounded-full bg-accent text-textMain shrink-0 flex items-center justify-center font-semibold">
                  3
                </span>
                <div>
                  <div className="font-semibold">Run all gold</div>
                  <div className="text-sm text-textMuted">
                    Kick off the full evaluation set for the report and watch ✓/✗ accumulate.
                  </div>
                </div>
              </li>
            </ol>
            <button
              type="button"
              onClick={openPicker}
              className="px-4 py-2 rounded-md bg-accent2 text-bg font-semibold hover:opacity-90"
              data-testid="landing-cta"
            >
              + New conversation
            </button>
            <p className="text-xs text-textMuted mt-4">
              Your conversation list will live in the sidebar on the left.
            </p>
          </>
        ) : (
          <>
            <h1 className="text-2xl font-bold mb-2 text-textMain">Pick a conversation</h1>
            <p className="text-textMuted mb-6">
              Choose one from the sidebar, or start a new one.
            </p>
            <button
              type="button"
              onClick={openPicker}
              className="px-4 py-2 rounded-md bg-accent2 text-bg font-semibold hover:opacity-90"
              data-testid="landing-cta"
            >
              + New conversation
            </button>
          </>
        )}
      </div>
    </div>
  );
}
