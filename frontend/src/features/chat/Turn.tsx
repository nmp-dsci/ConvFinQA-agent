import { AlertTriangle, Check, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { looseNumericMatch } from '../../numericMatch';
import type { Message } from '../../types';
import { errorCopy } from './errors';
import { fmtMs } from './format';
import { stageViews, totalLatency } from './stages';

/**
 * The banner that says the demo answered a different question than the one
 * typed.
 *
 * This is the single most load-bearing piece of copy in the chat. Without it a
 * paraphrase silently gets the nearest recorded answer and the visitor has no
 * way to know the words on screen are not the words they wrote.
 */
function MatchedBanner({ message }: { message: Message }) {
  if (!message.matchedQuestion) return null;
  return (
    <div
      data-testid="matched-banner"
      className="mb-2 rounded-md border border-amber-line bg-amber-soft px-2.5 py-2 text-[11px] leading-relaxed text-text"
    >
      <span className="mono-caps mr-1.5 text-amber">replayed</span>
      No recording for what you asked — playing the closest recorded question:{' '}
      <span className="font-medium">“{message.matchedQuestion}”</span>
      {typeof message.matchScore === 'number' && (
        <span className="ml-1 font-mono text-faint">(match {message.matchScore.toFixed(2)})</span>
      )}
    </div>
  );
}

function StageStrip({ message }: { message: Message }) {
  const views = stageViews(message);
  const latency = totalLatency(message);

  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-1.5 gap-y-1">
      {views.map((view) => (
        <span
          key={view.stage}
          data-stage={view.stage}
          data-state={view.state}
          title={view.detail ? `${view.stage} — ${view.detail}` : view.stage}
          className={cn(
            'rounded-[4px] border px-1.5 py-0.5 font-mono text-[10px] leading-tight',
            view.state === 'done' && 'border-line-2 bg-panel-2 text-muted',
            view.state === 'active' && 'border-amber-line bg-amber-soft text-amber',
            view.state === 'pending' && 'border-line bg-transparent text-faint',
            view.state === 'skipped' && 'border-line bg-transparent text-faint line-through'
          )}
        >
          {view.stage}
          {view.state === 'done' && view.detail ? (
            <span className="text-faint"> · {view.detail}</span>
          ) : null}
        </span>
      ))}
      <span className="ml-auto font-mono text-[10px] text-faint" title={
        latency === null
          ? 'No per-stage timings were recorded for this turn'
          : 'Sum of the measured per-stage latencies'
      }>
        {latency === null ? '— no timing' : fmtMs(latency)}
      </span>
    </div>
  );
}

function ErrorBlock({ message }: { message: Message }) {
  const copy = errorCopy(message.errorCode, message.errorText);
  return (
    <div className="rounded-md border border-bad/40 bg-bad/10 px-2.5 py-2">
      <div className="flex items-baseline gap-1.5">
        <AlertTriangle className="size-3.5 shrink-0 translate-y-0.5 text-bad" aria-hidden />
        <span className="text-[13px] font-medium text-text">{copy.title}</span>
        {message.errorCode && (
          <span className="ml-auto font-mono text-[10px] text-faint">{message.errorCode}</span>
        )}
      </div>
      <p className="mt-1 text-[11px] leading-relaxed text-muted">{copy.hint}</p>
      {message.errorText && message.errorText !== 'aborted' && (
        <p className="mt-1.5 font-mono text-[10.5px] leading-relaxed break-words text-faint">
          {message.errorText}
        </p>
      )}
    </div>
  );
}

interface Props {
  message: Message;
  selected: boolean;
  onSelect: (id: string) => void;
}

export function Turn({ message, selected, onSelect }: Props) {
  if (message.role === 'system') {
    return (
      <div data-role="system-message" className="my-2 text-center text-[11px] italic text-faint">
        {message.text}
      </div>
    );
  }

  if (message.role === 'user') {
    return (
      <div data-role="user-message" className="my-2 flex justify-end">
        <div className="max-w-[76%] rounded-lg rounded-br-sm bg-panel-2 px-3 py-2 text-[13px] leading-relaxed break-words whitespace-pre-wrap text-text">
          {message.text}
        </div>
      </div>
    );
  }

  const isStreaming = message.status === 'streaming';
  const isError = message.status === 'error';
  const goldVerdict =
    message.goldAnswer && message.text && message.status === 'done'
      ? looseNumericMatch(message.text, message.goldAnswer)
      : undefined;

  return (
    <div
      data-role="assistant-message"
      data-streaming={isStreaming ? 'true' : 'false'}
      data-final={message.status === 'done' ? 'true' : 'false'}
      data-gold={goldVerdict === undefined ? undefined : goldVerdict ? 'match' : 'mismatch'}
      data-selected={selected ? 'true' : 'false'}
      className="my-2 flex justify-start"
    >
      <button
        type="button"
        onClick={() => onSelect(message.id)}
        aria-pressed={selected}
        title="Show this turn in the trace inspector"
        className={cn(
          'w-full max-w-[92%] cursor-pointer rounded-lg border px-3 py-2.5 text-left transition-colors',
          selected ? 'border-amber-line bg-panel-2/50' : 'border-line hover:border-line-2'
        )}
      >
        <MatchedBanner message={message} />

        {isError ? (
          <ErrorBlock message={message} />
        ) : message.text ? (
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="font-mono text-2xl leading-none font-medium break-all text-text">
              {message.text}
            </span>
            {message.goldAnswer && message.status === 'done' && (
              <span
                className={cn(
                  'inline-flex items-baseline gap-1 font-mono text-[11px]',
                  goldVerdict ? 'text-good' : 'text-bad'
                )}
                title={
                  goldVerdict
                    ? `Matches the dataset gold answer (${message.goldAnswer})`
                    : `The dataset gold answer is ${message.goldAnswer}`
                }
              >
                {goldVerdict ? (
                  <Check className="size-3 self-center" aria-hidden />
                ) : (
                  <X className="size-3 self-center" aria-hidden />
                )}
                gold {message.goldAnswer}
              </span>
            )}
          </div>
        ) : (
          <div className="font-mono text-[13px] text-faint">
            <span className="animate-pulse">answering…</span>
          </div>
        )}

        {message.goldProgram && message.status === 'done' && (
          <div
            className="mt-1 font-mono text-[10.5px] break-all text-faint"
            title="The dataset's own program. It re-derives from raw values while the pipeline reuses prior answers, so a shorter program with the same answer is not a wrong one."
          >
            gold program {message.goldProgram}
          </div>
        )}

        <StageStrip message={message} />
      </button>
    </div>
  );
}
