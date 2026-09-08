import { AlertTriangle, Check, ShieldAlert, ShieldCheck, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { looseNumericMatch } from '../../numericMatch';
import type { Message } from '../../types';
import { errorCopy } from './errors';
import { fmtMs } from './format';
import { howLine } from './how';
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
      className="type-small mb-2 rounded-md border border-amber-line bg-amber-soft px-2.5 py-2 text-text"
    >
      <span className="mono-caps mr-1.5 text-amber">replayed</span>
      No recording for what you asked — playing the closest recorded question:{' '}
      <span className="font-medium">“{message.matchedQuestion}”</span>
      {typeof message.matchScore === 'number' && (
        <span className="type-num ml-1 text-faint">(match {message.matchScore.toFixed(2)})</span>
      )}
    </div>
  );
}

function StageStrip({ message }: { message: Message }) {
  const views = stageViews(message);
  const latency = totalLatency(message);

  return (
    <div className="mt-2.5 flex flex-wrap items-center gap-x-1.5 gap-y-1">
      {views.map((view) => (
        <span
          key={view.stage}
          data-stage={view.stage}
          data-state={view.state}
          title={view.detail ? `${view.stage} — ${view.detail}` : view.stage}
          className={cn(
            'type-num type-meta rounded-[4px] border px-1.5 py-0.5 leading-tight',
            view.state === 'done' && 'border-line-2 bg-panel-2 text-muted',
            view.state === 'active' && 'border-amber-line bg-amber-soft text-amber',
            view.state === 'pending' && 'border-line bg-transparent text-faint',
            view.state === 'skipped' && 'border-line bg-transparent text-faint line-through',
          )}
        >
          {view.stage}
          {view.state === 'done' && view.detail ? (
            <span className="text-faint"> · {view.detail}</span>
          ) : null}
        </span>
      ))}
      <span
        className="type-num type-meta ml-auto text-faint"
        title={
          latency === null
            ? 'No per-stage timings were recorded for this turn'
            : 'Sum of the measured per-stage latencies'
        }
      >
        {latency === null ? '— no timing' : fmtMs(latency)}
      </span>
    </div>
  );
}

/**
 * s12: the confidence judge's band beside a released answer.
 *
 * A `high` band is the judge saying the trace checks out; the probability is
 * its own estimate, shown so a reader can see how close to the line it was.
 * s13: a low band is advisory by default, so the answer is shown and the
 * badge has to carry the caution rather than being the reward for a pass.
 */
function JudgeBadge({ message }: { message: Message }) {
  const verdict = message.judge;
  if (!verdict || message.withheld) return null;
  const low = verdict.band === 'low';
  const Icon = low ? ShieldAlert : ShieldCheck;
  return (
    <span
      data-testid="judge-badge"
      data-band={verdict.band}
      className={cn(
        'type-num type-meta inline-flex items-center gap-1 rounded-[4px] border px-1.5 py-0.5',
        low ? 'border-amber-line bg-amber-soft text-amber' : 'border-good/40 bg-good/10 text-good',
      )}
      title={`Confidence judge (${verdict.version ?? 'judge'}): ${verdict.reason}`}
    >
      <Icon className="size-3" aria-hidden />
      {verdict.band} · p {verdict.p_correct.toFixed(2)}
    </span>
  );
}

/**
 * s13: the judge's caution, shown beside an answer it could not verify.
 *
 * The default policy is advisory, not abstention: withholding was measured
 * and destroyed four correct answers for every wrong one it stopped, for a
 * released-set accuracy indistinguishable from showing everything. So the
 * answer stands and the doubt is stated — the reason and the checks that
 * failed, which is the part a reader can actually act on.
 */
function CautionNote({ message }: { message: Message }) {
  const verdict = message.judge;
  if (!verdict || verdict.band !== 'low' || message.withheld) return null;
  const failed = Object.entries(verdict.checks ?? {}).filter(([, v]) => v !== 'pass');
  return (
    <div
      data-testid="judge-caution"
      className="mt-2 rounded-md border border-amber-line bg-amber-soft px-2.5 py-2"
    >
      <p className="type-small text-muted">
        <span className="font-medium text-text">Check this one.</span> The confidence judge could
        not verify it against the filing.{verdict.reason ? ` ${verdict.reason}` : ''}
      </p>
      {failed.length > 0 && (
        <p className="type-num type-meta mt-1 text-faint">
          {failed.map(([name, v]) => `${name}: ${v}`).join(' · ')}
        </p>
      )}
    </div>
  );
}

/**
 * s12: what the visitor sees when the judge withheld the answer (gate mode).
 *
 * The number exists — the session produced it and the trace keeps it — but the
 * judge could not verify it against the filing, so the product says so rather
 * than showing a figure it cannot stand behind.
 */
function WithheldBlock({ message }: { message: Message }) {
  const verdict = message.judge;
  const failed = verdict
    ? Object.entries(verdict.checks ?? {}).filter(([, v]) => v !== 'pass')
    : [];
  return (
    <div
      data-testid="withheld-block"
      data-band="low"
      className="rounded-md border border-amber-line bg-amber-soft px-2.5 py-2"
    >
      <div className="flex items-baseline gap-1.5">
        <ShieldAlert className="size-3.5 shrink-0 translate-y-0.5 text-amber" aria-hidden />
        <span className="type-body font-medium text-text">I'm not confident in this one</span>
        {verdict && (
          <span className="type-num type-meta ml-auto text-faint">
            low · p {verdict.p_correct.toFixed(2)}
          </span>
        )}
      </div>
      <p className="type-small mt-1 text-muted">
        The answer was computed but withheld: the confidence judge could not verify it against
        the filing.{verdict?.reason ? ` ${verdict.reason}` : ''}
      </p>
      {failed.length > 0 && (
        <p className="type-num type-meta mt-1 text-faint">
          {failed.map(([name, v]) => `${name}: ${v}`).join(' · ')}
        </p>
      )}
    </div>
  );
}

function ErrorBlock({ message }: { message: Message }) {
  const copy = errorCopy(message.errorCode, message.errorText);
  return (
    <div className="rounded-md border border-bad/40 bg-bad/10 px-2.5 py-2">
      <div className="flex items-baseline gap-1.5">
        <AlertTriangle className="size-3.5 shrink-0 translate-y-0.5 text-bad" aria-hidden />
        <span className="type-body font-medium text-text">{copy.title}</span>
        {message.errorCode && (
          <span className="type-num type-meta ml-auto text-faint">{message.errorCode}</span>
        )}
      </div>
      <p className="type-small mt-1 text-muted">{copy.hint}</p>
      {message.errorText && message.errorText !== 'aborted' && (
        <p className="type-num type-meta mt-1.5 break-words text-faint">{message.errorText}</p>
      )}
    </div>
  );
}

interface Props {
  message: Message;
  selected: boolean;
  onSelect: (id: string) => void;
}

/**
 * One turn. The hierarchy, top to bottom: the question (body), the answer
 * (the HUD step — it is the one number the reader came for) with the gold
 * verdict and the judge's band beside it, the trace in one sentence, then the
 * four stage chips with the measured latency. Nothing on the card is under
 * the meta step.
 */
export function Turn({ message, selected, onSelect }: Props) {
  if (message.role === 'system') {
    return (
      <div data-role="system-message" className="type-small my-2 text-center text-faint italic">
        {message.text}
      </div>
    );
  }

  if (message.role === 'user') {
    return (
      <div data-role="user-message" className="my-2.5 flex justify-end">
        <div className="type-body max-w-[76%] rounded-lg rounded-br-sm bg-panel-2 px-3 py-2 break-words whitespace-pre-wrap text-text">
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
  const how = message.status === 'done' && !isError ? howLine(message) : '';

  return (
    <div
      data-role="assistant-message"
      data-streaming={isStreaming ? 'true' : 'false'}
      data-final={message.status === 'done' ? 'true' : 'false'}
      data-gold={goldVerdict === undefined ? undefined : goldVerdict ? 'match' : 'mismatch'}
      data-selected={selected ? 'true' : 'false'}
      className="my-2.5 flex justify-start"
    >
      <button
        type="button"
        onClick={() => onSelect(message.id)}
        aria-pressed={selected}
        title="Show this turn in the trace inspector"
        className={cn(
          'w-full max-w-[92%] cursor-pointer rounded-lg border px-3.5 py-3 text-left transition-colors',
          selected ? 'border-amber-line bg-panel-2/50' : 'border-line hover:border-line-2',
        )}
      >
        <MatchedBanner message={message} />

        {isError ? (
          <ErrorBlock message={message} />
        ) : message.withheld && message.status === 'done' ? (
          <WithheldBlock message={message} />
        ) : message.text ? (
          <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <span className="type-hud break-all">{message.text}</span>
            {message.goldAnswer && message.status === 'done' && (
              <span
                className={cn(
                  'type-num type-small inline-flex items-baseline gap-1',
                  goldVerdict ? 'text-good' : 'text-bad',
                )}
                title={
                  goldVerdict
                    ? `Matches the dataset gold answer (${message.goldAnswer})`
                    : `The dataset gold answer is ${message.goldAnswer}`
                }
              >
                {goldVerdict ? (
                  <Check className="size-3.5 self-center" aria-hidden />
                ) : (
                  <X className="size-3.5 self-center" aria-hidden />
                )}
                gold {message.goldAnswer}
              </span>
            )}
            <span className="ml-auto">
              <JudgeBadge message={message} />
            </span>
          </div>
        ) : (
          <div className="type-num type-body text-faint">
            <span className="animate-pulse">answering…</span>
          </div>
        )}

        {how && (
          <p data-testid="turn-how" className="type-small mt-2 text-muted">
            {how}
          </p>
        )}

        {message.status === 'done' && <CautionNote message={message} />}

        {message.goldProgram && message.status === 'done' && (
          <div
            className="type-num type-meta mt-1.5 break-all text-faint"
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
