import { looseNumericMatch } from '../../numericMatch';
import type { Message } from '../../types';
import { errorCopy } from './errors';
import { fmtMs } from './format';
import { howLine } from './how';
import { type StageState, stageViews, totalLatency } from './stages';

/** Status is never colour alone: each stage state carries its glyph. */
const STAGE_GLYPH: Record<StageState, string> = {
  done: '●',
  active: '◐',
  pending: '○',
  skipped: '–',
};

const STAGE_CHIP: Record<StageState, string> = {
  done: 'chip',
  active: 'chip acc',
  pending: 'chip no',
  skipped: 'chip no skip',
};

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
    <div data-testid="matched-banner" className="callout">
      <b className="ch-kicker">◐ replayed</b>
      No recording for what you asked — playing the closest recorded question:{' '}
      <b>“{message.matchedQuestion}”</b>
      {typeof message.matchScore === 'number' && (
        <span className="ch-meta"> (match {message.matchScore.toFixed(2)})</span>
      )}
    </div>
  );
}

function StageStrip({ message }: { message: Message }) {
  const views = stageViews(message);
  const latency = totalLatency(message);

  return (
    <div className="ch-stages">
      {views.map((view) => (
        <span
          key={view.stage}
          data-stage={view.stage}
          data-state={view.state}
          title={view.detail ? `${view.stage} — ${view.detail}` : `${view.stage} — ${view.state}`}
          className={STAGE_CHIP[view.state]}
        >
          <span aria-hidden>{STAGE_GLYPH[view.state]}</span>
          {view.stage}
          {view.state === 'done' && view.detail ? <span className="d"> · {view.detail}</span> : null}
          {view.state === 'active' ? <span className="d"> · running</span> : null}
        </span>
      ))}
      <span
        className="ch-meta ch-push"
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
  return (
    <span
      data-testid="judge-badge"
      data-band={verdict.band}
      className={low ? 'tag warn' : 'tag ok'}
      title={`Confidence judge (${verdict.version ?? 'judge'}): ${verdict.reason}`}
    >
      <span aria-hidden>{low ? '◐ ' : '● '}</span>
      judge {verdict.band} · p {verdict.p_correct.toFixed(2)}
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
    <div data-testid="judge-caution" className="callout">
      <p>
        <b>◐ Check this one.</b> The confidence judge could not verify it against the
        filing.{verdict.reason ? ` ${verdict.reason}` : ''}
      </p>
      {failed.length > 0 && (
        <p className="ch-meta">{failed.map(([name, v]) => `${name}: ${v}`).join(' · ')}</p>
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
    <div data-testid="withheld-block" data-band="low" className="callout">
      <b className="ttl">
        ◐ I'm not confident in this one
        {verdict && <span className="ch-meta"> · judge low · p {verdict.p_correct.toFixed(2)}</span>}
      </b>
      <p>
        The answer was computed but withheld: the confidence judge could not verify it against
        the filing.{verdict?.reason ? ` ${verdict.reason}` : ''}
      </p>
      {failed.length > 0 && (
        <p className="ch-meta">{failed.map(([name, v]) => `${name}: ${v}`).join(' · ')}</p>
      )}
    </div>
  );
}

function ErrorBlock({ message }: { message: Message }) {
  const copy = errorCopy(message.errorCode, message.errorText);
  return (
    <div className="callout err">
      <b className="ttl">
        ✕ {copy.title}
        {message.errorCode && <span className="ch-meta"> · {message.errorCode}</span>}
      </b>
      <p>{copy.hint}</p>
      {message.errorText && message.errorText !== 'aborted' && (
        <p className="ch-meta">{message.errorText}</p>
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
      <p data-role="system-message" className="ch-sys">
        {message.text}
      </p>
    );
  }

  if (message.role === 'user') {
    return (
      <div data-role="user-message" className="ch-user">
        <div className="ch-q">{message.text}</div>
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
      className="ch-a"
    >
      <button
        type="button"
        onClick={() => onSelect(message.id)}
        aria-pressed={selected}
        title="Show this turn in the trace inspector"
        className={selected ? 'ch-turn on' : 'ch-turn'}
      >
        <MatchedBanner message={message} />

        {isError ? (
          <ErrorBlock message={message} />
        ) : message.withheld && message.status === 'done' ? (
          <WithheldBlock message={message} />
        ) : message.text ? (
          <span className="ch-ans-row">
            <span className="ch-ans">{message.text}</span>
            {message.goldAnswer && message.status === 'done' && (
              <span
                className={goldVerdict ? 'ch-gold v-ok' : 'ch-gold v-fail'}
                title={
                  goldVerdict
                    ? `Matches the dataset gold answer (${message.goldAnswer})`
                    : `The dataset gold answer is ${message.goldAnswer}`
                }
              >
                {goldVerdict ? '✓ matches gold' : `✗ gold ${message.goldAnswer}`}
              </span>
            )}
            <span className="ch-push">
              <JudgeBadge message={message} />
            </span>
          </span>
        ) : (
          <span className="ch-pending ch-pulse">answering…</span>
        )}

        {how && (
          <p data-testid="turn-how" className="ch-how">
            {how}
          </p>
        )}

        {message.status === 'done' && <CautionNote message={message} />}

        {message.goldProgram && message.status === 'done' && (
          <p
            className="ch-meta"
            title="The dataset's own program. It re-derives from raw values while the pipeline reuses prior answers, so a shorter program with the same answer is not a wrong one."
          >
            gold program {message.goldProgram}
          </p>
        )}

        <StageStrip message={message} />
      </button>
    </div>
  );
}
