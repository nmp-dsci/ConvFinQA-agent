import type { ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { PageHead } from '@/kit/ui';
import { NO_VALUE, formatFilingId, formatLatency, formatUsd } from '../landing/format';
import { STAGES, getEvalTrace, getLiveTrace } from './api';
import type { EvalTraceDetail, LiveTraceDetail, StageCapture, StageName } from './api';
import { bundleLine, formatCount, formatStamp, relativeTime, versionLabel } from './lib';
import {
  EmptyState,
  ErrorNote,
  Lamp,
  LampRow,
  LoadingRows,
  Panel,
  StatCells,
  TwoUp,
  Verdict,
} from './ui';
import { ak } from './useAdminData';
import './runs.css';

/**
 * One turn, stage by stage.
 *
 * Two kinds of turn open in this viewer. A live one comes out of the trace
 * store with timing, tokens and a cost; a scored one is reconstructed from a
 * committed predictions CSV and has the same four stage captures but never had
 * a clock on it. The route distinguishes them by `/runs/traces/eval` plus
 * query parameters, because a report id contains slashes and cannot survive a
 * single dynamic segment.
 *
 * The difference is rendered, not smoothed over: an eval turn shows no latency
 * card at all rather than a zero, because it was never measured.
 */

// ---------------------------------------------------------------------------
// Stage rendering
// ---------------------------------------------------------------------------

function Json({ value, label, className }: { value: unknown; label?: ReactNode; className?: string }) {
  if (value === null || value === undefined) {
    return (
      <div className={className}>
        {label && <p className="label">{label}</p>}
        <span className="muted">— nothing recorded</span>
      </div>
    );
  }
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  return (
    <div className={`code trace-json${className ? ` ${className}` : ''}`}>
      {label && <p className="label">{label}</p>}
      <pre>{text}</pre>
    </div>
  );
}

function StageCard({ stage, capture }: { stage: StageName; capture: StageCapture | null }) {
  const skipped = !capture;
  const metrics = capture?.metrics;

  return (
    <section
      data-testid={`stage-${stage}`}
      data-skipped={skipped ? 'true' : 'false'}
      className={`card stage-card${skipped ? ' skipped' : ''}`}
    >
      <div className="head">
        <h3>{stage}</h3>
        {skipped ? (
          <span className="status no small">skipped — the router did not need this stage</span>
        ) : (
          <span className="num small muted">
            {metrics?.latency_ms !== undefined ? formatLatency(metrics.latency_ms) : NO_VALUE} ·{' '}
            {metrics?.total_tokens !== undefined ? formatCount(metrics.total_tokens) : NO_VALUE} tok
          </span>
        )}
      </div>

      {!skipped && (
        <div className="stage-grid">
          <Json label="input" value={capture.input} />
          <Json label="output" value={capture.output} />
          {capture.reasoning && <Json label="reasoning" value={capture.reasoning} className="span2" />}
          {capture.trajectory && capture.trajectory.length > 0 && (
            <Json
              label={`tool loop · ${capture.trajectory.length} steps`}
              value={capture.trajectory}
              className="span2"
            />
          )}
        </div>
      )}
    </section>
  );
}

function Stages({ capture, endpoint }: { capture: Partial<Record<StageName, StageCapture | null>> | null | undefined; endpoint: string }) {
  const ran = STAGES.filter((stage) => capture?.[stage]).length;
  return (
    <Panel
      title={`Stages — ${ran} of ${STAGES.length} ran for this turn`}
      endpoint={endpoint}
      note="what each agent was given and what it returned; a dashed card is a stage the router skipped"
    >
      <div className="stages">
        {STAGES.map((stage) => (
          <StageCard key={stage} stage={stage} capture={capture?.[stage] ?? null} />
        ))}
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------

/** The question, then what was answered beside gold — the verdict is the page's claim. */
function Header({
  reportId,
  question,
  answer,
  gold,
  correct,
}: {
  reportId: string;
  question: string;
  answer: string;
  gold: string;
  correct: boolean | null;
}) {
  return (
    <section>
      <div className="question">{question}</div>
      <TwoUp>
        <div className={`answer ${correct === null ? '' : correct ? 'ok' : 'no'}`}>
          <p className="label">answered</p>
          <pre className="wrap-any">{answer || NO_VALUE}</pre>
          <p className="reason">
            <Verdict ok={correct}>
              {correct === null ? 'no gold answer' : correct ? 'correct' : 'incorrect'}
            </Verdict>
          </p>
        </div>
        <div className="answer">
          <p className="label">gold</p>
          <pre className="wrap-any">{gold || NO_VALUE}</pre>
        </div>
      </TwoUp>
      <p className="small muted wrap-any">
        <span className="mono">{reportId}</span>
      </p>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Live turn
// ---------------------------------------------------------------------------

function LiveTraceView({ trace }: { trace: LiveTraceDetail }) {
  const correct = trace.correct === null ? null : Boolean(trace.correct);
  const historyText = trace.capture?.history_text;
  // `total_tokens` is 0 on some rows that do carry input/output counts; the claim uses whichever is real.
  const tokens =
    trace.total_tokens ||
    (trace.input_tokens != null || trace.output_tokens != null
      ? (trace.input_tokens ?? 0) + (trace.output_tokens ?? 0)
      : null);
  const metered = trace.cost_usd != null || tokens != null;

  return (
    <>
      <LampRow>
        <Lamp
          label="source"
          value={trace.source}
          tone={trace.source === 'demo' ? 'amber' : 'good'}
          dashed={trace.source === 'demo'}
          title={
            trace.source === 'demo'
              ? 'Replayed from the recorded pack. Its timing is replay timing, not latency.'
              : 'Answered live by the model.'
          }
        />
        <Lamp
          label="when"
          value={relativeTime(trace.created_at)}
          tone="idle"
          dashed
          title={formatStamp(trace.created_at)}
        />
        {trace.session_id && (
          <Lamp
            label="session"
            value={trace.session_id.slice(0, 8)}
            tone="info"
            to={`/runs/traces?session=${trace.session_id}`}
            title={trace.session_id}
          />
        )}
        {trace.error_code && <Lamp label="error" value={trace.error_code} tone="bad" />}
      </LampRow>

      <Header
        reportId={trace.report_id}
        question={trace.question}
        answer={trace.answer ?? ''}
        gold={trace.gold_answer ?? ''}
        correct={correct}
      />

      {trace.error && (
        <div className="callout err">
          <span className="ttl">✕ {trace.error_code ?? 'error'}</span>
          <span className="wrap-any">{trace.error}</span>
          {trace.error_code && (
            <p className="small muted">
              classified as <code>{trace.error_code}</code> in{' '}
              <code>src/convfinqa/error_codes.py</code>
            </p>
          )}
        </div>
      )}

      <Panel
        title={
          metered
            ? `Cost of this turn — ${formatUsd(trace.cost_usd)} over ${formatCount(tokens)} tokens`
            : 'Cost of this turn — never metered'
        }
        endpoint={`/traces/${trace.trace_id}`}
      >
        <StatCells
          columns={3}
          cells={[
            {
              label: 'latency',
              value: formatLatency(trace.latency_ms),
              reason: 'this turn was never metered',
            },
            {
              label: 'tokens',
              value: formatCount(trace.total_tokens),
              reason: 'this turn was never metered',
            },
            {
              label: 'cost',
              value: formatUsd(trace.cost_usd),
              reason: 'this turn was never metered',
            },
          ]}
        />
        <div className="runs-gap">
          <StatCells
            columns={2}
            cells={[
              {
                label: 'input tokens',
                value: formatCount(trace.input_tokens),
                reason: 'not recorded for this turn',
              },
              {
                label: 'output tokens',
                value: formatCount(trace.output_tokens),
                reason: 'not recorded for this turn',
              },
            ]}
          />
        </div>
        {trace.program && trace.program !== 'nan' && <Json label="program" value={trace.program} />}
        <p className="small muted wrap-any">
          bundle <span className="mono">{trace.bundle_id ?? NO_VALUE}</span> · {bundleLine(trace.bundle)}
        </p>
      </Panel>

      <Stages capture={trace.capture} endpoint="capture" />

      {historyText && (
        <Panel
          title="Conversation history the agents saw"
          note="the resolved prior turns this answer depended on"
        >
          <Json value={historyText} />
        </Panel>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Scored eval turn
// ---------------------------------------------------------------------------

function EvalTraceView({ trace }: { trace: EvalTraceDetail }) {
  const hasProgram = Boolean(trace.program && trace.program !== 'nan');
  return (
    <>
      <LampRow>
        <Lamp
          label="source"
          value="eval"
          tone="info"
          dashed
          title="Reconstructed from a committed predictions CSV, not recorded live."
        />
        <Lamp
          label="version"
          value={trace.version}
          tone="info"
          to={`/runs/legacy?version=${trace.version}`}
        />
      </LampRow>

      <Header
        reportId={trace.report_id}
        question={trace.question}
        answer={trace.answer}
        gold={trace.gold_answer}
        correct={trace.correct}
      />

      <Panel
        title={hasProgram ? 'Program — what the calculator executed' : 'Program — none recorded'}
        endpoint={`/traces/eval/${trace.version}`}
        note="a scored turn has no clock on it — this run was never metered, so there is no latency or cost to show"
      >
        {/*
          The CSV writes a pandas `nan` for a turn that produced no program.
          Printing that string would present a missing value as a computed one.
        */}
        {hasProgram ? (
          <Json value={trace.program} />
        ) : (
          <p className="small muted">
            No program recorded for this turn — it was answered as a direct lookup, or the run
            predates program capture.
          </p>
        )}
      </Panel>

      <Stages capture={trace.capture} endpoint="capture" />

      {trace.history_text && (
        <Panel title="Conversation history the agents saw">
          <Json value={trace.history_text} />
        </Panel>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------

/** The page's claim: the verdict, on which turn, and what was answered against gold. */
function turnTitle(t: {
  correct: boolean | null;
  report_id: string;
  turn_index: number;
  answer: string | null;
  gold_answer: string | null;
}): ReactNode {
  const word = t.correct === null ? 'Unscored' : t.correct ? 'Correct' : 'Incorrect';
  return (
    <>
      <em>{word}</em> — {formatFilingId(t.report_id)}, turn {t.turn_index}: answered{' '}
      {t.answer || NO_VALUE} against gold {t.gold_answer || NO_VALUE}
    </>
  );
}

export default function TraceDetail() {
  const { traceId = '' } = useParams();
  const [params] = useSearchParams();

  const isEval = traceId === 'eval';
  const version = params.get('version') ?? '';
  const reportId = params.get('report_id') ?? '';
  const turnIndex = Number(params.get('turn_index') ?? '0');

  const live = useQuery({
    queryKey: ak.liveTrace(traceId),
    queryFn: () => getLiveTrace(traceId),
    enabled: !isEval && Boolean(traceId),
  });

  const evalTrace = useQuery({
    queryKey: ak.evalTrace(version, reportId, turnIndex),
    queryFn: () => getEvalTrace(version, reportId, turnIndex),
    enabled: isEval && Boolean(version && reportId),
  });

  const query = isEval ? evalTrace : live;

  const title =
    isEval && evalTrace.data
      ? turnTitle(evalTrace.data)
      : !isEval && live.data
        ? turnTitle({ ...live.data, correct: live.data.correct === null ? null : Boolean(live.data.correct) })
        : (
            <>
              One turn, <em>stage by stage</em>
            </>
          );

  const crumbs = isEval ? (
    <>
      <Link to="/runs">runs</Link> / <Link to={`/runs/legacy?version=${encodeURIComponent(version)}`}>{versionLabel(version) || 'scored turn'}</Link> /{' '}
      <span className="mono">
        {reportId ? `${formatFilingId(reportId)} · turn ${turnIndex}` : 'scored turn'}
      </span>
    </>
  ) : (
    <>
      <Link to="/runs">runs</Link> / <Link to="/runs/traces">served turns</Link> /{' '}
      <span className="mono">{traceId}</span>
    </>
  );

  return (
    <div className="stack" data-testid="admin-trace-detail">
      <div>
        <nav className="crumbs" aria-label="Breadcrumb">
          {crumbs}
        </nav>
        <PageHead
          title={title}
          lead={
            isEval
              ? 'A scored turn replayed out of a committed predictions CSV — the same four stage captures a live turn records, from a run that happened months ago.'
              : 'Every stage of one answered turn: what each agent was given, what it returned, and what it cost.'
          }
        />
      </div>
      {isEval && !(version && reportId) ? (
        <EmptyState>
          A scored turn needs <code>version</code>, <code>report_id</code> and{' '}
          <code>turn_index</code> in the query string. Open one from the answers table.
        </EmptyState>
      ) : query.isLoading ? (
        <LoadingRows rows={8} />
      ) : query.error ? (
        <ErrorNote error={query.error} />
      ) : isEval && evalTrace.data ? (
        <EvalTraceView trace={evalTrace.data} />
      ) : !isEval && live.data ? (
        <LiveTraceView trace={live.data} />
      ) : (
        <EmptyState>No such turn.</EmptyState>
      )}
    </div>
  );
}
