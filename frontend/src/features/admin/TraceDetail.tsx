import { useQuery } from '@tanstack/react-query';
import { ArrowLeft } from 'lucide-react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { NO_VALUE, formatFilingId, formatLatency, formatUsd } from '../landing/format';
import { STAGES, getEvalTrace, getLiveTrace } from './api';
import type { EvalTraceDetail, LiveTraceDetail, StageCapture, StageName } from './api';
import { bundleLine, formatCount, formatStamp, relativeTime } from './lib';
import {
  AdminPage,
  EmptyState,
  ErrorNote,
  Lamp,
  LampRow,
  LoadingRows,
  Panel,
  StatCells,
  Verdict,
} from './ui';
import { ak } from './useAdminData';

/**
 * One turn, stage by stage.
 *
 * Two kinds of turn open in this viewer. A live one comes out of the trace
 * store with timing, tokens and a cost; a scored one is reconstructed from a
 * committed predictions CSV and has the same four stage captures but never had
 * a clock on it. The route distinguishes them by `/admin/traces/eval` plus
 * query parameters, because a report id contains slashes and cannot survive a
 * single dynamic segment.
 *
 * The difference is rendered, not smoothed over: an eval turn shows no latency
 * card at all rather than a zero, because it was never measured.
 */

// ---------------------------------------------------------------------------
// Stage rendering
// ---------------------------------------------------------------------------

function Json({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="text-faint">—</span>;
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  return (
    <pre className="mt-1 max-h-64 overflow-auto rounded-[4px] border border-line bg-ground px-2 py-1.5 font-mono text-[10.5px] leading-relaxed break-words whitespace-pre-wrap text-muted">
      {text}
    </pre>
  );
}

function StageCard({ stage, capture }: { stage: StageName; capture: StageCapture | null }) {
  const skipped = !capture;
  const metrics = capture?.metrics;

  return (
    <div
      data-testid={`stage-${stage}`}
      data-skipped={skipped ? 'true' : 'false'}
      className={cn(
        'min-w-0 rounded-md border p-3',
        skipped ? 'border-line border-dashed bg-panel/40' : 'border-line bg-panel',
      )}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className={cn('type-body font-medium', skipped ? 'text-faint' : 'text-text')}>
          {stage}
        </span>
        {skipped ? (
          <span className="mono-caps">skipped — the router did not need this stage</span>
        ) : (
          <span className="type-num text-[11px] text-muted">
            {metrics?.latency_ms !== undefined ? formatLatency(metrics.latency_ms) : NO_VALUE} ·{' '}
            {metrics?.total_tokens !== undefined ? formatCount(metrics.total_tokens) : NO_VALUE} tok
          </span>
        )}
      </div>

      {!skipped && (
        <div className="mt-2 grid grid-cols-1 gap-2 lg:grid-cols-2">
          <div className="min-w-0">
            <div className="mono-caps">input</div>
            <Json value={capture.input} />
          </div>
          <div className="min-w-0">
            <div className="mono-caps">output</div>
            <Json value={capture.output} />
          </div>
          {capture.reasoning && (
            <div className="min-w-0 lg:col-span-2">
              <div className="mono-caps">reasoning</div>
              <Json value={capture.reasoning} />
            </div>
          )}
          {capture.trajectory && capture.trajectory.length > 0 && (
            <div className="min-w-0 lg:col-span-2">
              <div className="mono-caps">tool loop · {capture.trajectory.length} steps</div>
              <Json value={capture.trajectory} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function Header({
  reportId,
  turnIndex,
  question,
  answer,
  gold,
  correct,
  backTo,
}: {
  reportId: string;
  turnIndex: number;
  question: string;
  answer: string;
  gold: string;
  correct: boolean | null;
  backTo: string;
}) {
  return (
    <Panel title={formatFilingId(reportId)} endpoint={`turn ${turnIndex}`}>
      <p className="type-lede text-text">{question}</p>
      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2">
        <span className="flex items-baseline gap-2">
          <span className="mono-caps">answered</span>
          <span className="type-num text-[15px] text-text">{answer || NO_VALUE}</span>
        </span>
        <span className="flex items-baseline gap-2">
          <span className="mono-caps">gold</span>
          <span className="type-num text-[15px] text-muted">{gold || NO_VALUE}</span>
        </span>
        <Verdict ok={correct}>
          {correct === null ? 'no gold answer' : correct ? 'correct' : 'incorrect'}
        </Verdict>
        <Link
          to={backTo}
          className="type-small ml-auto inline-flex items-center gap-1 text-faint hover:text-amber"
        >
          <ArrowLeft aria-hidden className="size-3" />
          back to turns
        </Link>
      </div>
      <p className="type-meta mt-2 break-all text-faint">{reportId}</p>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Live turn
// ---------------------------------------------------------------------------

function LiveTraceView({ trace }: { trace: LiveTraceDetail }) {
  const correct = trace.correct === null ? null : Boolean(trace.correct);
  const historyText = trace.capture?.history_text;

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
        <Lamp label="when" value={relativeTime(trace.created_at)} tone="idle" dashed
          title={formatStamp(trace.created_at)} />
        {trace.session_id && (
          <Lamp
            label="session"
            value={trace.session_id.slice(0, 8)}
            tone="info"
            to={`/admin/traces?session=${trace.session_id}`}
            title={trace.session_id}
          />
        )}
        {trace.error_code && <Lamp label="error" value={trace.error_code} tone="bad" />}
      </LampRow>

      <Header
        reportId={trace.report_id}
        turnIndex={trace.turn_index}
        question={trace.question}
        answer={trace.answer ?? ''}
        gold={trace.gold_answer ?? ''}
        correct={correct}
        backTo="/admin/traces"
      />

      <Panel title="Cost of this turn" endpoint={`/traces/${trace.trace_id}`}>
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
        <div className="mt-1.5">
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
        {trace.program && trace.program !== 'nan' && (
          <div className="mt-2">
            <div className="mono-caps">program</div>
            <Json value={trace.program} />
          </div>
        )}
        <p className="type-meta mt-2 break-all text-faint">
          bundle {trace.bundle_id ?? NO_VALUE} · {bundleLine(trace.bundle)}
        </p>
      </Panel>

      {trace.error && (
        <Panel title="Error">
          <p className="type-small text-bad">{trace.error}</p>
          {trace.error_code && (
            <p className="type-meta mt-1 text-faint">
              classified as <code>{trace.error_code}</code> in{' '}
              <code>src/convfinqa/error_codes.py</code>
            </p>
          )}
        </Panel>
      )}

      <Panel title="Stages" endpoint="capture">
        <div className="flex flex-col gap-2">
          {STAGES.map((stage) => (
            <StageCard key={stage} stage={stage} capture={trace.capture?.[stage] ?? null} />
          ))}
        </div>
      </Panel>

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
          to={`/admin/evaluations?version=${trace.version}`}
        />
      </LampRow>

      <Header
        reportId={trace.report_id}
        turnIndex={trace.turn_index}
        question={trace.question}
        answer={trace.answer}
        gold={trace.gold_answer}
        correct={trace.correct}
        backTo="/admin/evaluations"
      />

      <Panel
        title="Program"
        endpoint={`/traces/eval/${trace.version}`}
        note="a scored turn has no clock on it — this run was never metered, so there is no latency or cost to show"
      >
        {/*
          The CSV writes a pandas `nan` for a turn that produced no program.
          Printing that string would present a missing value as a computed one.
        */}
        {trace.program && trace.program !== 'nan' ? (
          <Json value={trace.program} />
        ) : (
          <p className="type-small text-faint">
            No program recorded for this turn — it was answered as a direct lookup, or the run
            predates program capture.
          </p>
        )}
      </Panel>

      <Panel title="Stages" endpoint="capture">
        <div className="flex flex-col gap-2">
          {STAGES.map((stage) => (
            <StageCard key={stage} stage={stage} capture={trace.capture?.[stage] ?? null} />
          ))}
        </div>
      </Panel>

      {trace.history_text && (
        <Panel title="Conversation history the agents saw">
          <Json value={trace.history_text} />
        </Panel>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------

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

  return (
    <AdminPage
      testId="admin-trace-detail"
      eyebrow={isEval ? 'admin · traces · scored turn' : 'admin · traces · turn'}
      title="Turn detail"
      sub={
        isEval
          ? 'A scored turn replayed out of a committed predictions CSV — the same four stage captures a live turn records, from a run that happened months ago.'
          : 'Every stage of one answered turn: what each agent was given, what it returned, and what it cost.'
      }
    >
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
    </AdminPage>
  );
}
