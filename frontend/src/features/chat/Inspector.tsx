import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import * as api from '../../api';
import { useMode } from '../../modeStore';
import { qk } from '../../lib/queryClient';
import type { JudgeVerdict, Message, StageCapture, StageName, ToolTrace } from '../../types';
import { asText, EM_DASH, fmtCost, fmtMs, fmtTokens } from './format';
import { stageViews, totalLatency, totalTokens } from './stages';

/** `→ subtract(868, 641)` / `← 227.0` — the calculator's DSL loop, readably. */
interface ToolLine {
  tool: string;
  args: string;
  result?: string;
}

function fmtArgs(args: unknown): string {
  if (args === null || args === undefined) return '';
  if (typeof args === 'string') return args;
  if (typeof args === 'object' && !Array.isArray(args)) {
    return Object.entries(args as Record<string, unknown>)
      .map(([, value]) => (typeof value === 'string' ? value : JSON.stringify(value)))
      .join(', ');
  }
  return JSON.stringify(args);
}

function fromTools(tools: ToolTrace[]): ToolLine[] {
  return tools.map((t) => ({ tool: t.tool, args: fmtArgs(t.args), result: t.result }));
}

/**
 * The stored trace keeps the raw call/return frames rather than paired tools,
 * so a turn read back from `/traces/{id}` has to be re-paired here. Same
 * pairing rule as the live reducer: a return fills the most recent open call.
 */
function fromTrajectory(trajectory: Array<Record<string, unknown>>): ToolLine[] {
  const lines: ToolLine[] = [];
  for (const frame of trajectory) {
    const kind = String(frame.event ?? '');
    const tool = String(frame.tool ?? '');
    if (kind === 'tool_call') {
      lines.push({ tool, args: fmtArgs(frame.args) });
    } else if (kind === 'tool_return') {
      for (let i = lines.length - 1; i >= 0; i--) {
        if (lines[i].tool === tool && lines[i].result === undefined) {
          lines[i] = { ...lines[i], result: String(frame.result ?? '') };
          break;
        }
      }
    }
  }
  return lines;
}

function Block({ label, value }: { label: string; value: string }) {
  if (!value.trim()) return null;
  return (
    <div className="ch-block">
      <p className="label">{label}</p>
      <pre className="ch-pre">{value}</pre>
    </div>
  );
}

function ToolLoop({ lines }: { lines: ToolLine[] }) {
  if (lines.length === 0) return null;
  return (
    <div className="ch-block">
      <p className="label">tool loop · {lines.length} calls</p>
      <ol className="ch-tools">
        {lines.map((line, i) => (
          <li key={i}>
            → {line.tool}({line.args})
            {line.result !== undefined && <span className="r">{'  '}← {line.result}</span>}
          </li>
        ))}
      </ol>
    </div>
  );
}

interface StageRowProps {
  view: ReturnType<typeof stageViews>[number];
  message: Message;
  capture: StageCapture | null | undefined;
  /** Whether a stored trace was found at all, as opposed to found and thin. */
  traceLoaded: boolean;
  expanded: boolean;
  onToggle: () => void;
}

function StageRow({ view, message, capture, traceLoaded, expanded, onToggle }: StageRowProps) {
  const metrics = view.trace?.metrics ?? capture?.metrics;
  const output = view.trace?.output ?? capture?.output;
  const reasoning =
    (typeof output?.reasoning === 'string' ? output.reasoning : '') || capture?.reasoning || '';

  const toolLines =
    view.stage === 'calculator'
      ? (message.tools ?? []).length > 0
        ? fromTools(message.tools ?? [])
        : fromTrajectory(capture?.trajectory ?? [])
      : [];

  const skipped = view.state === 'skipped';
  const nameClass =
    view.state === 'active' ? 'ch-stage-name active' : skipped ? 'ch-stage-name skip' : 'ch-stage-name';

  return (
    <div className="ch-stage" data-stage={view.stage} data-state={view.state}>
      <button
        type="button"
        onClick={onToggle}
        disabled={skipped}
        aria-expanded={skipped ? undefined : expanded}
        className={expanded ? 'ch-stage-btn on' : 'ch-stage-btn'}
      >
        <span className="ch-glyph" aria-hidden>
          {skipped ? '–' : expanded ? '▾' : '▸'}
        </span>
        <span className={nameClass}>{view.stage}</span>
        <span className="ch-meta ch-push">
          {skipped ? 'skipped' : `${fmtMs(metrics?.latency_ms)} · ${fmtTokens(metrics?.total_tokens)} tok`}
        </span>
      </button>

      {!skipped && (view.detail || expanded) && (
        <div className="ch-stage-in">
          {view.detail && <p className="ch-meta">{view.detail}</p>}
          {expanded && (
            <>
              <Block label="input" value={asText(capture?.input)} />
              {!capture?.input && (
                <p className="small muted">
                  {traceLoaded
                    ? 'This turn’s stored trace records outputs only — a replayed turn is rebuilt from its recorded event stream, which never carried the stage inputs.'
                    : 'Stage inputs live in the trace store, not on the live stream, and no stored trace was found for this turn.'}
                </p>
              )}
              <Block label="reasoning" value={reasoning} />
              <Block label="output" value={asText(output)} />
              <ToolLoop lines={toolLines} />
            </>
          )}
        </div>
      )}
    </div>
  );
}

function Total({ label, value, reason }: { label: string; value: string; reason?: string }) {
  const absent = value === EM_DASH;
  return (
    <div className="stat" title={absent ? reason : undefined}>
      <div className={absent ? 'n absent' : 'n'}>{value}</div>
      <span className="label">{label}</span>
    </div>
  );
}

/** s12: the confidence judge's six checks, after the four stages. */
function JudgeRow({ verdict, withheld }: { verdict: JudgeVerdict; withheld: boolean }) {
  const low = verdict.band !== 'high';
  return (
    <div className="ch-stage ch-judge" data-stage="judge" data-band={verdict.band}>
      <div className="ch-judge-top">
        <span className="ch-stage-name">judge</span>
        <span className={low ? 'tag warn' : 'tag ok'}>
          <span aria-hidden>{low ? '◐ ' : '● '}</span>
          {verdict.band} · p {verdict.p_correct.toFixed(2)}
          {withheld ? ' · withheld' : ''}
        </span>
        {verdict.version && <span className="ch-meta ch-push">{verdict.version}</span>}
      </div>
      <p>{verdict.reason}</p>
      <ul className="ch-checks">
        {Object.entries(verdict.checks ?? {}).map(([name, value]) => (
          <li key={name} data-check={name} data-verdict={value}>
            <span className="k">{name}</span>
            <span className={value === 'pass' ? 'status ok' : value === 'fail' ? 'status err' : 'status warn'}>
              {value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * The docked trace inspector.
 *
 * Docked, never a modal: the argument the whole console makes is that an
 * agentic system's reasoning is the product, so hiding it behind a click would
 * undo the design. It is `--panel` chrome beside the thread, one row per stage,
 * each opened in place; "open trace" goes to the full record on the Runs page.
 */
export function Inspector({ message, turnNumber }: { message: Message | null; turnNumber: number }) {
  const [expanded, setExpanded] = useState<StageName | null>(null);
  const bundleId = useMode((s) => s.health?.bundle_id);
  const bundle = useMode((s) => s.health?.bundle);

  useEffect(() => {
    setExpanded(null);
  }, [message?.id]);

  // The live stream carries outputs but not stage *inputs* — those are only in
  // the stored trace. Fetched lazily, and its absence is a normal state (trace
  // capture can be off), never an error banner.
  const traceId = message?.traceId;
  const trace = useQuery({
    queryKey: qk.trace(traceId ?? ''),
    queryFn: () => api.getTrace(traceId as string),
    enabled: !!traceId,
    retry: false,
    staleTime: Infinity,
  });

  const views = message ? stageViews(message) : [];
  const latency = message ? (totalLatency(message) ?? trace.data?.latency_ms ?? null) : null;
  const tokens = message ? (totalTokens(message) ?? trace.data?.total_tokens ?? null) : null;
  const cost = trace.data?.cost_usd ?? null;

  return (
    <>
      <div className="ch-head">
        <span
          className="label ch-fill"
          title="What each stage was given, what it returned, and what it cost — the record behind the answer"
        >
          {message ? `trace · turn ${turnNumber}` : 'trace'}
        </span>
        {traceId && (
          <Link
            to={`/runs/traces/${encodeURIComponent(traceId)}`}
            className="ch-meta"
            title={`Open trace ${traceId} on the Runs page`}
          >
            {traceId.slice(0, 8)} · open trace ↗
          </Link>
        )}
      </div>

      <div
        data-testid="stage-output-panel"
        data-stage={expanded ?? 'all'}
        className="ch-scroll-y"
      >
        {!message ? (
          <p className="ch-note">
            Ask a question, or click any answer in the thread, to see the four stages that
            produced it — what each one was given, what it decided, and what it cost.
          </p>
        ) : (
          <>
            {views.map((view) => (
              <StageRow
                key={view.stage}
                view={view}
                message={message}
                capture={trace.data?.capture?.[view.stage]}
                traceLoaded={!!trace.data}
                expanded={expanded === view.stage}
                onToggle={() =>
                  setExpanded((prev) => (prev === view.stage ? null : view.stage))
                }
              />
            ))}

            {message.judge && <JudgeRow verdict={message.judge} withheld={!!message.withheld} />}

            <div className="stats ch-totals">
              <Total
                label="latency"
                value={fmtMs(latency)}
                reason="No per-stage timings were recorded for this turn"
              />
              <Total
                label="tokens"
                value={fmtTokens(tokens)}
                reason="No token counts were recorded for this turn"
              />
              <Total
                label="cost"
                value={fmtCost(cost)}
                reason="Cost is computed from token counts; none were recorded for this turn"
              />
            </div>

            {(latency === null || tokens === null) && (
              <p className="ch-note">
                {EM_DASH} means not measured, not zero. Recorded turns in the demo pack carry no
                latency or token figures yet; populating them needs a metered eval run.
              </p>
            )}
          </>
        )}
      </div>

      {bundle && (
        <p
          className="ch-meta ch-foot"
          title="The fingerprint this answer is attributable to — prompts, both models, the dataset and the code that ran them"
        >
          bundle {bundleId} · prompts {bundle.prompts_version} · {bundle.lm_mini} · dataset{' '}
          {bundle.dataset_hash} · code {bundle.code_sha}
        </p>
      )}
    </>
  );
}
