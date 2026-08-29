import { useQuery } from '@tanstack/react-query';
import { ChevronRight } from 'lucide-react';
import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import * as api from '../../api';
import { useMode } from '../../modeStore';
import { qk } from '../../lib/queryClient';
import type { Message, StageCapture, StageName, ToolTrace } from '../../types';
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
    <div className="mt-1.5">
      <div className="mono-caps">{label}</div>
      <pre className="mt-0.5 max-h-52 overflow-auto rounded-[4px] border border-line bg-panel px-1.5 py-1 font-mono text-[10.5px] leading-relaxed whitespace-pre-wrap break-words text-muted">
        {value}
      </pre>
    </div>
  );
}

function ToolLoop({ lines }: { lines: ToolLine[] }) {
  if (lines.length === 0) return null;
  return (
    <div className="mt-1.5">
      <div className="mono-caps">tool loop · {lines.length} calls</div>
      <ol className="mt-0.5 space-y-0.5 font-mono text-[10.5px] leading-relaxed">
        {lines.map((line, i) => (
          <li key={i} className="break-words">
            <span className="text-amber">→</span>{' '}
            <span className="text-text">
              {line.tool}({line.args})
            </span>
            {line.result !== undefined && (
              <>
                {'  '}
                <span className="text-good">←</span> <span className="text-muted">{line.result}</span>
              </>
            )}
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

  return (
    <div className="border-b border-line px-2.5 py-2" data-stage={view.stage} data-state={view.state}>
      <button
        type="button"
        onClick={onToggle}
        disabled={skipped}
        className="flex w-full items-baseline gap-1.5 text-left disabled:cursor-default"
      >
        <ChevronRight
          aria-hidden
          className={cn(
            'size-3 shrink-0 translate-y-0.5 text-faint transition-transform',
            expanded && 'rotate-90',
            skipped && 'opacity-0'
          )}
        />
        <span
          className={cn(
            'font-mono text-[11.5px]',
            skipped ? 'text-faint line-through' : 'text-text',
            view.state === 'active' && 'text-amber'
          )}
        >
          {view.stage}
        </span>
        <span className="ml-auto shrink-0 font-mono text-[10px] text-muted">
          {skipped ? 'skipped' : `${fmtMs(metrics?.latency_ms)} · ${fmtTokens(metrics?.total_tokens)} tok`}
        </span>
      </button>

      {!skipped && view.detail && (
        <div className="mt-0.5 pl-4.5 font-mono text-[10px] leading-relaxed break-words text-faint">
          {view.detail}
        </div>
      )}

      {expanded && !skipped && (
        <div className="pl-4.5">
          <Block label="input" value={asText(capture?.input)} />
          {!capture?.input && (
            <p className="mt-1.5 text-[10px] leading-relaxed text-faint">
              {traceLoaded
                ? 'This turn’s stored trace records outputs only — a replayed turn is rebuilt from its recorded event stream, which never carried the stage inputs.'
                : 'Stage inputs live in the trace store, not on the live stream, and no stored trace was found for this turn.'}
            </p>
          )}
          <Block label="reasoning" value={reasoning} />
          <Block label="output" value={asText(output)} />
          <ToolLoop lines={toolLines} />
        </div>
      )}
    </div>
  );
}

function Total({ label, value, reason }: { label: string; value: string; reason?: string }) {
  return (
    <div title={value === EM_DASH ? reason : undefined}>
      <div className="font-mono text-[13px] text-text">{value}</div>
      <div className="mono-caps">{label}</div>
    </div>
  );
}

/**
 * The docked trace inspector.
 *
 * Docked, never a modal: the argument the whole console makes is that an
 * agentic system's reasoning is the product, so hiding it behind a click would
 * undo the design. It sits at `--ground` beside the lit thread.
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
      <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-b border-line px-2.5">
        <span className="mono-caps">
          {message ? `trace · turn ${turnNumber}` : 'trace'}
        </span>
        {message?.traceId && (
          <span className="truncate font-mono text-[10px] text-faint" title={message.traceId}>
            {message.traceId.slice(0, 8)}
          </span>
        )}
      </div>

      <div
        data-testid="stage-output-panel"
        data-stage={expanded ?? 'all'}
        className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden"
      >
        {!message ? (
          <p className="px-2.5 py-4 text-[11px] leading-relaxed text-faint">
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

            <div className="grid grid-cols-3 gap-2 border-b border-line px-2.5 py-2.5">
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
              <p className="px-2.5 py-2 text-[10px] leading-relaxed text-faint">
                {EM_DASH} means not measured, not zero. Recorded turns in the demo pack carry no
                latency or token figures yet; populating them needs a metered eval run.
              </p>
            )}
          </>
        )}
      </div>

      {bundle && (
        <div
          className="shrink-0 border-t border-line px-2.5 py-2 font-mono text-[9.5px] leading-relaxed break-words text-faint"
          title="The fingerprint this answer is attributable to — prompts, both models, the dataset and the code that ran them"
        >
          bundle {bundleId} · prompts {bundle.prompts_version} · {bundle.lm_mini} · dataset{' '}
          {bundle.dataset_hash} · code {bundle.code_sha}
        </div>
      )}
    </>
  );
}
