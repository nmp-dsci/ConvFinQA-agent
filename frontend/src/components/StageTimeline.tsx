import { useState } from 'react';
import type { StageCapture, StageMetrics, StageName } from '../types';
import { Badge, Mono, formatMs } from './ui';

const STAGE_ORDER: StageName[] = ['triage', 'preprocess', 'retriever', 'calculator'];

const STAGE_BLURB: Record<StageName, string> = {
  triage: 'Is this a value to look up, or a calculation to plan?',
  preprocess: 'Resolve references to earlier turns; draft the DSL program.',
  retriever: 'Find each value in the filing’s text and table.',
  calculator: 'Execute the program through the six-operation tool loop.',
};

function JsonBlock({ value }: { value: unknown }) {
  if (value === null || value === undefined) return null;
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  return (
    <pre className="text-[11px] leading-relaxed bg-bg/60 rounded p-2 overflow-x-auto whitespace-pre-wrap break-words max-h-80">
      {text}
    </pre>
  );
}

function MetricChips({ metrics }: { metrics?: StageMetrics }) {
  if (!metrics) return null;
  const chips: string[] = [];
  if (metrics.latency_ms !== undefined) chips.push(formatMs(metrics.latency_ms));
  if (metrics.total_tokens) chips.push(`${metrics.total_tokens} tok`);
  if (!chips.length) return null;
  return (
    <span className="flex gap-1">
      {chips.map((c) => (
        <Badge key={c}>{c}</Badge>
      ))}
    </span>
  );
}

function StageCard({
  stage,
  capture,
  defaultOpen,
}: {
  stage: StageName;
  capture: StageCapture | null | undefined;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  // A null capture is meaningful, not missing: the number path deliberately
  // skips preprocess and calculator, and showing that is part of the trace.
  if (capture === null) {
    return (
      <li className="flex items-center gap-3 py-2 opacity-45">
        <span className="size-2 rounded-full bg-textMuted/40 shrink-0" />
        <span className="text-sm capitalize">{stage}</span>
        <Badge>skipped — number path</Badge>
      </li>
    );
  }
  if (!capture) return null;

  const trajectory = capture.trajectory ?? [];

  return (
    <li className="border-l-2 border-accent2/40 pl-4 py-2 -ml-[5px]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 text-left group"
        aria-expanded={open}
      >
        <span className="text-textMuted text-xs w-3">{open ? '▾' : '▸'}</span>
        <span className="text-sm capitalize font-medium group-hover:text-accent2">{stage}</span>
        <MetricChips metrics={capture.metrics} />
        {trajectory.length > 0 && <Badge tone="accent">{trajectory.length} tool calls</Badge>}
      </button>

      <div className="text-xs text-textMuted mt-0.5 ml-5">{STAGE_BLURB[stage]}</div>

      {open && (
        <div className="mt-2 ml-5 space-y-2">
          {capture.reasoning ? (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-textMuted mb-1">
                Reasoning
              </div>
              <div className="text-xs leading-relaxed whitespace-pre-wrap">{capture.reasoning}</div>
            </div>
          ) : null}

          <details>
            <summary className="text-[11px] uppercase tracking-wide text-textMuted cursor-pointer hover:text-textMain">
              Input
            </summary>
            <div className="mt-1">
              <JsonBlock value={capture.input} />
            </div>
          </details>

          <div>
            <div className="text-[11px] uppercase tracking-wide text-textMuted mb-1">Output</div>
            <JsonBlock value={capture.output} />
          </div>

          {trajectory.length > 0 && (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-textMuted mb-1">
                Tool loop
              </div>
              <ol className="space-y-1">
                {trajectory.map((step, i) => (
                  <li key={i} className="text-[11px] font-mono flex gap-2">
                    <span className="text-textMuted shrink-0">
                      {step.event === 'tool_return' ? '←' : '→'}
                    </span>
                    <span className="text-accent2 shrink-0">{String(step.tool ?? '')}</span>
                    <span className="break-all">
                      {step.event === 'tool_return'
                        ? String(step.result ?? '')
                        : JSON.stringify(step.args ?? {})}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

/**
 * The stage timeline: triage → preprocess → retriever → calculator, with each
 * stage's IO, reasoning, tool loop, latency and tokens.
 *
 * One component serves both a live serving turn and a turn replayed out of a
 * committed eval CSV, because both were produced by the same `capture`
 * structure. That is what lets a year-old scored run open in the same viewer as
 * a question asked ten seconds ago.
 */
export function StageTimeline({
  capture,
  expandAll = false,
}: {
  capture: Partial<Record<StageName, StageCapture | null>>;
  expandAll?: boolean;
}) {
  const present = STAGE_ORDER.filter((s) => capture[s] !== undefined);
  if (!present.length) {
    return (
      <div className="text-xs text-textMuted p-3">
        No per-stage capture was recorded for this turn.
      </div>
    );
  }
  return (
    <ol className="border-l border-white/10 ml-1">
      {present.map((stage) => (
        <StageCard
          key={stage}
          stage={stage}
          capture={capture[stage]}
          defaultOpen={expandAll || stage === 'calculator'}
        />
      ))}
    </ol>
  );
}

export function GoldComparison({
  answer,
  gold,
  correct,
  program,
  goldProgram,
}: {
  answer: string | null;
  gold: string | null;
  correct: boolean | null;
  program?: string | null;
  goldProgram?: string | null;
}) {
  if (gold === null || gold === undefined) return null;
  return (
    <div className="grid grid-cols-2 gap-3 text-xs bg-bg/40 rounded p-3">
      <div>
        <div className="text-textMuted mb-1">Predicted</div>
        <Mono className={correct ? 'text-accent2' : 'text-danger'}>{answer || '—'}</Mono>
        {program ? <div className="mt-1 text-textMuted break-all">{program}</div> : null}
      </div>
      <div>
        <div className="text-textMuted mb-1">Gold</div>
        <Mono>{gold}</Mono>
        {goldProgram ? <div className="mt-1 text-textMuted break-all">{goldProgram}</div> : null}
      </div>
    </div>
  );
}
