import { useState } from 'react';
import { looseNumericMatch } from '../numericMatch';
import type { Message, StageName, ToolTrace } from '../types';

const STAGE_LABEL: Record<StageName, string> = {
  triage: 'triage',
  preprocess: 'preprocess',
  retriever: 'retriever',
  calculator: 'calculator',
};
const STAGE_ORDER: StageName[] = ['triage', 'preprocess', 'retriever', 'calculator'];

function fmtArgs(args: unknown): string {
  if (typeof args === 'string') return args;
  try {
    return JSON.stringify(args);
  } catch {
    return String(args);
  }
}

function StageChips({
  message,
  selected,
  onToggle,
}: {
  message: Message;
  selected: StageName | null;
  onToggle: (stage: StageName) => void;
}) {
  const stages = message.stages ?? {};
  return (
    <div className="flex flex-wrap gap-1.5 mb-2">
      {STAGE_ORDER.filter((s) => stages[s]?.started).map((stage) => {
        const has = stages[stage]?.output !== undefined;
        const isActive = selected === stage;
        return (
          <button
            type="button"
            key={stage}
            data-stage={stage}
            data-has-output={has ? 'true' : 'false'}
            data-active={isActive ? 'true' : 'false'}
            onClick={() => onToggle(stage)}
            disabled={!has}
            title={has ? `Inspect ${stage} output` : `${stage} running…`}
            className={`text-[10px] px-2 py-0.5 rounded-full transition-colors ${
              isActive
                ? 'bg-accent2 text-bg ring-1 ring-accent2'
                : has
                  ? 'bg-accent text-textMain hover:bg-accent2 hover:text-bg cursor-pointer'
                  : 'bg-panel2 text-textMuted animate-pulse cursor-default'
            }`}
          >
            {STAGE_LABEL[stage]}
            {has ? '' : '…'}
          </button>
        );
      })}
    </div>
  );
}

function KV({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="text-textMuted shrink-0 min-w-20">{label}</span>
      <span className="font-mono break-all">{value}</span>
    </div>
  );
}

function ReasoningBlock({ reasoning }: { reasoning?: unknown }) {
  if (typeof reasoning !== 'string' || !reasoning.trim()) return null;
  return (
    <details className="mt-1.5">
      <summary className="cursor-pointer text-textMuted select-none">reasoning</summary>
      <div className="mt-1 pl-3 whitespace-pre-wrap text-textMuted">{reasoning}</div>
    </details>
  );
}

function ToolTraceList({ tools }: { tools: ToolTrace[] }) {
  if (tools.length === 0) {
    return <div className="text-textMuted italic">No tool calls.</div>;
  }
  return (
    <ul className="space-y-1">
      {tools.map((t, i) => (
        <li key={i} className="font-mono">
          <span className="text-accent2">⚙ {t.tool}</span>
          <span>({fmtArgs(t.args)})</span>
          {t.result !== undefined && (
            <span className="block pl-4 opacity-80">= {t.result}</span>
          )}
        </li>
      ))}
    </ul>
  );
}

function StageOutput({
  stage,
  message,
}: {
  stage: StageName;
  message: Message;
}) {
  const output = message.stages?.[stage]?.output as
    | Record<string, unknown>
    | undefined;
  if (!output) {
    return (
      <div className="text-textMuted italic">No output captured for this stage.</div>
    );
  }

  if (stage === 'triage') {
    return (
      <div className="space-y-1.5">
        <KV label="turn_type" value={String(output.turn_type ?? '—')} />
        <KV label="conv_type" value={String(output.conv_type ?? '—')} />
        <ReasoningBlock reasoning={output.reasoning} />
      </div>
    );
  }

  if (stage === 'preprocess') {
    const sq = Array.isArray(output.sub_questions)
      ? (output.sub_questions as string[])
      : [];
    return (
      <div className="space-y-1.5">
        <div>
          <span className="text-textMuted">sub_questions ({sq.length})</span>
          <ul className="list-decimal list-inside mt-1 space-y-0.5">
            {sq.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </div>
        <KV label="program" value={String(output.program ?? '—')} />
        <ReasoningBlock reasoning={output.reasoning} />
      </div>
    );
  }

  if (stage === 'retriever') {
    const answers = Array.isArray(output.answers)
      ? (output.answers as Array<{ question?: string; answer?: string }>)
      : [];
    return (
      <div className="space-y-1.5">
        <div className="text-textMuted">retrieved values ({answers.length})</div>
        <ul className="space-y-1.5 mt-1">
          {answers.map((a, i) => (
            <li key={i} className="border-l-2 border-panel2 pl-2">
              <div className="text-textMuted">Q: {a.question ?? '—'}</div>
              <div className="font-mono">A: {a.answer ?? '—'}</div>
            </li>
          ))}
        </ul>
        <ReasoningBlock reasoning={output.reasoning} />
      </div>
    );
  }

  // calculator
  const tools = message.tools ?? [];
  return (
    <div className="space-y-1.5">
      <KV label="answer" value={String(output.answer ?? '—')} />
      <div>
        <div className="text-textMuted mb-1">tool calls ({tools.length})</div>
        <ToolTraceList tools={tools} />
      </div>
    </div>
  );
}

export function MessageBubble({ message }: { message: Message }) {
  const [selectedStage, setSelectedStage] = useState<StageName | null>(null);

  const toggleStage = (stage: StageName) => {
    setSelectedStage((prev) => (prev === stage ? null : stage));
  };

  if (message.role === 'system') {
    return (
      <div
        className="text-center text-xs text-textMuted italic my-2"
        data-role="system-message"
      >
        {message.text}
      </div>
    );
  }

  if (message.role === 'user') {
    return (
      <div className="flex justify-end my-1.5" data-role="user-message">
        <div className="bg-bubbleUser text-textMain rounded-lg px-3 py-2 max-w-[75%] whitespace-pre-wrap break-words">
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
      className="flex justify-start my-1.5"
      data-role="assistant-message"
      data-streaming={isStreaming ? 'true' : 'false'}
      data-final={message.status === 'done' ? 'true' : 'false'}
      data-gold={
        goldVerdict === undefined ? undefined : goldVerdict ? 'match' : 'mismatch'
      }
    >
      <div
        className={`bg-bubbleAssistant rounded-lg px-3 py-2 max-w-[80%] min-w-0 overflow-x-hidden ${
          isError ? 'border border-danger' : ''
        }`}
      >
        <StageChips
          message={message}
          selected={selectedStage}
          onToggle={toggleStage}
        />
        {selectedStage && (
          <div
            data-testid="stage-output-panel"
            data-stage={selectedStage}
            className="mb-2 p-2 rounded bg-bg/50 border border-black/30 text-xs overflow-x-auto"
          >
            <StageOutput stage={selectedStage} message={message} />
          </div>
        )}
        {message.text ? (
          <div className="text-base whitespace-pre-wrap break-words">{message.text}</div>
        ) : isStreaming ? (
          <div className="text-sm text-textMuted italic">thinking…</div>
        ) : null}
        {message.goldAnswer && message.status === 'done' && (
          <div className="text-xs mt-1.5 space-y-0.5">
            {goldVerdict ? (
              <div className="text-accent2">✓ matches gold ({message.goldAnswer})</div>
            ) : (
              <div className="text-danger">✗ expected {message.goldAnswer}</div>
            )}
            {message.goldProgram && (
              <div className="text-textMuted font-mono break-all">
                program: {message.goldProgram}
              </div>
            )}
          </div>
        )}
        {isError && message.errorText && (
          <div className="text-xs mt-1.5 text-danger">Error: {message.errorText}</div>
        )}
      </div>
    </div>
  );
}
