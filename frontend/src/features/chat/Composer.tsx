import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import * as api from '../../api';
import { useIsDemo } from '../../modeStore';
import { useStore } from '../../store';
import type { DemoQuestion, ReportQuestion } from '../../types';

export interface Chip {
  key: string;
  question: string;
  gold?: string;
  goldProgram?: string;
}

interface Props {
  reportId: string;
  isStreaming: boolean;
}

export function Composer({ reportId, isStreaming }: Props) {
  const ask = useStore((s) => s.ask);
  const runAllGold = useStore((s) => s.runAllGold);
  const examples = useStore((s) => s.examples);
  const isDemo = useIsDemo();

  const [text, setText] = useState('');
  const [questions, setQuestions] = useState<ReportQuestion[]>([]);
  const [recorded, setRecorded] = useState<DemoQuestion[]>([]);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getQuestions(reportId)
      .then((qs) => !cancelled && setQuestions(qs))
      .catch(() => !cancelled && setQuestions([]));
    return () => {
      cancelled = true;
    };
  }, [reportId]);

  // In demo mode the chip rail is not a convenience — it is the set of
  // questions that actually have a recording. Steering people to it is what
  // keeps the demo feeling like the product rather than like a dead end.
  useEffect(() => {
    if (!isDemo) {
      setRecorded([]);
      return;
    }
    let cancelled = false;
    api
      .getDemoQuestions(reportId)
      .then((qs) => !cancelled && setRecorded(qs))
      .catch(() => !cancelled && setRecorded([]));
    return () => {
      cancelled = true;
    };
  }, [reportId, isDemo]);

  const chips: Chip[] = isDemo
    ? recorded.map((q) => ({
        key: `demo-${q.turn_index}`,
        question: q.question,
        gold: q.gold_answer,
      }))
    : questions.map((q) => ({
        key: `gold-${q.q_order}`,
        question: q.question,
        gold: q.gold_answer,
        goldProgram: q.gold_program,
      }));

  const send = async (question: string, gold?: string, goldProgram?: string) => {
    if (!question.trim() || isStreaming) return;
    setText('');
    await ask(reportId, question.trim(), gold, goldProgram);
  };

  // The one control in this pane that a demo deployment genuinely cannot
  // honour: a filing with no recording has nothing to replay, so running every
  // turn would produce a column of identical refusals. It is disabled with the
  // reason on screen rather than hidden — the demo shows the whole product.
  const inPack = examples.some((e) => e.reportId === reportId);
  const runAllBlocked = isDemo && !inPack;

  return (
    <div className="shrink-0 border-t border-line bg-panel px-3 py-2.5">
      {chips.length > 0 && !isStreaming && (
        <div className="mb-2 flex min-w-0 items-start gap-2" data-testid="suggestion-chips">
          <span
            className={cn('mono-caps mt-1 shrink-0', isDemo && 'text-amber')}
            title={
              isDemo
                ? 'These are the questions this deployment has a recording for'
                : 'The dataset questions for this filing, with their gold answers'
            }
          >
            {isDemo ? 'recorded' : 'dataset'}
          </span>
          <div className="flex min-w-0 flex-wrap gap-1">
            {chips.map((chip) => (
              <button
                key={chip.key}
                type="button"
                onClick={() => void send(chip.question, chip.gold, chip.goldProgram)}
                title={chip.question}
                className="max-w-[280px] truncate rounded-full border border-line-2 px-2 py-0.5 text-[11px] text-muted transition-colors hover:border-amber-line hover:text-amber"
              >
                {chip.question}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-end gap-2">
        <textarea
          ref={inputRef}
          value={text}
          rows={1}
          onChange={(e) => setText(e.target.value)}
          disabled={isStreaming}
          placeholder={
            isDemo
              ? 'Ask in your own words — the demo plays the closest recorded question'
              : 'Ask about this filing…'
          }
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void send(text);
            }
          }}
          data-testid="composer-input"
          className="min-h-[34px] min-w-0 flex-1 resize-none rounded-md border border-line bg-ground px-2.5 py-2 text-[13px] text-text outline-none transition-colors focus:border-amber-line max-sm:basis-full disabled:opacity-50"
        />
        <button
          type="button"
          onClick={() => void send(text)}
          disabled={isStreaming || !text.trim()}
          data-testid="composer-send"
          className="h-[34px] shrink-0 rounded-md bg-amber px-3 text-[12px] font-semibold text-amber-ink transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          Send
        </button>

        <fieldset
          disabled={runAllBlocked}
          className="m-0 shrink-0 border-0 p-0 disabled:opacity-40"
          title={
            runAllBlocked
              ? 'This deployment replays recorded conversations and has no recording for this filing'
              : 'Run every question for this filing in order, threading the conversation'
          }
        >
          <button
            type="button"
            onClick={() => void runAllGold(reportId, chips)}
            disabled={isStreaming}
            data-testid="composer-run-all"
            className="h-[34px] rounded-md border border-line-2 px-3 text-[12px] font-medium text-muted transition-colors hover:border-amber-line hover:text-amber disabled:opacity-40"
          >
            {isDemo ? 'Replay all' : 'Run all gold'}
            {chips.length > 0 ? ` (${chips.length})` : ''}
          </button>
        </fieldset>
      </div>

      {runAllBlocked && (
        <p className="mt-1.5 text-[10.5px] text-faint">
          <span className="mono-caps mr-1 text-amber">demo</span>
          No recording for this filing, so there is nothing to replay. Open one of the
          recorded examples in the sessions pane to watch a full conversation.
        </p>
      )}
    </div>
  );
}
