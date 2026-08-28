import { useEffect, useState } from 'react';
import * as api from '../api';
import { useIsDemo } from '../modeStore';
import { useStore } from '../store';
import type { DemoQuestion, ReportQuestion } from '../types';
import { Badge } from './ui';

interface Props {
  reportId: string;
  isStreaming: boolean;
}

export function Composer({ reportId, isStreaming }: Props) {
  const ask = useStore((s) => s.ask);
  const runAllGold = useStore((s) => s.runAllGold);
  const [text, setText] = useState('');
  const [questions, setQuestions] = useState<ReportQuestion[]>([]);
  const [recorded, setRecorded] = useState<DemoQuestion[]>([]);
  const isDemo = useIsDemo();

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

  const send = async (question: string, gold?: string, goldProgram?: string) => {
    if (!question.trim() || isStreaming) return;
    setText('');
    await ask(reportId, question.trim(), gold, goldProgram);
  };

  const chips = isDemo
    ? recorded.map((q) => ({
        key: q.turn_index,
        question: q.question,
        gold: q.gold_answer,
        goldProgram: undefined as string | undefined,
      }))
    : questions.map((q) => ({
        key: q.q_order,
        question: q.question,
        gold: q.gold_answer,
        goldProgram: q.gold_program,
      }));

  return (
    <div className="border-t border-black/40 bg-panel p-3 shrink-0">
      {chips.length > 0 && !isStreaming && (
        <div className="mb-2" data-testid="suggestion-chips">
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[11px] text-textMuted">
              {isDemo ? 'Questions with a recording' : 'Dataset questions for this filing'}
            </span>
            {isDemo && <Badge tone="warn">demo replay</Badge>}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {chips.map((chip) => (
              <button
                key={chip.key}
                type="button"
                onClick={() => void send(chip.question, chip.gold, chip.goldProgram)}
                className="text-xs px-2 py-1 rounded-full bg-panel2 hover:bg-accent text-textMain truncate max-w-[260px]"
                title={chip.question}
              >
                {chip.question}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-2 items-end">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={1}
          placeholder={
            isDemo
              ? 'Ask in your own words — the demo answers what it has recorded'
              : 'Ask a question…'
          }
          disabled={isStreaming}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void send(text);
            }
          }}
          className="flex-1 bg-panel2 text-textMain placeholder:text-textMuted px-3 py-2 rounded-md outline-none resize-none disabled:opacity-50"
          data-testid="composer-input"
        />
        <button
          type="button"
          onClick={() => void send(text)}
          disabled={isStreaming || !text.trim()}
          className="px-3 py-2 rounded-md bg-accent2 text-bg font-semibold disabled:opacity-50"
          data-testid="composer-send"
        >
          Send
        </button>
        <button
          type="button"
          onClick={() => void runAllGold(reportId)}
          disabled={isStreaming}
          className="px-3 py-2 rounded-md bg-panel2 text-textMain font-semibold disabled:opacity-50"
          data-testid="composer-run-all"
          title="Run every dataset question for this filing, in order, threading the conversation"
        >
          Run all gold{chips.length > 0 ? ` (${chips.length})` : ''}
        </button>
      </div>
    </div>
  );
}
