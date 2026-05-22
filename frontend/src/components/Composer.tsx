import { useEffect, useState } from 'react';
import * as api from '../api';
import { useStore } from '../store';
import type { ReportQuestion } from '../types';

interface Props {
  reportId: string;
  isStreaming: boolean;
}

export function Composer({ reportId, isStreaming }: Props) {
  const ask = useStore((s) => s.ask);
  const runAllGold = useStore((s) => s.runAllGold);
  const [text, setText] = useState('');
  const [questions, setQuestions] = useState<ReportQuestion[]>([]);

  useEffect(() => {
    let cancelled = false;
    api
      .getQuestions(reportId)
      .then((qs) => {
        if (!cancelled) setQuestions(qs);
      })
      .catch(() => {
        if (!cancelled) setQuestions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [reportId]);

  const send = async (question: string, gold?: string, goldProgram?: string) => {
    if (!question.trim() || isStreaming) return;
    setText('');
    await ask(reportId, question.trim(), gold, goldProgram);
  };

  return (
    <div className="border-t border-black/40 bg-panel p-3 shrink-0">
      {questions.length > 0 && !isStreaming && (
        <div className="flex flex-wrap gap-1.5 mb-2" data-testid="suggestion-chips">
          {questions.map((q) => (
            <button
              key={q.q_order}
              type="button"
              onClick={() => void send(q.question, q.gold_answer, q.gold_program)}
              className="text-xs px-2 py-1 rounded-full bg-panel2 hover:bg-accent text-textMain truncate max-w-[260px]"
              title={q.question}
            >
              {q.question}
            </button>
          ))}
        </div>
      )}
      <div className="flex gap-2 items-end">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={1}
          placeholder="Ask a question…"
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
          title="Run all gold questions sequentially"
        >
          Run all gold{questions.length > 0 ? ` (${questions.length})` : ''}
        </button>
      </div>
    </div>
  );
}
