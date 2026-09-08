import { describe, expect, it } from 'vitest';
import type { Message } from '../../types';
import { howLine } from './how';

function message(patch: Partial<Message> = {}): Message {
  return { id: 'm1', role: 'assistant', text: '35.4%', status: 'done', createdAt: 0, ...patch };
}

describe('the how line', () => {
  it('narrates a program turn from its stage outputs', () => {
    const line = howLine(
      message({
        stages: {
          triage: { started: true, output: { turn_type: 'program', conv_type: 'Type I' } },
          preprocess: {
            started: true,
            output: { sub_questions: ['a', 'b'], program: 'divide(subtract(A, B), B)' },
          },
          retriever: {
            started: true,
            output: { answers: [{ question: 'a', answer: '868' }, { question: 'b', answer: '641' }] },
          },
          calculator: { started: true, output: { answer: '35.4' } },
        },
        tools: [
          { tool: 'subtract', args: { a: 868, b: 641 }, result: '227' },
          { tool: 'divide', args: { a: 227, b: 641 }, result: '0.354' },
        ],
      }),
    );
    expect(line).toBe(
      'A computation in a Type I conversation. Preprocess split it into 2 sub-questions and planned divide(subtract(A, B), B). The retriever found 868, 641 in the filing. The calculator ran 2 calls (subtract, divide) to reach the answer.',
    );
  });

  it('says a look-up skipped the two middle stages', () => {
    const line = howLine(
      message({
        stages: {
          triage: { started: true, output: { turn_type: 'number', conv_type: 'Type II' } },
          retriever: { started: true, output: { answers: [{ question: 'q', answer: '641' }] } },
        },
      }),
    );
    expect(line).toMatch(/^A look-up in a Type II conversation: triage sent it straight to the retriever/);
    expect(line).toMatch(/found 641 in the filing\.$/);
    expect(line).not.toMatch(/Preprocess/);
  });

  it('says nothing while streaming or with no trace', () => {
    expect(howLine(message({ status: 'streaming' }))).toBe('');
    expect(howLine(message({ stages: {} }))).toBe('');
  });
});
