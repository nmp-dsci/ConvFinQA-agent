import { describe, expect, it } from 'vitest';
import { looseNumericMatch } from './numericMatch';
import type { Message, SSEEvent } from './types';
import { applyEvent } from './store';

function blank(): Message {
  return {
    id: 'm1',
    role: 'assistant',
    text: '',
    status: 'streaming',
    createdAt: 0,
  };
}

function apply(message: Message, events: SSEEvent[]): Message {
  return events.reduce(applyEvent, message);
}

describe('SSE reducer', () => {
  it('records a stage output with its metrics', () => {
    const result = apply(blank(), [
      { event: 'stage_start', stage: 'triage' },
      {
        event: 'stage_output',
        stage: 'triage',
        output: { turn_type: 'number' },
        metrics: { latency_ms: 120, total_tokens: 88 },
      },
    ]);
    expect(result.stages?.triage?.started).toBe(true);
    expect(result.stages?.triage?.output).toEqual({ turn_type: 'number' });
    expect(result.stages?.triage?.metrics?.total_tokens).toBe(88);
  });

  it('pairs a tool return with the matching open call', () => {
    const result = apply(blank(), [
      { event: 'tool_call', stage: 'calculator', tool: 'subtract', args: { a: 5, b: 2 } },
      { event: 'tool_return', stage: 'calculator', tool: 'subtract', result: '3' },
    ]);
    expect(result.tools).toHaveLength(1);
    expect(result.tools?.[0].result).toBe('3');
  });

  it('does not overwrite an already-resolved tool call', () => {
    const result = apply(blank(), [
      { event: 'tool_call', stage: 'calculator', tool: 'add', args: { a: 1, b: 1 } },
      { event: 'tool_return', stage: 'calculator', tool: 'add', result: '2' },
      { event: 'tool_call', stage: 'calculator', tool: 'add', args: { a: 2, b: 2 } },
      { event: 'tool_return', stage: 'calculator', tool: 'add', result: '4' },
    ]);
    expect(result.tools?.map((t) => t.result)).toEqual(['2', '4']);
  });

  it('carries the trace id through `done` so the turn is inspectable', () => {
    const result = apply(blank(), [
      { event: 'answer', answer: '42' },
      { event: 'done', turn_index: 0, trace_id: 'abc123' },
    ]);
    expect(result.text).toBe('42');
    expect(result.status).toBe('done');
    expect(result.traceId).toBe('abc123');
  });

  it('keeps a stable error code, not just prose', () => {
    const result = apply(blank(), [
      { event: 'error', error: 'no recording for that question', code: 'no_recording' },
    ]);
    expect(result.status).toBe('error');
    expect(result.errorCode).toBe('no_recording');
  });

  it('does not resurrect an errored turn on `done`', () => {
    // `unknown` is the real fallback the backend classifies to — the literal
    // `"error"` this fixture used predates `convfinqa/error_codes.py`.
    const result = apply(blank(), [
      { event: 'error', error: 'boom', code: 'unknown' },
      { event: 'done', turn_index: 0 },
    ]);
    expect(result.status).toBe('error');
  });

  it('records a fuzzy replay match so the turn can say what it played', () => {
    const result = apply(blank(), [
      {
        event: 'matched',
        matched_question: 'what is the net change in cash from operations from 2008 to 2009?',
        asked_question: 'how much did operating cash move between 2008 and 2009?',
        score: 0.71,
      },
      { event: 'answer', answer: '227.0' },
      { event: 'done', turn_index: 0, trace_id: 't1' },
    ]);
    expect(result.matchedQuestion).toMatch(/net change in cash/);
    expect(result.askedQuestion).toMatch(/operating cash move/);
    expect(result.matchScore).toBe(0.71);
  });

  it('leaves matchedQuestion unset on an exact match', () => {
    // The server sends no `matched` frame when the question was recorded
    // verbatim, and `done` carries an empty string rather than omitting it.
    const result = apply(blank(), [
      { event: 'answer', answer: '227.0' },
      { event: 'done', turn_index: 0, matched_question: '' },
    ]);
    expect(result.matchedQuestion).toBeUndefined();
  });

  it('does not let `done` clear a match the `matched` frame already set', () => {
    const result = apply(blank(), [
      {
        event: 'matched',
        matched_question: 'recorded question',
        asked_question: 'paraphrase',
        score: 0.66,
      },
      { event: 'done', turn_index: 0 },
    ]);
    expect(result.matchedQuestion).toBe('recorded question');
  });
});

describe('looseNumericMatch', () => {
  // This is the *display* check, deliberately looser than the Python oracle in
  // `evaluation.metrics`: it strips symbols and compares rounded integers. It is
  // not a reimplementation of scoring, and the app must never present it as one
  // — the authoritative verdict is the `correct` column the backend serves.
  it('ignores currency and thousands separators', () => {
    expect(looseNumericMatch('$1,234', '1234')).toBe(true);
  });

  it('matches on rounded integers', () => {
    expect(looseNumericMatch('59.7%', '60%')).toBe(true);
  });

  it('rejects a genuinely different number', () => {
    expect(looseNumericMatch('12.0', '48.0')).toBe(false);
  });

  it('does not equate a percentage with its decimal form', () => {
    // 90.9 and 0.9091 round to 91 and 1. Callers that need that equivalence
    // must use the backend's verdict, not this helper.
    expect(looseNumericMatch('90.9%', '0.9091')).toBe(false);
  });

  it('handles a missing prediction without throwing', () => {
    expect(() => looseNumericMatch('', '1.0')).not.toThrow();
    expect(looseNumericMatch('', '1.0')).toBe(false);
  });
});
