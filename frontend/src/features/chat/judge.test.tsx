import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { applyEvent } from '../../store';
import type { JudgeVerdict, Message, SSEEvent } from '../../types';
import { Turn } from './Turn';

/**
 * s12: the confidence judge's verdict on the wire and on the screen.
 *
 * Two contracts. The reducer must record the `judge` frame and the `withheld`
 * flag on `answer` so a replayed transcript still shows what the judge did;
 * and the turn must render a withheld answer as a statement of no confidence,
 * never as an empty number or a spinner that never resolves.
 */

const VERDICT: JudgeVerdict = {
  version: 'judge_j1',
  band: 'high',
  p_correct: 0.97,
  reason: 'every operand is in its cited cell',
  checks: {
    operand_in_source: 'pass',
    period_matches: 'pass',
    reference_resolved: 'pass',
    program_matches: 'pass',
    arithmetic_verified: 'pass',
    unit_and_scale: 'pass',
  },
};

function message(overrides: Partial<Message> = {}): Message {
  return {
    id: 'm1',
    role: 'assistant',
    text: '',
    status: 'streaming',
    createdAt: 0,
    ...overrides,
  };
}

function reduce(events: SSEEvent[], start = message()): Message {
  return events.reduce(applyEvent, start);
}

describe('judge frames', () => {
  it('records the verdict and releases a high-band answer', () => {
    const m = reduce([
      { event: 'judge', ...VERDICT },
      { event: 'answer', answer: '150', program: 'subtract(A, B)', band: 'high', withheld: false },
      { event: 'done', turn_index: 0 },
    ]);
    expect(m.judge?.band).toBe('high');
    expect(m.judge?.checks.unit_and_scale).toBe('pass');
    expect(m.text).toBe('150');
    expect(m.withheld).toBe(false);
  });

  it('marks a low-band answer withheld with an empty text', () => {
    const m = reduce([
      { event: 'judge', ...VERDICT, band: 'low', p_correct: 0.3 },
      { event: 'answer', answer: '', program: '', band: 'low', withheld: true },
      { event: 'done', turn_index: 0 },
    ]);
    expect(m.withheld).toBe(true);
    expect(m.text).toBe('');
    expect(m.judge?.band).toBe('low');
  });

  it('leaves pipeline and demo turns untouched', () => {
    const m = reduce([{ event: 'answer', answer: '42' }, { event: 'done', turn_index: 0 }]);
    expect(m.judge).toBeUndefined();
    expect(m.withheld).toBe(false);
  });
});

describe('Turn with a judge', () => {
  it('shows the band beside a released answer', () => {
    const html = renderToStaticMarkup(
      <Turn
        message={message({ text: '150', status: 'done', judge: VERDICT, withheld: false })}
        selected={false}
        onSelect={() => {}}
      />
    );
    expect(html).toContain('data-testid="judge-badge"');
    expect(html).toContain('data-band="high"');
    expect(html).toContain('p 0.97');
    expect(html).toContain('150');
  });

  it('says it is not confident instead of showing a withheld number', () => {
    const verdict: JudgeVerdict = {
      ...VERDICT,
      band: 'low',
      p_correct: 0.3,
      reason: 'the cited cell holds a different period',
      checks: { ...VERDICT.checks, period_matches: 'fail' },
    };
    const html = renderToStaticMarkup(
      <Turn
        message={message({ text: '', status: 'done', judge: verdict, withheld: true })}
        selected={false}
        onSelect={() => {}}
      />
    );
    expect(html).toContain('data-testid="withheld-block"');
    expect(html).toContain('not confident in this one');
    expect(html).toContain('the cited cell holds a different period');
    expect(html).toContain('period_matches: fail');
    expect(html).not.toContain('answering…');
    expect(html).not.toContain('judge-badge');
  });
});
