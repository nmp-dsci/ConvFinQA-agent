import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactElement } from 'react';
import committed from '../../../../evaluation/fine_tuning.json';
import type { CaseRow, FineTuningRecord } from './api';
import { FineTuningView } from './FineTuningPage';
import { campaignsLink, caseCounts, headline, matches, overallCases, passIndex, roundCases, startTest } from './model';

// The committed record itself: `tests/test_fine_tuning.py` holds it to its inputs, this file holds
// the page to it.
const RECORD = committed as unknown as FineTuningRecord;

function html(node: ReactElement, at = '/optimise/fine-tuning'): string {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return renderToStaticMarkup(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[at]}>{node}</MemoryRouter>
    </QueryClientProvider>,
  );
}

function count(markup: string, needle: string): number {
  return markup.split(needle).length - 1;
}

describe('the overall story', () => {
  const markup = html(<FineTuningView data={RECORD} />);

  it('claims base to the last pick on test, and how far toward the teacher', () => {
    const h = headline(RECORD);
    expect(h.base?.accuracy).toBeCloseTo(0.5616, 3);
    expect(h.final?.accuracy).toBeCloseTo(0.8138, 3);
    expect(markup).toContain(
      'Two rounds of fine-tuning took the student from 56.2% to <em>81.4%</em> on test 100, 80% of the way to its teacher',
    );
  });

  it('draws every stage, then the teacher as a reference row', () => {
    expect(markup).toContain('data-testid="ft-progress-chart"');
    for (const s of RECORD.progression.test) expect(markup).toContain(`data-stage="${s.key}"`);
    expect(markup).toContain('data-stage="teacher"');
    expect(markup).toContain('teacher 87.7%');
  });

  it('says which steps the gate would call significant', () => {
    expect(markup).toContain('3 steps on test 100, 1 of them significant on the gate’s rule');
  });

  it('opens each round from a card, and links each pass to its run on Campaigns', () => {
    expect(count(markup, 'data-round=')).toBe(RECORD.rounds.length + RECORD.planned.length);
    const base = passIndex(RECORD).get(RECORD.progression.test[0].pass!)!;
    expect(markup).toContain(`href="${campaignsLink(base.run_id)}"`);
  });

  it('says what it left off and why', () => {
    expect(markup).toContain('data-testid="ft-left-out"');
    expect(markup).toContain('an incomplete pass');
  });
});

describe('the DPO rounds (s40)', () => {
  const markup = html(<FineTuningView data={RECORD} />);

  it('puts DPO v1 on the progression from sft2 e1, on val only, and keeps the pilot off it', () => {
    const stage = RECORD.progression.val.find((s) => s.key === 'dpo1full')!;
    expect(stage.vs_prev?.from).toContain('sft2-e1');
    expect(stage.vs_prev?.delta_pp).toBeCloseTo(6.3, 1);
    expect(RECORD.progression.test.find((s) => s.key === 'dpo1full')?.pass).toBeNull();
    expect(RECORD.progression.val.some((s) => s.key === 'dpo1')).toBe(false);
    expect(markup).toContain('not opened on test (the owner’s word opens it)');
    // the test claim still covers only the rounds opened on test
    expect(headline(RECORD).nRounds).toBe(2);
  });

  it('says where the student is now on val, and lists DPO v2 as built, not trained', () => {
    expect(markup).toContain('<strong>DPO v1 · epoch 1</strong> reached 82.7% on val');
    expect(markup).toContain('2 SFT rounds and 2 DPO rounds, each picked on val; 1 more built, not trained');
    expect(markup).toContain('data-round="dpo_v2"');
  });

  it('sets the student beside every other agent on test 100', () => {
    expect(markup).toContain('data-testid="ft-agents-table"');
    expect(markup).toContain('90.5%');
    expect(markup).toContain('data-stage="latest-val"');
  });

  it('opens DPO v1 with its pairs, objective and held-out pair accuracy', () => {
    const dpo = html(<FineTuningView data={RECORD} />, '/optimise/fine-tuning?round=dpo1full');
    expect(dpo).toContain(
      'DPO v1 started from sft2 e1 and trained 1 epoch on 2,306 preference pairs; epoch 1 scored highest on val at 82.7%',
    );
    expect(dpo).toContain('73% prefer chosen');
    expect(dpo).toContain('DPO β 0.1 + α 1 × NLL on the chosen side');
  });
});

describe('a round', () => {
  const markup = html(<FineTuningView data={RECORD} />, '/optimise/fine-tuning?round=sft2');

  it('draws its sweep with the pick marked, and its loss', () => {
    expect(markup).toContain('data-testid="ft-checkpoint-chart"');
    expect(markup).toContain('data-checkpoint="e1" data-best="true"');
    expect(markup).toContain('data-testid="ft-loss-chart"');
  });

  it('claims where it started and what val picked', () => {
    expect(markup).toContain(
      'Round 2 started from sft1 e2 and trained 2 epochs on 916 conversations; epoch 1 scored highest on val at 76.4%',
    );
  });

  it('lists every checkpoint, the pick highlighted', () => {
    const r2 = RECORD.rounds.find((r) => r.key === 'sft2')!;
    expect(count(markup, 'data-checkpoint=')).toBe(r2.checkpoints.length * 2); // the chart and the table
    expect(markup).toContain('class="pro" data-checkpoint="e1"');
  });

  it('lists round 1’s re-scores under the table, not on its line', () => {
    const r1 = html(<FineTuningView data={RECORD} />, '/optimise/fine-tuning?round=sft1');
    expect(r1).toContain('data-testid="ft-rescores"');
    expect(r1).toContain('checked harness');
    expect(r1).toContain('temperature 0.2');
  });
});

describe('the cases', () => {
  it('lines up every test stage, then the teacher, judged from base to the last pick', () => {
    const set = overallCases(RECORD)!;
    const opened = RECORD.progression.test.filter((s) => s.pass).length;
    expect(set.split).toBe('test');
    expect(set.columns).toHaveLength(opened + 1);
    expect(set.to).toBe(opened - 1);
    expect(set.span).toBe(set.to + 1);
  });

  it('lines up a round on val from its start through each checkpoint, judged at the pick', () => {
    const r2 = RECORD.rounds.find((r) => r.key === 'sft2')!;
    const val = roundCases(r2, RECORD, 'val')!;
    expect(val.columns.map((c) => c.label)).toEqual(['Start · sft1 e2', 'epoch 0.5', 'epoch 1', 'epoch 1.5', 'epoch 2']);
    expect(val.to).toBe(2);
    const test = roundCases(r2, RECORD, 'test')!;
    expect(test.columns[0].pass).toBe(startTest(r2, RECORD)!.id);
    expect(test.columns).toHaveLength(2);
  });

  it('filters a case by where it started and where it ended', () => {
    const set = { split: 'val' as const, columns: [], from: 0, to: 1, fromLabel: 'a', toLabel: 'b', span: 3 };
    const row = (pattern: boolean[], turn = 'program'): CaseRow => ({
      report_id: 'r',
      turn_index: 0,
      question: 'q',
      gold_answer: '1',
      gold_program: '',
      turn_type: turn,
      answers: pattern.map((correct) => ({ answer: '1', correct })),
    });
    const rows = [row([false, true, true]), row([true, false, false]), row([true, true, false]), row([false, false, false], 'number')];
    expect(matches(rows[0], set, 'fixed', '')).toBe(true);
    expect(matches(rows[1], set, 'broken', '')).toBe(true);
    // changed later than the pick still counts as changed along the way
    expect(matches(rows[2], set, 'changed', '')).toBe(true);
    expect(matches(rows[3], set, 'changed', '')).toBe(false);
    expect(caseCounts(rows, set, '')).toEqual({ changed: 3, fixed: 1, broken: 1, wrong: 2, all: 4 });
    expect(caseCounts(rows, set, 'number')).toEqual({ changed: 0, fixed: 0, broken: 0, wrong: 1, all: 1 });
  });
});
