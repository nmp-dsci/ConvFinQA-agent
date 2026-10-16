import { describe, expect, it } from 'vitest';
import { CAMPAIGNS, PASSES, STEPS, TRAIN, pass } from './fixtures';
import {
  championLine,
  dateBands,
  daysBetween,
  filterRows,
  formatDay,
  formatPp,
  ledgerRows,
  orderPasses,
  passLabel,
  recordSentence,
  shortModel,
  storySummary,
  timelineModel,
} from './timeline';
import { assignRows } from './TimelineChart';

describe('the timeline model', () => {
  it('orders passes by start time, untimed ones last and in their given order', () => {
    const untimed = pass({ run_name: 'x', version: 'v9', at_ms: null });
    const ordered = orderPasses([untimed, ...PASSES]);
    expect(ordered[0].version).toBe('v2');
    expect(ordered.at(-1)).toBe(untimed);
    const times = ordered.slice(0, -1).map((p) => p.at_ms as number);
    expect(times).toEqual([...times].sort((a, b) => a - b));
  });

  it('keeps an incomplete pass in the ledger and off the axis', () => {
    const m = timelineModel(PASSES, STEPS);
    expect(m.passes).toHaveLength(PASSES.length);
    expect(m.points).toHaveLength(PASSES.length - 1);
    expect(m.excluded.map((p) => p.outcome)).toEqual(['incomplete']);
    expect(m.points.some((p) => p.accuracy === 0.444126)).toBe(false);
  });

  it('bands by day and breaks the axis only between days that are not consecutive', () => {
    const bands = dateBands(timelineModel(PASSES, STEPS).points);
    expect(bands.map((b) => [b.date, b.from, b.to, b.breakBefore])).toEqual([
      ['2026-09-04', 0, 2, false],
      ['2026-09-05', 3, 4, false],
      ['2026-10-15', 5, 5, true],
    ]);
    expect(bands[2].gapDays).toBe(40);
  });

  it('steps the champion line at the pass that measured each champion, not when it was written', () => {
    const m = timelineModel(PASSES, STEPS);
    const line = m.line;
    expect(line.map((s) => [s.step.version, s.from, s.to])).toEqual([
      ['v2', 0, 2],
      ['v8', 2, 3],
      ['sdk_v1', 3, 5],
    ]);
    // sdk_v2 ran before sdk_v1's promotion was recorded and is still set against sdk_v1's level.
    expect(m.points[4].version).toBe('sdk_v2');
    expect(line[2].from).toBeLessThan(4);
    // The universal-champion record restates sdk_v1: folded in, not a second step.
    expect(line[2].restated.map((r) => r.event)).toEqual(['universal_champion']);
    // The legacy promotion with no gate accuracy is not on the line.
    expect(line.some((s) => s.step.version === 'v3_1')).toBe(false);
    expect(line[2].pass?.run_name).toBe('sdk-evalloop-test100-sdk_v1·s1-20260905_174627');
  });

  it('drops a step recorded after the last pass and a line over no points', () => {
    const late = { ...STEPS[3], effective_at_ms: Number.MAX_SAFE_INTEGER, version: 'v99' };
    expect(championLine(timelineModel(PASSES, []).points, [late])).toEqual([]);
    expect(championLine([], STEPS)).toEqual([]);
  });

  it('summarises the story without counting anything by hand', () => {
    const s = storySummary(timelineModel(PASSES, STEPS));
    expect(s.champion?.step.version).toBe('sdk_v1');
    expect(s.since).toBe('2026-09-05');
    expect(s.challengersSince.map((p) => p.version)).toEqual(['sdk_v2', 'sdk_v1']);
    expect(s.nBeat).toBe(0);
    expect([s.nPasses, s.nVersions, s.nModels]).toEqual([7, 5, 3]);
  });

  it('writes the Overview record from the data', () => {
    const s = storySummary(timelineModel(PASSES, STEPS));
    expect(recordSentence(s, 349)).toBe(
      'Champion sdk-v1 · sonnet-5 · high at 90.5% on 349 unseen questions since 5 Sep 2026; challengers since: sdk-v2 · sonnet-5 −2.87pp (rejected), sdk-v1 · haiku-5-5 −3.15pp (falls short)',
    );
    expect(recordSentence(storySummary(timelineModel([], [])))).toBe(
      'No champion has been measured on the gate split yet',
    );
  });
});

describe('the words', () => {
  it('names models the way run names do', () => {
    expect(shortModel('claude-haiku-4-5-20251001')).toBe('haiku-4-5');
    expect(shortModel('claude-sonnet-5')).toBe('sonnet-5');
    expect(shortModel('deepseek-v4-flash')).toBe('deepseek-v4-flash');
    expect(passLabel(PASSES[1])).toBe('llm-v2');
    expect(passLabel(PASSES[6])).toBe('sdk-v1 · haiku-5-5');
  });

  it('formats days and deltas', () => {
    expect(formatDay('2026-09-05')).toBe('5 Sep 2026');
    expect(formatDay('2026-09-05', false)).toBe('5 Sep');
    expect(formatDay(null)).toBe('—');
    expect(daysBetween('2026-09-06', '2026-10-15')).toBe(39);
    expect(formatPp(2.2923)).toBe('+2.29pp');
    expect(formatPp(-3.1519)).toBe('−3.15pp');
    expect(formatPp(null)).toBe('—');
  });
});

describe('the ledger rows', () => {
  it('adds train passes only when asked, in time order, marked as train', () => {
    const test = ledgerRows(timelineModel(PASSES, STEPS).passes);
    expect(test.every((r) => r.split === 'test')).toBe(true);
    const both = ledgerRows(timelineModel(PASSES, STEPS).passes, TRAIN);
    expect(both).toHaveLength(PASSES.length + TRAIN.length);
    expect(both.filter((r) => r.split === 'train').map((r) => r.outcome)).toEqual(['train', 'train']);
    const times = both.map((r) => r.at_ms as number);
    expect(times).toEqual([...times].sort((a, b) => a - b));
  });

  it('filters by runtime, model and outcome together', () => {
    const rows = ledgerRows(timelineModel(PASSES, STEPS).passes, TRAIN);
    expect(filterRows(rows, { runtime: 'pipeline', model: '', outcome: '' })).toHaveLength(3);
    expect(filterRows(rows, { runtime: '', model: 'haiku-5-5', outcome: '' })).toHaveLength(2);
    expect(filterRows(rows, { runtime: '', model: 'haiku-5-5', outcome: 'falls_short' })).toHaveLength(1);
  });
});

describe('label rows', () => {
  it('shares a row when labels clear each other and staggers when they would collide', () => {
    expect(assignRows([{ x: 50, width: 40 }, { x: 150, width: 40 }])).toEqual([0, 0]);
    expect(assignRows([{ x: 50, width: 120 }, { x: 100, width: 120 }, { x: 150, width: 120 }])).toEqual([0, 1, 2]);
    // Past four rows, the row with the most room takes it rather than a fifth.
    const crowd = Array.from({ length: 6 }, (_, i) => ({ x: 100 + i * 5, width: 200 }));
    expect(Math.max(...assignRows(crowd))).toBe(3);
  });
});

describe('the fixture matches what the backend serves', () => {
  it('carries the gate split size the pages print', () => {
    expect(CAMPAIGNS.split.gate_questions).toBe(349);
  });
});

describe('label placement', () => {
  it('moves an edge label inside the frame instead of cutting it off', async () => {
    const { clampCentre } = await import('./TimelineChart');
    expect(clampCentre(670, 140, 4, 676)).toBe(606);
    expect(clampCentre(10, 60, 4, 676)).toBe(34);
    expect(clampCentre(300, 60, 4, 676)).toBe(300);
  });
});
