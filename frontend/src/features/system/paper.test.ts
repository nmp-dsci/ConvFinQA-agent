import { describe, expect, it } from 'vitest';
import {
  BASELINES,
  BENCHMARK_CAVEATS,
  DATASET_SCALE,
  DISTRIBUTIONS,
  DSL_OPS,
  FINDINGS,
  PAPER,
  SLICE_BASELINES,
} from './paper';
import { OPEN_WORK, ourSliceAccuracy, sliceRows } from './benchmark';
import type { ModelAccuracy } from '../../types';

/**
 * These tests guard the one property this page sells: that every number on it
 * is either cited to the paper or read live from the backend, and that nothing
 * describing *this system* is a literal in a source file.
 */

describe('the paper is cited, not paraphrased', () => {
  it('names the arXiv id and the committed PDF', () => {
    expect(PAPER.arxivId).toBe('2210.03849');
    expect(PAPER.localPath).toMatch(/\.pdf$/);
  });

  it('gives every dataset statistic a table or figure number', () => {
    for (const stat of DATASET_SCALE) {
      expect(stat.source, stat.label).toMatch(/Table|Figure/);
    }
    for (const block of DISTRIBUTIONS) {
      expect(block.source, block.title).toMatch(/Table|Figure/);
    }
    for (const row of BASELINES) {
      expect(row.source, row.system).toMatch(/Table|Figure/);
    }
    for (const slice of SLICE_BASELINES) {
      expect(slice.source, slice.label).toMatch(/Table|Figure/);
    }
  });

  it('keeps each distribution summing to a whole', () => {
    for (const block of DISTRIBUTIONS) {
      const total = block.bars.reduce((sum, bar) => sum + bar.pct, 0);
      expect(total, block.title).toBeGreaterThan(99);
      expect(total, block.title).toBeLessThan(101);
    }
  });

  it('carries the whole six-operation DSL', () => {
    expect(DSL_OPS.map((o) => o.op)).toEqual([
      'add',
      'subtract',
      'multiply',
      'divide',
      'exp',
      'greater',
    ]);
  });

  it('states no figure for this system as a literal', () => {
    // The one row describing this project is spliced in from the live read.
    // If a hardcoded "77.1" ever appears in the baseline table, the benchmark
    // has stopped being live and starts drifting the moment a version ships.
    const blob = JSON.stringify(BASELINES);
    expect(blob).not.toMatch(/77\.1|77\.7|this project/i);
  });
});

describe('the caveats travel with the table', () => {
  it('keeps all four', () => {
    expect(BENCHMARK_CAVEATS).toHaveLength(4);
  });

  it('explains the execution/program gap rather than only naming it', () => {
    const gap = BENCHMARK_CAVEATS.find((c) => /program accuracy/i.test(c.title));
    expect(gap).toBeDefined();
    // The explanation is the whole point: without the prior-answers mechanism
    // the reader concludes the system is wrong two turns in three.
    expect(gap?.body).toMatch(/prior answers/i);
    expect(gap?.body).toMatch(/not evidence/i);
  });
});

describe('the paper findings each get an answer and a place to check it', () => {
  it('covers all six', () => {
    expect(FINDINGS).toHaveLength(6);
    for (const finding of FINDINGS) {
      expect(finding.paper.length).toBeGreaterThan(20);
      expect(finding.response.length).toBeGreaterThan(20);
      expect(finding.evidence.length).toBeGreaterThan(10);
    }
  });
});

// ---------------------------------------------------------------------------
// The live half
// ---------------------------------------------------------------------------

const summary: ModelAccuracy = {
  overall: { label: 'overall', accuracy: 0.7714, n_correct: 594, n_total: 770 },
  by_turn_type: [
    { label: 'Number', accuracy: 0.877, n_correct: 249, n_total: 284 },
    { label: 'Program', accuracy: 0.71, n_correct: 345, n_total: 486 },
  ],
  by_conv_type: [
    { label: 'Type I', accuracy: 0.788, n_correct: 504, n_total: 640 },
    { label: 'Type II', accuracy: 0.692, n_correct: 90, n_total: 130 },
  ],
  by_q_order: [
    { label: '0', accuracy: 0.82, n_correct: 164, n_total: 200 },
    { label: '1', accuracy: 0.794, n_correct: 158, n_total: 199 },
  ],
};

describe('ourSliceAccuracy', () => {
  it('finds a slice in either grouping', () => {
    expect(ourSliceAccuracy(summary, 'Number')).toBeCloseTo(87.7, 1);
    expect(ourSliceAccuracy(summary, 'Type II')).toBeCloseTo(69.2, 1);
  });

  it('returns null rather than zero for a slice that was never scored', () => {
    expect(ourSliceAccuracy(summary, 'Type II · second half')).toBeNull();
    expect(ourSliceAccuracy(undefined, 'Number')).toBeNull();
  });
});

describe('sliceRows', () => {
  it('renders every paper slice even when our side is missing', () => {
    const rows = sliceRows(undefined);
    expect(rows).toHaveLength(SLICE_BASELINES.length + 1);
    for (const row of rows) {
      expect(row.ours).toBeNull();
      expect(row.finqanet).toBeGreaterThan(0);
    }
  });

  it('keeps the unmeasured slice unmeasured even with a summary in hand', () => {
    const rows = sliceRows(summary);
    const unscored = rows[rows.length - 1];
    expect(unscored.ours).toBeNull();
    expect(unscored.why).toMatch(/not scored/i);
  });
});

describe('open work is stated, not softened', () => {
  it('says GEPA/DSPy runs are broken today', () => {
    const gepa = OPEN_WORK.find((item) => /dspy|gepa/i.test(item.title));
    expect(gepa).toBeDefined();
    expect(gepa?.status).toBe('broken');
    expect(gepa?.body).toMatch(/thinking/i);
  });

  it('says the demo pack carries no latency or token metrics', () => {
    const metrics = OPEN_WORK.find((item) => /metric/i.test(item.title));
    expect(metrics).toBeDefined();
    expect(metrics?.body).toMatch(/not been authorised|has not authorised/i);
  });

  it('never marks a broken item as done', () => {
    for (const item of OPEN_WORK) {
      expect(['broken', 'open', 'deferred']).toContain(item.status);
    }
  });
});
