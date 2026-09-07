import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { JudgeTable } from './Runtimes';
import { judgeHeadline, judgeRows } from './runtimeStory';
import type { JudgeSplitMetrics, JudgeSummary } from './api';

/**
 * s12: the judge's rows are read from `story.json → judge`; the champion's
 * test row is the headline and the optimise split never appears (it is
 * balanced 50/50, so its coverage means nothing at production prevalence).
 */

function metrics(over: Partial<JudgeSplitMetrics> = {}): JudgeSplitMetrics {
  return {
    n: 349,
    n_wrong: 33,
    accuracy: 0.9054,
    error_target: 0.01,
    coverage: 0.85,
    n_high: 297,
    n_high_wrong: 2,
    high_band_accuracy: 0.9933,
    high_band_error: 0.0067,
    high_band_error_upper95: 0.021,
    failure_capture: 0.939,
    n_failures_caught: 31,
    n_failures_missed: 2,
    false_alarm_rate: 0.16,
    n_false_alarms: 50,
    meets_target: true,
    auroc: 0.91,
    brier: 0.08,
    ece: 0.04,
    coverage_at_target: 0.8,
    threshold_at_target: 0.9,
    ...over,
  };
}

const SUMMARY: JudgeSummary = {
  dataset: { name: 'judge_v1', runtime_version: 'sdk_v1' },
  champion: 'judge_j2',
  runtime_version: 'sdk_v1',
  error_target: 0.01,
  versions: [
    { version: 'judge_j1', splits: { optimise: metrics(), calibrate: metrics({ coverage: 0.7 }) } },
    { version: 'judge_j2', splits: { calibrate: metrics({ coverage: 0.8 }), test: metrics() } },
  ],
  gates: [],
};

describe('judgeRows', () => {
  it('lists calibrate and test rows only, champion flagged', () => {
    const rows = judgeRows(SUMMARY);
    expect(rows.map((r) => `${r.version}:${r.split}`)).toEqual([
      'judge_j1:calibrate',
      'judge_j2:calibrate',
      'judge_j2:test',
    ]);
    expect(rows.filter((r) => r.isChampion).map((r) => r.version)).toEqual(['judge_j2', 'judge_j2']);
  });

  it('headlines the champion on the test split', () => {
    const head = judgeHeadline(SUMMARY);
    expect(head?.version).toBe('judge_j2');
    expect(head?.split).toBe('test');
    expect(head?.coverage).toBe(0.85);
    expect(judgeHeadline(null)).toBeNull();
    expect(judgeHeadline({ ...SUMMARY, champion: 'judge_j1' })).toBeNull();
  });
});

describe('JudgeTable', () => {
  it('renders the bound, the capture and the target verdict', () => {
    const html = renderToStaticMarkup(<JudgeTable rows={judgeRows(SUMMARY)} target={0.01} />);
    expect(html).toContain('data-testid="judge-table"');
    expect(html).toContain('99.33%');
    expect(html).toContain('≤ 2.10%');
    expect(html).toContain('(31/33)');
    expect(html).toContain('≤ 1% met');
    expect(html).toContain('data-champion="true"');
  });

  it('says not yet run when nothing has been scored', () => {
    const html = renderToStaticMarkup(<JudgeTable rows={[]} target={0.01} />);
    expect(html).toContain('not yet run');
  });
});
