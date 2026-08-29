import { describe, expect, it } from 'vitest';
import {
  NO_VALUE,
  formatCount,
  formatFilingId,
  formatLatency,
  formatPercent,
  formatPointsDelta,
  formatUsd,
  sparkPolylines,
  tickerOf,
} from './format';

describe('formatFilingId', () => {
  it('reduces a dataset report id to ticker, year, page', () => {
    expect(formatFilingId('Double_MAR/2010/page_55.pdf')).toBe('MAR · 2010 · p.55');
    expect(formatFilingId('Single_VLO/2011/page_126.pdf-1')).toBe('VLO · 2011 · p.126');
  });

  it('falls back to the raw id rather than guessing', () => {
    expect(formatFilingId('something-else')).toBe('something-else');
    expect(tickerOf('something-else')).toBe('something-else');
  });
});

describe('null is never zero', () => {
  it('renders an em dash for every absent figure', () => {
    expect(formatPercent(null)).toBe(NO_VALUE);
    expect(formatLatency(null)).toBe(NO_VALUE);
    expect(formatUsd(null)).toBe(NO_VALUE);
    expect(formatCount(null)).toBe(NO_VALUE);
    expect(formatPointsDelta(undefined)).toBe(NO_VALUE);
  });

  it('still renders a measured zero, which is a real result', () => {
    expect(formatPercent(0)).toBe('0.0%');
    expect(formatCount(0)).toBe('0');
    expect(formatUsd(0)).toBe('$0.0000');
  });
});

describe('number formats', () => {
  it('formats accuracy from the live /admin/versions shape', () => {
    expect(formatPercent(0.771429)).toBe('77.1%');
    expect(formatPercent(0.352697)).toBe('35.3%');
  });

  it('switches latency units at a second', () => {
    expect(formatLatency(940)).toBe('940ms');
    expect(formatLatency(3181.4)).toBe('3.2s');
  });

  it('keeps four decimals on sub-dollar costs so a turn is not $0.00', () => {
    expect(formatUsd(0.001791)).toBe('$0.0018');
    expect(formatUsd(3.125)).toBe('$3.13');
  });

  it('abbreviates large counts only', () => {
    expect(formatCount(12)).toBe('12');
    expect(formatCount(770)).toBe('770');
    expect(formatCount(10_800)).toBe('10.8k');
  });

  it('signs a percentage-point delta', () => {
    expect(formatPointsDelta(0.048544)).toBe('+4.9pp');
    expect(formatPointsDelta(-0.042071)).toBe('-4.2pp');
  });
});

describe('sparkPolylines', () => {
  it('refuses to draw when fewer than two buckets were measured', () => {
    expect(sparkPolylines([null, null, null])).toEqual([]);
    expect(sparkPolylines([null, 5, null])).toEqual([]);
  });

  it('breaks the line at unmeasured buckets instead of plotting zeros', () => {
    const runs = sparkPolylines([1, 2, null, null, 3, 4]);
    expect(runs).toHaveLength(2);
    expect(runs[0].split(' ')).toHaveLength(2);
    expect(runs[1].split(' ')).toHaveLength(2);
  });

  it('draws a measured all-zero series on the floor, not mid-height', () => {
    const [run] = sparkPolylines([0, 0, 0, 0], { width: 100, height: 18, pad: 2 });
    expect(run).toBe('0.00,16.00 33.33,16.00 66.67,16.00 100.00,16.00');
  });

  it('spans the full height for a real range', () => {
    const [run] = sparkPolylines([0, 10], { width: 100, height: 18, pad: 2 });
    expect(run).toBe('0.00,16.00 100.00,2.00');
  });
});
