/**
 * Formatting for the status board.
 *
 * Every function here has one job the board depends on: distinguishing "we
 * measured this and it is zero" from "we never measured this". `null` in means
 * an em dash out — never `0`, never `0.0%`, never `$0.0000`. A zero that was
 * actually measured formats normally, because that is a real result.
 *
 * Kept free of React so the rules are unit-testable without a DOM.
 */

/** The em dash the board uses wherever a figure does not exist. */
export const NO_VALUE = '—';

/**
 * `Double_MAR/2010/page_55.pdf` → `MAR · 2010 · p.55`.
 *
 * The dataset's report ids are paths: a `Single_`/`Double_` conversation-type
 * prefix, a ticker, a fiscal year, and a page file, sometimes with a `-1`
 * conversation suffix. A reader recognises the ticker and the year; the rest is
 * noise on a card. Anything that does not parse falls back to the raw id
 * rather than to a guess — an unrecognised filing shown under a wrong ticker
 * would be worse than an ugly one.
 */
export function formatFilingId(reportId: string): string {
  const match = /^(?:Single|Double)_([A-Za-z0-9.&-]+)\/(\d{4})\/page_(\d+)\.pdf(?:-\d+)?$/.exec(
    reportId,
  );
  if (!match) return reportId;
  const [, ticker, year, page] = match;
  return `${ticker} · ${year} · p.${page}`;
}

/** The ticker alone, for a tight label. Falls back to the whole id. */
export function tickerOf(reportId: string): string {
  const match = /^(?:Single|Double)_([A-Za-z0-9.&-]+)\//.exec(reportId);
  return match ? match[1] : reportId;
}

/** `0.771429` → `77.1%`. `null` → `—`. */
export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NO_VALUE;
  return `${(value * 100).toFixed(digits)}%`;
}

/** A signed percentage-point delta: `0.0049` → `+0.5pp`. */
export function formatPointsDelta(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NO_VALUE;
  const points = value * 100;
  const sign = points > 0 ? '+' : '';
  return `${sign}${points.toFixed(digits)}pp`;
}

/**
 * Latency, at the precision the number deserves: `3181.4` → `3.2s`,
 * `940` → `940ms`. Milliseconds below a second, seconds above it — nobody
 * reads "3181ms" as three seconds at a glance.
 */
export function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return NO_VALUE;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Money, at four decimals because a turn costs fractions of a cent and
 * rounding it to two would print `$0.00` for every turn this system has ever
 * served. Totals above a dollar drop to two decimals.
 */
export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NO_VALUE;
  if (Math.abs(value) >= 1) return `$${value.toFixed(2)}`;
  return `$${value.toFixed(4)}`;
}

/** `12` → `12`; `10800` → `10.8k`. Counts, never measurements. */
export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NO_VALUE;
  if (Math.abs(value) >= 10_000) return `${(value / 1000).toFixed(1)}k`;
  return String(Math.round(value));
}

/** `770` → `770 questions`, `1` → `1 question`. */
export function plural(n: number, one: string, many = `${one}s`): string {
  return `${formatCount(n)} ${n === 1 ? one : many}`;
}

// ---------------------------------------------------------------------------
// Sparklines
// ---------------------------------------------------------------------------

export interface SparkGeometry {
  width: number;
  height: number;
  /** Vertical inset so the stroke is not clipped at the extremes. */
  pad: number;
}

const DEFAULT_GEOMETRY: SparkGeometry = { width: 100, height: 18, pad: 2 };

/**
 * Turn an hourly series into SVG polyline point strings.
 *
 * Two rules encode the honesty contract:
 *
 *  - A `null` bucket is a hole, not a zero. Nulls break the line into separate
 *    polylines rather than being plotted on the floor, so an hour nothing was
 *    measured never reads as an hour of perfect latency.
 *  - Fewer than two real points returns `[]`. The caller must then say "not
 *    enough data" rather than draw something. A single point drawn as a flat
 *    line across the whole card is the exact lie this function exists to
 *    prevent.
 *
 * A genuinely flat run of measured values (twelve hours of zero errors) does
 * draw, at the baseline — that is a real observation, not an absence.
 */
export function sparkPolylines(
  values: Array<number | null | undefined>,
  geometry: Partial<SparkGeometry> = {},
): string[] {
  const { width, height, pad } = { ...DEFAULT_GEOMETRY, ...geometry };
  const real = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  if (real.length < 2) return [];

  const max = Math.max(...real);
  const min = Math.min(...real);
  const span = max - min;
  const step = values.length > 1 ? width / (values.length - 1) : width;
  const usable = height - pad * 2;

  const y = (v: number): number => {
    // A degenerate range has no shape to show. Sit a constant series on the
    // floor when it is all zero, and mid-height otherwise, so "flat and low"
    // and "flat and high" do not look identical.
    if (span === 0) return max === 0 ? height - pad : height / 2;
    return height - pad - ((v - min) / span) * usable;
  };

  const runs: string[] = [];
  let current: string[] = [];
  values.forEach((v, i) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      if (current.length > 1) runs.push(current.join(' '));
      current = [];
      return;
    }
    current.push(`${(i * step).toFixed(2)},${y(v).toFixed(2)}`);
  });
  if (current.length > 1) runs.push(current.join(' '));
  return runs;
}
