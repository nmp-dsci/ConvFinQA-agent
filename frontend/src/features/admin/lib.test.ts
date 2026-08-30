import { describe, expect, it } from 'vitest';
import {
  EMPTY_TRACE_FILTER,
  absenceReason,
  bundleLine,
  clip,
  formatEpochMs,
  formatRunDuration,
  formatStamp,
  fraction,
  joinVersionRows,
  matchesTraceFilter,
  observedErrorCodes,
  relativeTime,
  sourceNote,
} from './lib';
import type { TraceFilter } from './lib';
import type { SourceMetrics, VersionAccuracyRow } from '../../lib/api';
import type { TraceSummary, VersionAccuracy } from '../../types';

const NO_VALUE = '—';

function trace(patch: Partial<TraceSummary> = {}): TraceSummary {
  return {
    trace_id: 't1',
    created_at: '2026-08-29T09:00:00+00:00',
    source: 'serving',
    session_id: 'abc-123',
    report_id: 'Double_MAR/2010/page_55.pdf',
    turn_index: 0,
    question: 'what was the cash from operations in 2008?',
    answer: '641',
    program: '',
    gold_answer: '641',
    correct: true,
    bundle_id: 'b1',
    latency_ms: 3000,
    total_tokens: 4000,
    cost_usd: 0.0014,
    error: '',
    error_code: '',
    ...patch,
  };
}

function filter(patch: Partial<TraceFilter> = {}): TraceFilter {
  return { ...EMPTY_TRACE_FILTER, ...patch };
}

// ---------------------------------------------------------------------------

describe('time formatting', () => {
  it('renders an absent stamp as an em dash rather than as an epoch', () => {
    expect(formatStamp(null)).toBe(NO_VALUE);
    expect(formatStamp('')).toBe(NO_VALUE);
    expect(formatStamp('not a date')).toBe(NO_VALUE);
  });

  it('coarsens relative time as it recedes', () => {
    const now = Date.parse('2026-08-29T12:00:00Z');
    expect(relativeTime('2026-08-29T11:59:30Z', now)).toBe('30s ago');
    expect(relativeTime('2026-08-29T11:30:00Z', now)).toBe('30m ago');
    expect(relativeTime('2026-08-29T06:00:00Z', now)).toBe('6h ago');
    expect(relativeTime('2026-08-25T12:00:00Z', now)).toBe('4d ago');
  });

  it('treats a zero or missing epoch as absent, not as 1970', () => {
    expect(formatEpochMs(0)).toBe(NO_VALUE);
    expect(formatEpochMs(null)).toBe(NO_VALUE);
  });

  it('refuses to invent a duration from a run with no end time', () => {
    expect(formatRunDuration(1787876735137, 0)).toBe(NO_VALUE);
    expect(formatRunDuration(1787876735137, 1787876735137)).toBe(NO_VALUE);
    expect(formatRunDuration(1000, 1026)).toBe('26ms');
    expect(formatRunDuration(0, 90_000)).toBe(NO_VALUE);
    expect(formatRunDuration(1000, 91_000)).toBe('2m');
  });
});

// ---------------------------------------------------------------------------

describe('matchesTraceFilter', () => {
  it('keeps everything under an empty filter', () => {
    expect(matchesTraceFilter(trace(), filter())).toBe(true);
  });

  it('separates unscored turns from incorrect ones', () => {
    const unscored = trace({ correct: null, gold_answer: null });
    const wrong = trace({ correct: false });

    expect(matchesTraceFilter(unscored, filter({ correctness: 'incorrect' }))).toBe(false);
    expect(matchesTraceFilter(unscored, filter({ correctness: 'unscored' }))).toBe(true);
    expect(matchesTraceFilter(wrong, filter({ correctness: 'incorrect' }))).toBe(true);
    expect(matchesTraceFilter(wrong, filter({ correctness: 'unscored' }))).toBe(false);
    expect(matchesTraceFilter(trace(), filter({ correctness: 'correct' }))).toBe(true);
  });

  it('matches "errors only" on a bare error with no classified code', () => {
    const classified = trace({ error: 'boom', error_code: 'timeout' });
    const unclassified = trace({ error: 'boom', error_code: '' });
    const clean = trace();

    expect(matchesTraceFilter(classified, filter({ errorCode: 'any' }))).toBe(true);
    expect(matchesTraceFilter(unclassified, filter({ errorCode: 'any' }))).toBe(true);
    expect(matchesTraceFilter(clean, filter({ errorCode: 'any' }))).toBe(false);
    expect(matchesTraceFilter(classified, filter({ errorCode: 'timeout' }))).toBe(true);
    expect(matchesTraceFilter(classified, filter({ errorCode: 'rate_limited' }))).toBe(false);
  });

  it('filters by source, session, report and free text', () => {
    expect(matchesTraceFilter(trace(), filter({ source: 'demo' }))).toBe(false);
    expect(matchesTraceFilter(trace({ source: 'demo' }), filter({ source: 'demo' }))).toBe(true);
    expect(matchesTraceFilter(trace(), filter({ sessionId: 'abc' }))).toBe(true);
    expect(matchesTraceFilter(trace(), filter({ sessionId: 'zzz' }))).toBe(false);
    expect(matchesTraceFilter(trace(), filter({ reportId: 'mar/2010' }))).toBe(true);
    expect(matchesTraceFilter(trace(), filter({ q: 'CASH FROM' }))).toBe(true);
    expect(matchesTraceFilter(trace(), filter({ q: 'inventories' }))).toBe(false);
  });
});

describe('observedErrorCodes', () => {
  it('reports an unclassified error as "unknown" and never duplicates', () => {
    const codes = observedErrorCodes([
      trace(),
      trace({ error: 'x', error_code: 'timeout' }),
      trace({ error: 'y', error_code: '' }),
      trace({ error: 'z', error_code: 'timeout' }),
    ]);
    expect(codes).toEqual(['timeout', 'unknown']);
  });
});

// ---------------------------------------------------------------------------

describe('joinVersionRows', () => {
  const versions: VersionAccuracyRow[] = [
    {
      version: 'v1',
      exe_acc: 0.72987,
      prog_acc: 0.346473,
      n_questions: 770,
      n_program_turns: 482,
      n_program_correct: 167,
    },
    {
      version: 'v2',
      exe_acc: 0.771429,
      prog_acc: 0.352697,
      n_questions: 770,
      n_program_turns: 482,
      n_program_correct: 170,
    },
  ];

  const holdouts: VersionAccuracy[] = [
    {
      version: 'v1',
      accuracy: 0.72987,
      n_questions: 770,
      holdout_accuracy: 0.728155,
      holdout_n_questions: 309,
      slices: {},
    },
    {
      version: 'v2',
      accuracy: 0.771429,
      n_questions: 770,
      holdout_accuracy: 0.776699,
      holdout_n_questions: 309,
      slices: {},
    },
  ];

  it('keeps overall and holdout in separate fields — never one blended figure', () => {
    const rows = joinVersionRows(versions, holdouts, 'v2');
    const v2 = rows.find((r) => r.version === 'v2')!;

    expect(v2.overall).toBe(0.771429);
    expect(v2.nQuestions).toBe(770);
    expect(v2.holdout).toBe(0.776699);
    expect(v2.holdoutN).toBe(309);
    // The two are different populations; no field is their mean.
    expect(rows.every((r) => r.overall !== r.holdout || r.holdout === null)).toBe(true);
  });

  it('marks exactly the champion', () => {
    const rows = joinVersionRows(versions, holdouts, 'v2');
    expect(rows.filter((r) => r.isChampion).map((r) => r.version)).toEqual(['v2']);
  });

  it('leaves holdout null when the experiments payload is absent', () => {
    const rows = joinVersionRows(versions, undefined, null);
    expect(rows.every((r) => r.holdout === null && r.holdoutN === null)).toBe(true);
  });

  it('still lists a version that only the experiments payload knows about', () => {
    const rows = joinVersionRows([], holdouts, 'v2');
    expect(rows.map((r) => r.version)).toEqual(['v1', 'v2']);
    expect(rows[0].progAcc).toBeNull();
  });
});

// ---------------------------------------------------------------------------

describe('absenceReason', () => {
  const base: SourceMetrics = {
    source: 'demo',
    n_turns: 0,
    latency_ms: { p50: null, p95: null, mean: null, n_measured: 0 },
    tokens_per_turn: { p50: null, mean: null, total: 0, n_measured: 0 },
    cost_usd: { per_turn: null, total: 0, n_measured: 0 },
    accuracy: { accuracy: null, n_correct: 0, n_scored: 0 },
    errors: { n_errors: 0, error_rate: null, by_code: {} },
    series: [],
  };

  it('distinguishes "nothing happened" from "happened but was never metered"', () => {
    expect(absenceReason(base, 'latency')).toMatch(/no turns/);
    expect(absenceReason({ ...base, n_turns: 17 }, 'latency')).toMatch(/never metered/);
  });

  it('explains a missing accuracy as missing gold, not as a failure', () => {
    expect(absenceReason({ ...base, n_turns: 17 }, 'accuracy')).toMatch(/gold/);
  });

  it('says so when there is no metrics endpoint at all', () => {
    expect(absenceReason(null, 'cost')).toMatch(/no metrics endpoint/);
  });
});

describe('sourceNote', () => {
  it('says out loud that replay timing is not latency', () => {
    expect(sourceNote('demo', undefined)).toMatch(/replay timing, not latency/);
    expect(sourceNote('serving', undefined)).toMatch(/live serving turns/);
    expect(sourceNote('eval', undefined)).toMatch(/not user traffic/);
  });
});

// ---------------------------------------------------------------------------

describe('small helpers', () => {
  it('clips only what is too long, and marks the clip', () => {
    expect(clip('short', 10)).toBe('short');
    expect(clip('abcdefghijkl', 6)).toBe('abcde…');
  });

  it('renders a fraction only when both halves exist', () => {
    expect(fraction(170, 482)).toBe('170 / 482');
    expect(fraction(null, 482)).toBe(NO_VALUE);
  });

  it('builds a bundle line from whatever fields are present', () => {
    expect(bundleLine(undefined)).toBe(NO_VALUE);
    expect(bundleLine({})).toBe(NO_VALUE);
    expect(bundleLine({ prompts_version: 'v2', code_sha: 'abc1234' })).toBe(
      'prompts v2 · code abc1234',
    );
  });
});
