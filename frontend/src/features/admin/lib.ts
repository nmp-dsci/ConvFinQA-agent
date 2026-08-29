/**
 * Pure helpers for the admin console.
 *
 * Everything here is deliberately React-free so the rules that matter — which
 * turn a filter keeps, what an absent metric is allowed to say, how a version's
 * two accuracies are kept apart — can be unit-tested without a DOM. The vitest
 * environment for this repo is `node`, so a helper that reached for `document`
 * would be untestable as well as wrong.
 *
 * Formatting of numbers lives in `features/landing/format.ts` and is imported,
 * not re-derived: `NO_VALUE`, `formatLatency`, `formatUsd` and `sparkPolylines`
 * encode the null rules the whole app is judged on, and a second copy of them
 * would be a second place for those rules to drift.
 */

import { NO_VALUE, formatCount, formatPercent } from '../landing/format';
import type { SourceMetrics, VersionAccuracyRow } from '../../lib/api';
import type { TraceSummary, VersionAccuracy } from '../../types';

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------

/** `2026-08-29T09:59:37Z` → `29 Aug 09:59`. Absent input → an em dash. */
export function formatStamp(iso: string | null | undefined): string {
  if (!iso) return NO_VALUE;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return NO_VALUE;
  const day = d.getDate();
  const month = d.toLocaleString('en-GB', { month: 'short' });
  const time = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  return `${day} ${month} ${time}`;
}

/**
 * `…37s ago`. Coarse on purpose: a trace list wants "2m ago", not a duration to
 * the millisecond, and precision nobody asked for reads as precision the data
 * does not have.
 */
export function relativeTime(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return NO_VALUE;
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return NO_VALUE;
  const seconds = Math.max(0, Math.round((now - then) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/** MLflow stores run times as epoch milliseconds. */
export function formatEpochMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms <= 0) return NO_VALUE;
  return formatStamp(new Date(ms).toISOString());
}

/** `1787876735137` → `1787876735163` = `26ms`. Both absent or zero → em dash. */
export function formatRunDuration(start: number, end: number): string {
  if (!Number.isFinite(start) || !Number.isFinite(end) || start <= 0 || end <= start) {
    return NO_VALUE;
  }
  const ms = end - start;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m`;
  return `${(ms / 3_600_000).toFixed(1)}h`;
}

// ---------------------------------------------------------------------------
// Trace filtering
// ---------------------------------------------------------------------------

export type CorrectnessFilter = 'all' | 'correct' | 'incorrect' | 'unscored';

export interface TraceFilter {
  /** Server-side already, but re-applied so a stale page cannot show a lie. */
  source: string;
  correctness: CorrectnessFilter;
  /** `''` = any. Matches `error_code` exactly; `'any'` means "has an error". */
  errorCode: string;
  sessionId: string;
  reportId: string;
  /** Free text across question and answer. */
  q: string;
}

export const EMPTY_TRACE_FILTER: TraceFilter = {
  source: '',
  correctness: 'all',
  errorCode: '',
  sessionId: '',
  reportId: '',
  q: '',
};

/**
 * `correct` arrives as SQLite's 0/1 through pydantic, and as `null` for a turn
 * with no gold answer. "Unscored" is a third state, not a synonym for wrong —
 * folding it into "incorrect" would quietly inflate the error count on a page
 * whose whole job is to be countable.
 */
export function matchesTraceFilter(row: TraceSummary, f: TraceFilter): boolean {
  if (f.source && row.source !== f.source) return false;

  if (f.correctness === 'correct' && row.correct !== true) return false;
  if (f.correctness === 'incorrect' && row.correct !== false) return false;
  if (f.correctness === 'unscored' && row.correct !== null && row.correct !== undefined) {
    return false;
  }

  const code = row.error_code ?? '';
  if (f.errorCode === 'any' && !code && !row.error) return false;
  if (f.errorCode && f.errorCode !== 'any' && code !== f.errorCode) return false;

  if (f.sessionId && !(row.session_id ?? '').includes(f.sessionId)) return false;
  if (f.reportId && !row.report_id.toLowerCase().includes(f.reportId.toLowerCase())) return false;

  if (f.q) {
    const needle = f.q.toLowerCase();
    const hay = `${row.question} ${row.answer ?? ''} ${row.gold_answer ?? ''}`.toLowerCase();
    if (!hay.includes(needle)) return false;
  }
  return true;
}

/** Error codes actually present in a page of traces, for the filter select. */
export function observedErrorCodes(rows: TraceSummary[]): string[] {
  const seen = new Set<string>();
  for (const row of rows) {
    const code = row.error_code ?? '';
    if (code) seen.add(code);
    else if (row.error) seen.add('unknown');
  }
  return [...seen].sort();
}

// ---------------------------------------------------------------------------
// Accuracy — the two-number rule, in one place
// ---------------------------------------------------------------------------

export interface VersionRow {
  version: string;
  /** All 770 scored questions, seen and unseen mixed. Never "held out". */
  overall: number | null;
  nQuestions: number | null;
  /** Only from `/admin/experiments`. `/admin/versions` cannot compute it. */
  holdout: number | null;
  holdoutN: number | null;
  progAcc: number | null;
  nProgramTurns: number | null;
  nProgramCorrect: number | null;
  isChampion: boolean;
}

/**
 * Join `/admin/versions` (execution + program accuracy) to
 * `/admin/experiments` (holdout) without ever averaging the two.
 *
 * They stay in separate fields all the way to the cell. `optimizer_train` and
 * `never_seen` are different populations, and a single blended "accuracy" is
 * exactly the claim this project exists not to make.
 */
export function joinVersionRows(
  versions: VersionAccuracyRow[] | undefined,
  holdouts: VersionAccuracy[] | undefined,
  champion: string | null,
): VersionRow[] {
  const byVersion = new Map<string, VersionRow>();

  for (const v of versions ?? []) {
    byVersion.set(v.version, {
      version: v.version,
      overall: v.exe_acc,
      nQuestions: v.n_questions,
      holdout: null,
      holdoutN: null,
      progAcc: v.prog_acc,
      nProgramTurns: v.n_program_turns,
      nProgramCorrect: v.n_program_correct,
      isChampion: v.version === champion,
    });
  }

  for (const h of holdouts ?? []) {
    const existing = byVersion.get(h.version);
    if (existing) {
      existing.holdout = h.holdout_accuracy;
      existing.holdoutN = h.holdout_n_questions;
      if (existing.overall === null) existing.overall = h.accuracy;
      continue;
    }
    byVersion.set(h.version, {
      version: h.version,
      overall: h.accuracy,
      nQuestions: h.n_questions,
      holdout: h.holdout_accuracy,
      holdoutN: h.holdout_n_questions,
      progAcc: null,
      nProgramTurns: null,
      nProgramCorrect: null,
      isChampion: h.version === champion,
    });
  }

  return [...byVersion.values()];
}

/**
 * The one-line explanation that must accompany program accuracy anywhere it
 * appears. ~35% against ~77% execution is not the system failing: the pipeline
 * answers a turn from prior conversation *answers* where gold re-derives from
 * raw values, so the same number arrives via a shorter program.
 */
export const PROG_ACC_CAVEAT =
  'Program accuracy is much lower than execution accuracy by design: the pipeline reuses prior ' +
  'answers in the conversation (divide(132, 111)) where gold re-derives from raw values ' +
  '(subtract(243, 111), divide(#0, 111)). Same answer, shorter program.';

// ---------------------------------------------------------------------------
// Metrics — absence, with a reason
// ---------------------------------------------------------------------------

export interface Absent {
  value: string;
  reason: string;
}

/**
 * Why a metric is missing, said in the terms of the deployment looking at it.
 *
 * The demo pack carries no latency or cost, so `n_measured: 0` is the *normal*
 * state of half this console rather than an outage — and "0 turns served" and
 * "turns served but never metered" are different facts a reader has to be able
 * to tell apart.
 */
export function absenceReason(
  metrics: SourceMetrics | null,
  what: 'latency' | 'cost' | 'tokens' | 'accuracy',
): string {
  if (!metrics) return 'no metrics endpoint on this deployment';
  if (metrics.n_turns === 0) return 'no turns in the last 24 h';
  if (what === 'accuracy') return 'no turn in this window carried a gold answer';
  return 'turns served but never metered — awaiting a metered eval run';
}

/** `serving` in a live deployment, `demo` in the replay one. Never summed. */
export function sourceNote(source: string, generatedAt: string | undefined): string {
  const stamp = generatedAt ? ` · read ${formatStamp(generatedAt)}` : '';
  if (source === 'demo') {
    return `source: demo replay — these turns were recorded in development and played back, so their timing is replay timing, not latency${stamp}`;
  }
  if (source === 'eval') {
    return `source: batch evaluation runs, not user traffic${stamp}`;
  }
  return `source: live serving turns from this process, last 24 h${stamp}`;
}

// ---------------------------------------------------------------------------
// Small display helpers
// ---------------------------------------------------------------------------

/** `Number` / `Program`, `Type I` / `Type II` and friends, safely. */
export function titleCase(value: string): string {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

/** A compact `n_correct / n_total` caption. */
export function fraction(correct: number | null, total: number | null): string {
  if (correct === null || total === null) return NO_VALUE;
  return `${formatCount(correct)} / ${formatCount(total)}`;
}

/** `0.771429` → `77.1%`, kept here so callers need one import, not two. */
export { formatPercent, formatCount, NO_VALUE };

/** Truncate a long rule or question for a table cell without hiding the fact. */
export function clip(text: string, max = 160): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1)}…`;
}

/**
 * The bundle fingerprint on one line. A version label is meaningless when every
 * model is an API; this is what an answer is actually attributable to.
 */
export function bundleLine(bundle: Partial<Record<string, unknown>> | undefined): string {
  if (!bundle) return NO_VALUE;
  const parts = [
    bundle.prompts_version && `prompts ${bundle.prompts_version}`,
    bundle.gepa_overlay && `gepa ${bundle.gepa_overlay}`,
    bundle.lm_mini && String(bundle.lm_mini),
    bundle.dataset_hash && `dataset ${bundle.dataset_hash}`,
    bundle.code_sha && `code ${bundle.code_sha}`,
  ].filter(Boolean);
  return parts.length ? parts.join(' · ') : NO_VALUE;
}
