/**
 * The reads the admin console needs that the shared client does not cover.
 *
 * `lib/api.ts` is the app-wide surface and is left alone — Phase 5 shares it.
 * What lives here is either a call whose parameters the shared wrapper fixes
 * (traces caps `limit` at 200, answers at 500) or a response whose real shape
 * is richer than the shared type says. Two of those are worth spelling out:
 *
 *  - `GET /traces/{id}` has no response model on the server, so it returns the
 *    whole row: `correct` comes back as SQLite's raw `0 | 1 | null`, not as a
 *    bool, and `input_tokens`, `output_tokens`, `bundle` and `capture` are
 *    present where `TraceSummary` has none of them.
 *  - `GET /traces/eval/{version}/{report_id}` is a different shape again — no
 *    `trace_id`, no timing, `source` pinned to `"eval"` — because it is
 *    reconstructed from a committed predictions CSV rather than recorded live.
 *
 * Typing those two as `TraceDetail` would have the viewer quietly render
 * `latency_ms` for a row that never had one.
 */

import { ApiError, getApiBase } from '../../api';
import type { AnswerRow, BundleSpec, StageMetrics, TraceSummary } from '../../types';

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`);
  if (!res.ok) {
    let message = `${path} failed: ${res.status}`;
    let code = '';
    try {
      const body = await res.json();
      const detail = body?.detail;
      if (typeof detail === 'string') message = detail;
      else if (detail && typeof detail === 'object') {
        message = String(detail.message ?? detail.reason ?? message);
        code = String(detail.code ?? '');
      }
    } catch {
      // Non-JSON body — keep the status-based message.
    }
    throw new ApiError(message, res.status, code);
  }
  // FastAPI serves the built SPA from a catch-all, so a path that does not
  // resolve comes back as HTML with status 200 rather than as a 404. Parsing
  // that as JSON would surface as a render-time crash instead of an empty
  // panel, which is the harder failure to diagnose.
  if (!res.headers.get('content-type')?.includes('json')) {
    throw new ApiError(`${path} did not return JSON — is the backend running?`, res.status);
  }
  return res.json() as Promise<T>;
}

function encodeRid(rid: string): string {
  return rid
    .split('/')
    .map((s) => encodeURIComponent(s))
    .join('/');
}

// ---------------------------------------------------------------------------
// Answers
// ---------------------------------------------------------------------------

export interface AnswersQuery {
  reportId?: string;
  onlyDisagreements?: boolean;
  /** Server allows up to 2000; the full scored set is 770 rows. */
  limit?: number;
}

export function getAnswers(opts: AnswersQuery = {}): Promise<AnswerRow[]> {
  const params = new URLSearchParams();
  if (opts.reportId) params.set('report_id', opts.reportId);
  if (opts.onlyDisagreements) params.set('only_disagreements', 'true');
  params.set('limit', String(opts.limit ?? 2000));
  return getJson<AnswerRow[]>(`/eval/answers?${params.toString()}`);
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

export interface TracesQuery {
  reportId?: string;
  sessionId?: string;
  source?: string;
  /** Server maximum is 500. */
  limit?: number;
  offset?: number;
}

export function listTraces(opts: TracesQuery = {}): Promise<TraceSummary[]> {
  const params = new URLSearchParams();
  if (opts.reportId) params.set('report_id', opts.reportId);
  if (opts.sessionId) params.set('session_id', opts.sessionId);
  if (opts.source) params.set('source', opts.source);
  params.set('limit', String(Math.min(opts.limit ?? 300, 500)));
  params.set('offset', String(opts.offset ?? 0));
  // No trailing slash: `/traces/` redirects with a 307 that costs a round trip.
  return getJson<TraceSummary[]>(`/traces?${params.toString()}`);
}

/** One captured stage. Every field is optional — a stage may be skipped. */
export interface StageCapture {
  input?: unknown;
  output?: Record<string, unknown> | null;
  reasoning?: string;
  metrics?: StageMetrics;
  trajectory?: Array<Record<string, unknown>>;
}

export type StageName = 'triage' | 'preprocess' | 'retriever' | 'calculator';

export const STAGES: StageName[] = ['triage', 'preprocess', 'retriever', 'calculator'];

/** `GET /traces/{trace_id}` — the raw row, not a `TraceSummary`. */
export interface LiveTraceDetail {
  trace_id: string;
  created_at: string;
  source: string;
  session_id: string | null;
  report_id: string;
  turn_index: number;
  question: string;
  answer: string | null;
  program: string | null;
  gold_answer: string | null;
  /** SQLite INTEGER, unvalidated on this route: `0 | 1 | null`, not a bool. */
  correct: number | null;
  bundle_id: string | null;
  bundle: Partial<BundleSpec>;
  latency_ms: number | null;
  total_tokens: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
  error: string | null;
  error_code: string | null;
  /** `history_text` is a sibling of the four stages inside `capture`. */
  capture: Partial<Record<StageName, StageCapture | null>> & { history_text?: string };
}

export function getLiveTrace(traceId: string): Promise<LiveTraceDetail> {
  return getJson<LiveTraceDetail>(`/traces/${encodeURIComponent(traceId)}`);
}

/** `GET /traces/eval/{version}/{report_id}` — reconstructed from a CSV. */
export interface EvalTraceDetail {
  version: string;
  report_id: string;
  turn_index: number;
  question: string;
  answer: string;
  program: string;
  gold_answer: string;
  correct: boolean;
  history_text: string;
  capture: Record<StageName, StageCapture | null>;
  source: 'eval';
}

export function getEvalTrace(
  version: string,
  reportId: string,
  turnIndex: number,
): Promise<EvalTraceDetail> {
  return getJson<EvalTraceDetail>(
    `/traces/eval/${encodeURIComponent(version)}/${encodeRid(reportId)}?turn_index=${turnIndex}`,
  );
}
