/**
 * The typed API surface the redesigned console talks to.
 *
 * `src/api.ts` is still the transport — it owns the api-base resolution, the
 * owner-token header, the SSE plumbing and `ApiError`. This module re-exports
 * it so feature code has one import, and adds the endpoints the later phases
 * need that the old console never called.
 *
 * Every shape here was read off `src/convfinqa/serving/routes/*.py`. Nothing is
 * invented: if a field is not in the response model, it is not in the type.
 */
export * from '../api';

import { ApiError, getApiBase } from '../api';

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
  return res.json() as Promise<T>;
}

function encodeRid(rid: string): string {
  // The FastAPI routes use {report_id:path}, so slashes survive encoding.
  return rid
    .split('/')
    .map((s) => encodeURIComponent(s))
    .join('/');
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/** `GET /reports/{report_id:path}` — declared after /questions and /document. */
export interface ReportSummary {
  report_id: string;
  n_questions: number;
  /** Truncated to 400 characters server-side. */
  doc_summary: string;
  /** "optimizer_train" | "never_seen" | "unknown". Never blend the first two. */
  split: string;
  in_demo_pack: boolean;
}

export function getReportSummary(reportId: string): Promise<ReportSummary> {
  return getJson<ReportSummary>(`/reports/${encodeRid(reportId)}`);
}

/** `GET /demo/reports` — the conversations the recorded pack can replay. */
export interface DemoReport {
  report_id: string;
  n_questions: number;
}

export function listDemoReports(): Promise<DemoReport[]> {
  return getJson<DemoReport[]>('/demo/reports');
}

// ---------------------------------------------------------------------------
// Admin reads the old console did not use
// ---------------------------------------------------------------------------

/** `GET /admin/experiments/{run_id}`. */
export interface RunRecord {
  run_id: string;
  run_name: string;
  kind: string;
  bundle_id: string;
  status: string;
  start_time: number;
  end_time: number;
  params: Record<string, string>;
  metrics: Record<string, number>;
  tags: Record<string, string>;
}

export function getExperimentRun(runId: string): Promise<RunRecord> {
  return getJson<RunRecord>(`/admin/experiments/${encodeURIComponent(runId)}`);
}

/**
 * `GET /admin/versions` — every version with a committed predictions CSV.
 *
 * Phase 1 changed this from `list[str]` to a row per version carrying both
 * accuracies. Both are here because they disagree: `exe_acc` is "did the final
 * number come out right" (~77%), `prog_acc` is "was the program the same shape
 * as gold" (~35%). The gap is real and expected — the pipeline answers a turn
 * from prior *answers* (`divide(132, 111)`) where gold re-derives from raw
 * values (`subtract(243, 111), divide(#0, 111)`) — so a surface showing the
 * second number owes the reader that explanation.
 *
 * Note what is NOT here: holdout accuracy. Splitting `optimizer_train` from
 * `never_seen` needs `/admin/experiments`, which computes both. Never present
 * `exe_acc` as a held-out figure.
 */
export interface VersionAccuracyRow {
  version: string;
  /** Execution accuracy over all 770 scored questions — seen and unseen mixed. */
  exe_acc: number;
  /** Program accuracy over the program turns only. */
  prog_acc: number;
  n_questions: number;
  n_program_turns: number;
  n_program_correct: number;
}

export function listVersions(): Promise<VersionAccuracyRow[]> {
  return getJson<VersionAccuracyRow[]>('/admin/versions');
}

/** `GET /admin/rules/variants` — the s7 rule-store variants present. */
export function listRuleVariants(): Promise<string[]> {
  return getJson<string[]>('/admin/rules/variants');
}

/**
 * `GET /traces/stats`. Without a trace store the server returns only the two
 * counts, so the latency/token fields are genuinely optional — not merely
 * "sometimes null".
 */
export interface TraceStats {
  n_turns: number;
  n_reports: number;
  avg_latency_ms?: number;
  total_tokens?: number;
}

export function getTraceStats(): Promise<TraceStats> {
  return getJson<TraceStats>('/traces/stats');
}

// ---------------------------------------------------------------------------
// Production metrics — Phase 1 adds the route; this client ships ahead of it
// ---------------------------------------------------------------------------

export type MetricsSource = 'serving' | 'demo' | 'eval';

/**
 * `GET /metrics/production`, one block per `source`.
 *
 * The grouping is the honesty contract, not a convenience: replay timing is
 * not latency. A turn recorded at 6.7s and replayed in 2s is a 6.7s turn, so
 * `serving`, `demo` and `eval` are counted separately and must never be summed
 * into one headline number. A card showing anything from `demo` or `eval` owes
 * the reader a "recorded in dev" source line.
 *
 * Typed against the live response from the Phase 1 route, not from prose.
 * Fields that are `null` mean "nothing measured", which is distinct from zero
 * and has to render as an em dash rather than as 0.
 */
export interface MetricsStat {
  p50: number | null;
  p95?: number | null;
  mean: number | null;
  n_measured: number;
}

export interface SourceMetrics {
  source: MetricsSource;
  n_turns: number;
  latency_ms: MetricsStat;
  tokens_per_turn: { p50: number | null; mean: number | null; total: number; n_measured: number };
  cost_usd: { per_turn: number | null; total: number; n_measured: number };
  accuracy: { accuracy: number | null; n_correct: number; n_scored: number };
  errors: {
    n_errors: number;
    error_rate: number | null;
    by_code: Record<string, number>;
  };
  series: Array<{
    hour: string;
    n_turns: number;
    n_errors: number;
    p50_latency_ms: number | null;
    cost_usd: number;
  }>;
}

export interface ProductionMetrics {
  generated_at: string;
  window_hours: number;
  n_turns_total: number;
  trace_capture_enabled: boolean;
  sources: Record<MetricsSource, SourceMetrics>;
}

/**
 * Returns `null` rather than throwing when the route is not there.
 *
 * Two ways it can be absent, and both have to be survivable: a plain 404, and
 * — because FastAPI serves the built SPA from a catch-all — a 200 carrying
 * index.html. Treating an HTML 200 as success would hand the UI a parse error
 * from deep inside a render instead of an empty card. Any other failure still
 * raises; hiding a 500 would be the same mistake in the other direction.
 */
export async function getProductionMetrics(): Promise<ProductionMetrics | null> {
  try {
    const res = await fetch(`${getApiBase()}/metrics/production`);
    if (res.status === 404) return null;
    if (!res.ok) throw new ApiError(`/metrics/production failed: ${res.status}`, res.status);
    if (!res.headers.get('content-type')?.includes('json')) return null;
    return (await res.json()) as ProductionMetrics;
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}
