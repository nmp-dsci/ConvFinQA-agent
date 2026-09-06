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

export interface DatasetRow {
  split: string;
  report_id: string;
  turn_index: number;
  question: string;
  gold_answer: string;
  gold_program: string;
  turn_type: string;
  conv_type: string;
  /**
   * Per-subagent gold, derived from the gold program and answer rather than
   * labelled. This is the same derivation the loop's attribution rule uses, so
   * showing it is how a disputed attribution gets checked against gold by a
   * human instead of argued about.
   */
  expected_triage: string;
  expected_skeleton: string[];
  expected_operands: string[];
  expected_answer: string;
}

export function getDataset(split: string): Promise<DatasetRow[]> {
  return getJson<DatasetRow[]>(`/eval/dataset?split=${encodeURIComponent(split)}`);
}

export interface CampaignExperiment {
  label: string;
  campaign: string;
  target_agent: string;
  baseline_version: string;
  candidate_version: string;
  promoted: boolean;
  at: number | null;
  accuracy_delta: number | null;
  cluster_p_one_sided: number | null;
  delta_ci_lo: number | null;
  delta_ci_hi: number | null;
  n_compared: number | null;
  fixed: number | null;
  broken: number | null;
  accuracy_baseline: number | null;
  accuracy_candidate: number | null;
  panel_baseline: Record<string, number | null>;
  panel_candidate: Record<string, number | null>;
  summary_of_changes: string;
  rationale: string;
  diff: string;
  /**
   * The Agent SDK arm's target is a *failure class* inside the single prompt,
   * not a subagent, so `target_agent` and `target_class` carry the same string
   * for an SDK row and only `target_agent` is set for a pipeline row. `edits`
   * are the tagged changes made inside that one prompt; a pipeline row has
   * none, and an SDK row recorded before edits were logged has none either.
   */
  target_class?: string;
  runtime?: string;
  edits?: Array<Record<string, unknown>>;
}

export interface CampaignSummary {
  name: string;
  n_experiments: number;
  n_promoted: number;
  n_remaining: number;
  blocked_agents: string[];
  complete: boolean;
  /**
   * The experiment cap this campaign is judged against — 5 for the pipeline, 2
   * for the SDK arm. Optional because a server built before it served the field
   * omits it, and a page must then say it does not know the cap rather than
   * assume the pipeline's.
   */
  cap?: number;
  runtime?: string;
}

/**
 * One runtime's arm of the cross-runtime comparison, on the fixed gate split.
 *
 * Every field is nullable because the story keeps an arm's keys present and
 * empty until a run of that arm exists — which is what lets the Runtimes page
 * print "not yet run" instead of a zero it never measured.
 */
export interface RuntimeArm {
  version: string | null;
  run_name: string | null;
  accuracy: number | null;
  by_turn_type: { number: number | null; program: number | null } | null;
  panel: Record<string, number | null> | null;
  cost: number | null;
  wall: number | null;
  /**
   * Execution accuracy's check: how often the *program* matched gold. Derived
   * by the server from the committed predictions CSV named by `run_name`, so it
   * is absent rather than zero when that CSV is not on disk.
   */
  program_accuracy?: number | null;
  /** The sdk arm's model id (`sdk_model` param); absent on the pipeline arm. */
  model?: string | null;
}

/** One turn-type slice of the cross-runtime gate. */
export interface RuntimeSlice {
  n: number | null;
  baseline_accuracy: number | null;
  candidate_accuracy: number | null;
  delta_pp: number | null;
  fixed: number | null;
  broken: number | null;
  n_flip_clusters?: number | null;
  cluster_z?: number | null;
  cluster_p_one_sided: number | null;
  mcnemar_p_one_sided?: number | null;
}

/** The paired verdict between the two runtimes, aggregate and per slice. */
export interface RuntimeGate {
  delta_pp: number | null;
  p_value: number | null;
  ci: Array<number | null> | null;
  fixed?: number | null;
  broken?: number | null;
  candidate_version?: string | null;
  promoted?: boolean | null;
  gate_id?: string | null;
  by_turn_type?: Partial<Record<'number' | 'program', RuntimeSlice>> | null;
}

export interface RuntimeComparison {
  pipeline: RuntimeArm;
  agent_sdk: RuntimeArm;
  gate: RuntimeGate | null;
}

/** The sdk champion scored on one model — an arm plus the model it ran on. */
export interface SdkModelArm extends RuntimeArm {
  model: string | null;
  n_scored?: number | null;
}

/**
 * One model's paired verdict against the reference model's run of the same
 * prompt: the gate's own test, computed from the two committed CSVs. Every
 * statistic is null when either CSV is missing or incomplete.
 */
export interface SdkModelPair {
  baseline_model: string;
  candidate_model: string;
  baseline_run: string | null;
  candidate_run: string | null;
  n_compared: number | null;
  delta_pp: number | null;
  cluster_z?: number | null;
  /** One-sided, in the direction of `delta_pp` — not the gate's towards-better p. */
  p_value: number | null;
  ci: Array<number | null> | null;
  fixed: number | null;
  broken: number | null;
  significant: boolean | null;
  by_turn_type?: Partial<Record<'number' | 'program', RuntimeSlice>> | null;
}

/**
 * One prompt, several models: the half of the cross-runtime confound that can
 * be measured. A scoring pass, not an experiment — nothing here promotes.
 */
export interface SdkModelComparison {
  version: string | null;
  reference_model: string;
  models: SdkModelArm[];
  pairs: SdkModelPair[];
}

export interface ChampionPoint {
  version: string;
  at: number | null;
  accuracy: number | null;
  panel: Record<string, number | null>;
  moved_by: string | null;
  target_agent: string | null;
}

export interface CampaignsResponse {
  champion: string | null;
  /**
   * The champion's accuracy on the fixed gate split — the figure the campaign
   * optimises and gates against, and what the status board leads with. Null
   * until a gate run for the champion exists in the story.
   */
  champion_accuracy: number | null;
  champion_panel: Record<string, number | null>;
  rule: string;
  generated_at: string;
  split: Record<string, unknown>;
  campaigns: CampaignSummary[];
  experiments: CampaignExperiment[];
  champion_track: ChampionPoint[];
  /**
   * The Agent SDK arm (s10). Additive on the server, so these are optional
   * here: a backend that predates the experiment serves the same route without
   * them and every existing view keeps working.
   */
  sdk_champion?: string | null;
  runtime_comparison?: RuntimeComparison | null;
  sdk_model_comparison?: SdkModelComparison | null;
  sdk_campaigns?: CampaignSummary[];
  sdk_experiments?: CampaignExperiment[];
}

export function getCampaigns(campaign = ''): Promise<CampaignsResponse> {
  const q = campaign ? `?campaign=${encodeURIComponent(campaign)}` : '';
  return getJson<CampaignsResponse>(`/eval/campaigns${q}`);
}
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
