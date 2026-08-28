import { fetchEventSource } from '@microsoft/fetch-event-source';
import type {
  AnswerRow,
  ComparisonResult,
  DemoQuestion,
  EvalSummary,
  ExperimentsPayload,
  Health,
  PredRow,
  RegistryPayload,
  ReportDocument,
  ReportQuestion,
  ResearchStatus,
  RulesPayload,
  SplitSummary,
  SSEEvent,
  TraceDetail,
  TraceSummary,
} from './types';

const API_BASE_KEY = 'convfinqa.apiBase';
const OWNER_TOKEN_KEY = 'convfinqa.ownerToken';

export function getApiBase(): string {
  if (typeof window === 'undefined') return '';
  try {
    return window.localStorage.getItem(API_BASE_KEY) ?? '';
  } catch {
    return '';
  }
}

/**
 * The owner token is a local convenience, never a security boundary — the
 * server decides. It lives in localStorage so a dev session survives a reload;
 * on the demo there is nothing it can unlock, because admin writes are refused
 * by mode as well as by token.
 */
export function getOwnerToken(): string {
  if (typeof window === 'undefined') return '';
  try {
    return window.localStorage.getItem(OWNER_TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
}

export function setOwnerToken(token: string): void {
  try {
    if (token) window.localStorage.setItem(OWNER_TOKEN_KEY, token);
    else window.localStorage.removeItem(OWNER_TOKEN_KEY);
  } catch {
    // Private browsing — the token simply does not persist.
  }
}

function url(path: string): string {
  return `${getApiBase()}${path}`;
}

function encodeRid(rid: string): string {
  // FastAPI route uses {rid:path}, so encoding still resolves correctly.
  return rid
    .split('/')
    .map((s) => encodeURIComponent(s))
    .join('/');
}

export class SessionGoneError extends Error {
  constructor() {
    super('Session no longer exists on server (TTL evicted or never created).');
    this.name = 'SessionGoneError';
  }
}

/** An error the server described with a stable `code` the UI can act on. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code = '',
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(url(path));
  if (!res.ok) throw await toApiError(res, path);
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown, owner = false): Promise<T> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (owner) headers['x-owner-token'] = getOwnerToken();
  const res = await fetch(url(path), {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await toApiError(res, path);
  return res.json() as Promise<T>;
}

async function toApiError(res: Response, path: string): Promise<ApiError> {
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
  return new ApiError(message, res.status, code);
}

// ---------------------------------------------------------------------------
// Health & mode
// ---------------------------------------------------------------------------

export function getHealth(): Promise<Health> {
  return getJson<Health>('/healthz');
}

// ---------------------------------------------------------------------------
// Reports & chat
// ---------------------------------------------------------------------------

export async function listReports(q = '', limit = 500): Promise<string[]> {
  const params = new URLSearchParams();
  if (q) params.set('q', q);
  params.set('limit', String(limit));
  return getJson<string[]>(`/reports?${params.toString()}`);
}

export function getQuestions(reportId: string): Promise<ReportQuestion[]> {
  return getJson<ReportQuestion[]>(`/reports/${encodeRid(reportId)}/questions`);
}

export function getDocument(reportId: string): Promise<ReportDocument> {
  return getJson<ReportDocument>(`/reports/${encodeRid(reportId)}/document`);
}

export function getDemoQuestions(reportId: string): Promise<DemoQuestion[]> {
  return getJson<DemoQuestion[]>(`/demo/questions?report_id=${encodeURIComponent(reportId)}`);
}

export async function createSession(reportId: string): Promise<string> {
  const body = await postJson<{ session_id: string }>('/sessions', { report_id: reportId });
  return body.session_id;
}

export interface SessionInfo {
  session_id: string;
  report_id: string;
  n_turns: number;
  history: Array<{ question: string; answer: string; report_id: string }>;
}

export async function getSession(sessionId: string): Promise<SessionInfo | null> {
  const res = await fetch(url(`/sessions/${sessionId}`));
  if (res.status === 404) return null;
  if (!res.ok) throw await toApiError(res, 'getSession');
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  // Swallow 404 — the server may have already evicted via TTL.
  const res = await fetch(url(`/sessions/${sessionId}`), { method: 'DELETE' });
  if (!res.ok && res.status !== 404) {
    throw new Error(`deleteSession failed: ${res.status}`);
  }
}

export interface StreamAskArgs {
  sessionId: string;
  question: string;
  signal: AbortSignal;
  onEvent: (event: SSEEvent) => void;
}

export async function streamAsk(args: StreamAskArgs): Promise<void> {
  const { sessionId, question, signal, onEvent } = args;
  await fetchEventSource(url(`/sessions/${sessionId}/ask/stream`), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question }),
    signal,
    openWhenHidden: true,
    async onopen(res) {
      if (res.status === 404) throw new SessionGoneError();
      if (!res.ok) throw new Error(`streamAsk HTTP ${res.status}`);
    },
    onmessage(ev) {
      if (!ev.data) return;
      try {
        onEvent(JSON.parse(ev.data) as SSEEvent);
      } catch {
        // Malformed frame — drop silently rather than abort the stream.
      }
    },
    onerror(err) {
      // Re-throw to stop the library's automatic retry; the caller decides.
      throw err;
    },
  });
}

// ---------------------------------------------------------------------------
// Eval, splits, answers
// ---------------------------------------------------------------------------

export function listEvalRuns(): Promise<string[]> {
  return getJson<string[]>('/eval/runs');
}

export function getEvalSummary(runName: string): Promise<EvalSummary> {
  return getJson<EvalSummary>(`/eval/runs/${encodeURIComponent(runName)}/summary`);
}

export function getEvalPredictions(runName: string, model: string): Promise<PredRow[]> {
  const params = new URLSearchParams({ model });
  return getJson<PredRow[]>(`/eval/runs/${encodeURIComponent(runName)}/predictions?${params}`);
}

export function getSplits(): Promise<SplitSummary[]> {
  return getJson<SplitSummary[]>('/eval/splits');
}

export function getAnswers(reportId = '', onlyDisagreements = false): Promise<AnswerRow[]> {
  const params = new URLSearchParams();
  if (reportId) params.set('report_id', reportId);
  if (onlyDisagreements) params.set('only_disagreements', 'true');
  params.set('limit', '500');
  return getJson<AnswerRow[]>(`/eval/answers?${params.toString()}`);
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

export function listTraces(opts: { reportId?: string; source?: string } = {}): Promise<
  TraceSummary[]
> {
  const params = new URLSearchParams();
  if (opts.reportId) params.set('report_id', opts.reportId);
  if (opts.source) params.set('source', opts.source);
  params.set('limit', '200');
  return getJson<TraceSummary[]>(`/traces?${params.toString()}`);
}

export function getTrace(traceId: string): Promise<TraceDetail> {
  return getJson<TraceDetail>(`/traces/${traceId}`);
}

export function getEvalTrace(
  version: string,
  reportId: string,
  turnIndex: number,
): Promise<TraceDetail> {
  return getJson<TraceDetail>(
    `/traces/eval/${encodeURIComponent(version)}/${encodeRid(reportId)}?turn_index=${turnIndex}`,
  );
}

export function getTraceStats(): Promise<{ n_turns: number; n_reports: number }> {
  return getJson('/traces/stats');
}

// ---------------------------------------------------------------------------
// Admin: experiments, registry, research, rules
// ---------------------------------------------------------------------------

export function getExperiments(): Promise<ExperimentsPayload> {
  return getJson<ExperimentsPayload>('/admin/experiments');
}

export function getRegistry(): Promise<RegistryPayload> {
  return getJson<RegistryPayload>('/admin/registry');
}

export function compareVersions(baseline: string, candidate: string): Promise<ComparisonResult> {
  const params = new URLSearchParams({ baseline, candidate });
  return getJson<ComparisonResult>(`/admin/compare?${params.toString()}`);
}

export function promoteVersion(version: string, force = false): Promise<unknown> {
  return postJson('/admin/registry/promote', { version, force }, true);
}

export function setChallenger(version: string): Promise<unknown> {
  return postJson('/admin/registry/challenger', { version }, true);
}

export function getRules(variant = ''): Promise<RulesPayload> {
  const params = variant ? `?variant=${encodeURIComponent(variant)}` : '';
  return getJson<RulesPayload>(`/admin/rules${params}`);
}

export function getResearchStatus(): Promise<ResearchStatus> {
  return getJson<ResearchStatus>('/admin/research/status');
}

export function startResearch(body: {
  kind: string;
  limit: number;
  retry_n: number;
  variant?: string;
}): Promise<unknown> {
  return postJson('/admin/research/start', body, true);
}

export function cancelResearch(): Promise<unknown> {
  return postJson('/admin/research/cancel', {}, true);
}

export async function streamResearch(
  signal: AbortSignal,
  onEvent: (event: Record<string, unknown>) => void,
): Promise<void> {
  await fetchEventSource(url('/admin/research/stream'), {
    signal,
    openWhenHidden: true,
    async onopen(res) {
      if (!res.ok) throw new Error(`streamResearch HTTP ${res.status}`);
    },
    onmessage(ev) {
      if (!ev.data) return;
      try {
        onEvent(JSON.parse(ev.data));
      } catch {
        // Ignore malformed frames.
      }
    },
    onerror(err) {
      throw err;
    },
  });
}

export function normalizeArgs(args: unknown): unknown {
  if (typeof args === 'string') {
    try {
      return JSON.parse(args);
    } catch {
      return args;
    }
  }
  return args;
}
