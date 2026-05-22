import { fetchEventSource } from '@microsoft/fetch-event-source';
import type { EvalSummary, PredRow, ReportDocument, ReportQuestion, SSEEvent } from './types';

const API_BASE_KEY = 'convfinqa.apiBase';

export function getApiBase(): string {
  if (typeof window === 'undefined') return '';
  try {
    return window.localStorage.getItem(API_BASE_KEY) ?? '';
  } catch {
    return '';
  }
}

function url(path: string): string {
  return `${getApiBase()}${path}`;
}

function encodeRid(rid: string): string {
  // FastAPI route uses {rid:path}, so encoding still resolves correctly.
  return rid.split('/').map(encodeURIComponent).join('/');
}

export class SessionGoneError extends Error {
  constructor() {
    super('Session no longer exists on server (TTL evicted or never created).');
    this.name = 'SessionGoneError';
  }
}

export async function listReports(q = '', limit = 500): Promise<string[]> {
  const params = new URLSearchParams();
  if (q) params.set('q', q);
  params.set('limit', String(limit));
  const res = await fetch(url(`/reports?${params.toString()}`));
  if (!res.ok) throw new Error(`listReports failed: ${res.status}`);
  return res.json();
}

export async function getQuestions(reportId: string): Promise<ReportQuestion[]> {
  const res = await fetch(url(`/reports/${encodeRid(reportId)}/questions`));
  if (!res.ok) throw new Error(`getQuestions failed: ${res.status}`);
  return res.json();
}

export async function getDocument(reportId: string): Promise<ReportDocument> {
  const res = await fetch(url(`/reports/${encodeRid(reportId)}/document`));
  if (!res.ok) throw new Error(`getDocument failed: ${res.status}`);
  return res.json();
}

export async function createSession(reportId: string): Promise<string> {
  const res = await fetch(url('/sessions'), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ report_id: reportId }),
  });
  if (!res.ok) throw new Error(`createSession failed: ${res.status}`);
  const body = (await res.json()) as { session_id: string };
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
  if (!res.ok) throw new Error(`getSession failed: ${res.status}`);
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  // Swallow 404 — server may have already evicted via TTL
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

export async function listEvalRuns(): Promise<string[]> {
  const res = await fetch(url('/eval/runs'));
  if (!res.ok) throw new Error(`listEvalRuns failed: ${res.status}`);
  return res.json();
}

export async function getEvalSummary(runName: string): Promise<EvalSummary> {
  const res = await fetch(url(`/eval/runs/${encodeURIComponent(runName)}/summary`));
  if (!res.ok) throw new Error(`getEvalSummary failed: ${res.status}`);
  return res.json();
}

export async function getEvalPredictions(runName: string, model: string): Promise<PredRow[]> {
  const params = new URLSearchParams({ model });
  const res = await fetch(url(`/eval/runs/${encodeURIComponent(runName)}/predictions?${params}`));
  if (!res.ok) throw new Error(`getEvalPredictions failed: ${res.status}`);
  return res.json();
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
