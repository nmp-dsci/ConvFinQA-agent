import { useQuery } from '@tanstack/react-query';
import { ApiError, getApiBase } from '../../api';

/**
 * `GET /eval/readiness` — the production gen-AI rubric, R1–R9, as the repo
 * scores itself. Hand-maintained in `evaluation/readiness.json`, checked on
 * every pull request by `evalloop.readiness.problems` (every proof path must
 * exist, every app route must be served, the score must match the rows), so
 * what the app shows is a checked claim rather than a typed one.
 */

export type ReadinessStatus = 'shipped' | 'partial' | 'designed' | 'na';

export interface ReadinessRow {
  ref: string;
  key: string;
  order: number;
  label: string;
  short: string;
  question: string;
  status: ReadinessStatus;
  how: string;
  proof: string[];
  app: string[];
}

export interface ReadinessHalf {
  label: string;
  range: string;
  upto: number;
  blurb: string;
}

export interface Readiness {
  score: string;
  rung: number;
  measured: string;
  statuses: Record<ReadinessStatus, { glyph: string; label: string; blurb: string }>;
  halves: ReadinessHalf[];
  rows: ReadinessRow[];
}

export async function getReadiness(): Promise<Readiness> {
  const res = await fetch(`${getApiBase()}/eval/readiness`);
  if (!res.ok) {
    let message = `/eval/readiness failed: ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === 'string') message = body.detail;
    } catch {
      // keep the status message
    }
    throw new ApiError(message, res.status, '');
  }
  return res.json() as Promise<Readiness>;
}

export function useReadiness() {
  return useQuery({
    queryKey: ['eval-readiness'],
    queryFn: getReadiness,
    staleTime: Infinity,
    retry: false,
  });
}

/** `6 / 9` from the rows themselves, so a stale `score` string cannot disagree. */
export function shippedCount(rows: ReadinessRow[]): { shipped: number; total: number } {
  return { shipped: rows.filter((r) => r.status === 'shipped').length, total: rows.length };
}
