/**
 * `GET /eval/gold-review`: every diagnosis whose gold was doubted or whose attribution needed a
 * ruling, read from the committed append-only ledger `evaluation/diagnostics/evalloop/diagnoses.jsonl`.
 * Keyless and read-only, so it is live in the demo.
 */
import { useQuery } from '@tanstack/react-query';
import { getApiBase } from '../../api';
import { get } from '@/kit/http';

/** Why a row is in the review set. A row can carry more than one. */
export type GoldReviewReason = 'gold_suspect_flag' | 'derived_gold_suspect' | 'ambiguous' | 'adjudicated';

export interface GoldReviewCase {
  diagnosis_id: string;
  diagnosed_at: string;
  runtime: string;
  version: string;
  split: string;
  report_id: string;
  question_id: string;
  turn_index: number | null;
  question: string;
  gold_answer: string;
  gold_program: string;
  pred_answer: string;
  pred_program: string;
  /** The attribution rule's verdict: an agent, or `gold_suspect` / `ambiguous` (no agent). */
  derived_agent: string;
  /** The teacher's own stage call. */
  stage: string;
  label: string;
  missing_gold_operands: string[];
  gold_suspect: boolean;
  attribution_disputed: boolean;
  adjudicated: boolean;
  adjudication_reason: string;
  attribution_reason: string;
  what_went_wrong: string;
  confidence: number | null;
  reasons: GoldReviewReason[];
}

export interface GoldReviewSummary {
  n_diagnoses: number;
  n_gold_suspect_flags: number;
  n_derived_gold_suspect: number;
  n_ambiguous: number;
  n_adjudicated: number;
  n_disputed: number;
  n_review_rows: number;
  n_review_questions: number;
}

export interface GoldReviewResponse {
  source: string;
  /** Set, with `cases` empty, when the ledger is absent. */
  reason: string;
  summary: GoldReviewSummary;
  cases: GoldReviewCase[];
}

export function getGoldReview(): Promise<GoldReviewResponse> {
  return get<GoldReviewResponse>(`${getApiBase()}/eval/gold-review`);
}

export function useGoldReview() {
  return useQuery({
    queryKey: ['eval-gold-review'],
    queryFn: getGoldReview,
    staleTime: 60_000, // an append-only ledger: it only grows, and only when the loop runs
  });
}
