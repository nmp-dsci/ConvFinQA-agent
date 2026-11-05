/**
 * Optimise › Fine-tuning's two reads (s32): the committed record and the cases behind it.
 *
 * `GET /eval/fine-tuning` serves `evaluation/fine_tuning.json` verbatim — built by
 * `convfinqa-sft progress` from committed passes and training records, and held to them by
 * `tests/test_fine_tuning.py` — so these types mirror `sft/progress.py::build`.
 * `GET /eval/fine-tuning/questions` lines up the named passes question by question; the record is
 * its allow-list, so only a complete fine-tuning pass can be named.
 */
import { getJson } from '../admin/api';

export type Harness = 'pydantic_ai' | 'pydantic_ai_checked';
export type SplitName = 'val' | 'test';

export interface Slice {
  n: number;
  correct: number;
  accuracy: number | null;
}

export interface FtPass {
  id: string;
  csv: string;
  run_id: string | null;
  date: string;
  split: SplitName | null;
  role: 'base' | 'checkpoint' | 'teacher';
  model: string;
  effort: string;
  harness: Harness;
  temperature: number | null;
  round: string | null;
  epoch: number | null;
  n_conversations: number;
  n_unscored: number;
  complete: boolean;
  n: number;
  correct: number;
  accuracy: number | null;
  program: Slice;
  number: Slice;
}

/** `to` against `from` on their shared questions: the gate's statistics. */
export interface Paired {
  from: string;
  to: string;
  n: number;
  delta_pp: number;
  fixed: number;
  broken: number;
  p_one_sided: number;
  ci_pp: [number, number];
  same_harness: boolean;
}

export interface Checkpoint {
  key: string;
  epoch: number;
  step: number;
  model: string;
  val_pass: string | null;
  test_pass: string | null;
  vs_start: Paired | null;
}

export interface LossPoint {
  step: number;
  epoch: number;
  loss: number;
  learning_rate: number;
}

export interface FtRound {
  key: string;
  label: string;
  run_id: string;
  record: string;
  started: string;
  finished: string;
  instance: string;
  code_sha: string;
  start: { label: string; round: string | null; model: string; val_pass: string | null };
  dataset: {
    version: string;
    card: string;
    manifest: string;
    n_rows: number;
    n_prefix_rows: number;
    n_turns: number;
    n_type2: number;
    n_dropped: number;
    tokens_trained: number;
  };
  settings: {
    learning_rate: number;
    schedule: string;
    epochs: number;
    effective_batch: number;
    max_length: number;
    lora_r: number;
    lora_alpha: number;
    lora_dropout: number;
    target_modules: string;
    save: string;
  };
  steps: number;
  train_loss: number;
  final_loss: number;
  duration_s: number;
  cost_usd: number;
  cost_basis: string;
  loss: LossPoint[];
  sweep_harness: Harness | null;
  checkpoints: Checkpoint[];
  best: string | null;
  rescores: string[];
}

export interface Stage {
  key: string;
  label: string;
  kind: 'base' | 'round' | 'harness';
  round?: string;
  model: string | null;
  harness: Harness | null;
  pass: string | null;
  vs_prev: Paired | null;
}

export interface FineTuningRecord {
  schema: number;
  student: string;
  splits: Record<SplitName, { manifest: string; n_conversations: number; role: string }>;
  harnesses: Record<Harness, { label: string; about: string }>;
  default_temperature: number;
  rounds: FtRound[];
  progression: Record<SplitName, Stage[]>;
  teacher: {
    name: string;
    model: string;
    effort: string;
    pass: string;
    vs_final: Paired | null;
    gap_closed: number | null;
  } | null;
  passes: FtPass[];
  excluded: { id: string; reason: string }[];
  skipped_runs: { run_id: string; reason: string }[];
  inputs: { path: string; sha256: string }[];
}

export interface CaseAnswer {
  answer: string;
  correct: boolean;
  error?: string;
}

export interface CaseRow {
  report_id: string;
  turn_index: number;
  question: string;
  gold_answer: string;
  gold_program: string;
  turn_type: string;
  answers: (CaseAnswer | null)[];
}

export interface CasesPayload {
  split: SplitName;
  passes: string[];
  rows: CaseRow[];
}

export function getFineTuning(): Promise<FineTuningRecord> {
  return getJson<FineTuningRecord>('/eval/fine-tuning');
}

export function getFineTuningCases(passIds: readonly string[]): Promise<CasesPayload> {
  const q = passIds.map((id) => `pass=${encodeURIComponent(id)}`).join('&');
  return getJson<CasesPayload>(`/eval/fine-tuning/questions?${q}`);
}
