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
  /** DPO only: the held-out preference pairs at this checkpoint's step. */
  pair_eval?: PairEval | null;
}

/** A DPO evaluation on held-out pairs: how often the policy prefers the chosen side, and by how much. */
export interface PairEval {
  step: number;
  reward_accuracy: number;
  margin: number;
  chosen_dlogp: number;
  rejected_dlogp: number;
  n: number;
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
  /** `sft` trains on the teacher's conversations; `dpo` on chosen/rejected pairs, from an SFT adapter. */
  method: 'sft' | 'dpo';
  /** A pilot is a sibling of the round it scouted, not a step on the progression. */
  pilot: boolean;
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
    /** SFT: conversations. DPO: preference pairs. */
    n_rows: number;
    /** SFT: questions trained. DPO: the distinct questions the pairs come from. */
    n_turns: number;
    n_prefix_rows?: number;
    n_type2?: number;
    n_dropped?: number;
    tokens_trained?: number;
    source?: string | null;
    n_eval_pairs?: number;
    program_pairs?: number;
    number_pairs?: number;
  };
  settings: {
    learning_rate: number;
    schedule: string;
    epochs: number;
    effective_batch: number;
    max_length?: number;
    beta?: number;
    nll_alpha?: number;
    max_grad_norm?: number;
    reference?: string;
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
  pair_eval: PairEval[];
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
  /** Every other agent's newest complete test-100 pass, best first (`story.json → test_timeline`). */
  agents: AgentPass[];
  /** DPO datasets built but not yet trained on. */
  planned: PlannedRound[];
  passes: FtPass[];
  excluded: { id: string; reason: string }[];
  skipped_runs: { run_id: string; reason: string }[];
  inputs: { path: string; sha256: string }[];
}

export interface AgentPass {
  name: string;
  runtime: string;
  version: string;
  model: string;
  effort: string | null;
  accuracy: number;
  n: number | null;
  date: string;
  run_id: string | null;
  outcome: string | null;
  label: string | null;
  champion: boolean;
}

export interface PlannedRound {
  key: string;
  label: string;
  status: string;
  card: string;
  n_pairs: number;
  n_eval_pairs: number;
  n_questions: number;
  program_pairs: number;
  number_pairs: number;
  note: string;
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
