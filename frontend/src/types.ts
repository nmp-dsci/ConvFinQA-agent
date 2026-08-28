export type StageName = 'triage' | 'preprocess' | 'retriever' | 'calculator';

export interface StageMetrics {
  latency_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
}

export type SSEEvent =
  | { event: 'stage_start'; stage: StageName }
  | {
      event: 'stage_output';
      stage: StageName;
      output: Record<string, unknown>;
      metrics?: StageMetrics;
    }
  | { event: 'tool_call'; stage: StageName; tool: string; args: unknown }
  | { event: 'tool_return'; stage: StageName; tool: string; result: string }
  | { event: 'answer'; answer: string; program?: string }
  | { event: 'done'; turn_index: number; trace_id?: string }
  | { event: 'error'; error: string; code?: string };

export interface ToolTrace {
  tool: string;
  args: unknown;
  result?: string;
}

export interface StageTrace {
  started: boolean;
  output?: Record<string, unknown>;
  metrics?: StageMetrics;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  text: string;
  goldAnswer?: string;
  goldProgram?: string;
  status: 'pending' | 'streaming' | 'done' | 'error';
  stages?: Partial<Record<StageName, StageTrace>>;
  tools?: ToolTrace[];
  errorText?: string;
  errorCode?: string;
  traceId?: string;
  createdAt: number;
}

export interface Conversation {
  reportId: string;
  sessionId: string | null;
  messages: Message[];
  lastUsedAt: number;
  lastReadAt: number;
  unreadCount: number;
  isStreaming: boolean;
}

export interface ReportQuestion {
  q_order: number;
  question: string;
  gold_answer: string;
  gold_program: string;
}

export interface ReportDocument {
  report_id: string;
  pre_text: string;
  post_text: string;
  table: Record<string, Record<string, number | string>>;
}

export interface AccuracySlice {
  label: string;
  accuracy: number;
  n_correct: number;
  n_total: number;
}

export interface ModelAccuracy {
  overall: AccuracySlice;
  by_turn_type: AccuracySlice[];
  by_conv_type: AccuracySlice[];
  by_q_order: AccuracySlice[];
}

export interface EvalSummary {
  run_name: string;
  available_models: string[];
  models: Record<string, ModelAccuracy>;
}

export interface PredRow {
  report_id: string;
  turn_index: number;
  question: string;
  gold_answer: string;
  gold_program: string;
  pred_answer: string;
  pred_program: string;
  correct: boolean;
  q_order: number;
  turn_type: string;
  conv_type: string;
}

// ---------------------------------------------------------------------------
// Mode — one build, two deployments. `/healthz` tells the app which it is in.
// ---------------------------------------------------------------------------

export type AppMode = 'dev' | 'demo';

export interface BundleSpec {
  prompts_version: string;
  gepa_overlay: string | null;
  lm_mini: string;
  lm_max: string;
  dataset_hash: string;
  code_sha: string;
}

export interface Health {
  ok: boolean;
  mode: AppMode;
  champion: string | null;
  bundle_id: string;
  bundle: BundleSpec;
  demo_reports: number;
}

export interface DemoQuestion {
  turn_index: number;
  question: string;
  gold_answer: string;
  correct: boolean;
}

// ---------------------------------------------------------------------------
// Data & answers explorer
// ---------------------------------------------------------------------------

export interface SplitSummary {
  name: string;
  description: string;
  n_conversations: number;
  n_questions: number;
  report_ids: string[];
}

export interface VersionAnswer {
  version: string;
  pred_answer: string;
  pred_program: string;
  correct: boolean;
}

export interface AnswerRow {
  report_id: string;
  turn_index: number;
  question: string;
  gold_answer: string;
  gold_program: string;
  gold_turn_type: string;
  gold_conv_type: string;
  versions: VersionAnswer[];
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

export interface TraceSummary {
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
  correct: boolean | null;
  bundle_id: string | null;
  latency_ms: number | null;
  total_tokens: number | null;
  error: string | null;
}

export interface StageCapture {
  input?: unknown;
  output?: Record<string, unknown>;
  reasoning?: string;
  metrics?: StageMetrics;
  trajectory?: Array<Record<string, unknown>>;
}

export interface TraceDetail extends TraceSummary {
  capture: Partial<Record<StageName, StageCapture | null>>;
  bundle: Partial<BundleSpec>;
  history_text?: string;
}

// ---------------------------------------------------------------------------
// Experiments & registry
// ---------------------------------------------------------------------------

export interface ExperimentRun {
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

export interface VersionAccuracy {
  version: string;
  /** Over all 200 scored conversations — mixes seen and unseen. Not "held out". */
  accuracy: number;
  n_questions: number;
  /** Over the conversations no optimizer ever saw. The generalisation number. */
  holdout_accuracy: number;
  holdout_n_questions: number;
  slices: Record<string, Record<string, number>>;
}

export interface ExperimentsPayload {
  source: 'live' | 'snapshot';
  runs: ExperimentRun[];
  registry: RegistryPayload;
  versions: VersionAccuracy[];
  exported_at?: string | null;
  tracking?: Record<string, unknown>;
  mode?: AppMode;
}

export interface RegistryVersion {
  version: string;
  registered_at: string;
  source: string;
  bundle_id: string;
  bundle: BundleSpec;
  runs: string[];
  metrics: Record<string, number>;
  notes: string;
}

export interface PromotionEvent {
  at: string;
  event: string;
  version: string;
  previous_champion: string | null;
  actor: string;
  forced: boolean;
  reason: string;
  comparison: ComparisonResult | null;
}

export interface RegistryPayload {
  model: string;
  aliases: Record<string, string>;
  versions: RegistryVersion[];
  history: PromotionEvent[];
  mode?: AppMode;
  can_promote?: boolean;
}

export interface Flip {
  report_id: string;
  q_order: number;
  question: string;
  gold_answer: string;
  baseline_answer: string;
  candidate_answer: string;
}

export interface ComparisonResult {
  baseline_version: string;
  candidate_version: string;
  baseline_accuracy: number;
  candidate_accuracy: number;
  accuracy_delta: number;
  n_compared: number;
  accuracy_ok: boolean;
  no_regressions: boolean;
  promotable: boolean;
  reason: string;
  regressions: Flip[];
  improvements: Flip[];
  slice_deltas: Record<string, Record<string, number>>;
  notes: string[];
}

// ---------------------------------------------------------------------------
// Research console
// ---------------------------------------------------------------------------

export interface ResearchJob {
  job_id: string;
  kind: string;
  args: Record<string, unknown>;
  status: 'running' | 'succeeded' | 'failed' | 'cancelled';
  started_at: string;
  finished_at: string | null;
  returncode: number | null;
  log_tail: string[];
}

export interface ResearchStatus {
  busy: boolean;
  current: ResearchJob | null;
  history: ResearchJob[];
  can_launch: boolean;
  mode: AppMode;
}

export interface Rule {
  [key: string]: unknown;
}

export interface RulesPayload {
  variant: string;
  agents: Record<string, { rules: Rule[]; attempts: Rule[] }>;
}
