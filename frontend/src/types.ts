export type StageName = 'triage' | 'preprocess' | 'retriever' | 'calculator';

export type SSEEvent =
  | { event: 'stage_start'; stage: StageName }
  | { event: 'stage_output'; stage: StageName; output: Record<string, unknown> }
  | { event: 'tool_call'; stage: StageName; tool: string; args: unknown }
  | { event: 'tool_return'; stage: StageName; tool: string; result: string }
  | { event: 'answer'; answer: string }
  | { event: 'done'; turn_index: number }
  | { event: 'error'; error: string };

export interface ToolTrace {
  tool: string;
  args: unknown;
  result?: string;
}

export interface StageTrace {
  started: boolean;
  output?: Record<string, unknown>;
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
