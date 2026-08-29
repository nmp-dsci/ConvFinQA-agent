import type { Message, StageName, StageTrace } from '../../types';
import { sumMeasured } from './format';

export const STAGE_ORDER: StageName[] = ['triage', 'preprocess', 'retriever', 'calculator'];

export type StageState = 'done' | 'active' | 'pending' | 'skipped';

export interface StageView {
  stage: StageName;
  state: StageState;
  /** One line of what this stage decided, for the thread's compact strip. */
  detail: string;
  trace: StageTrace | undefined;
}

function triageTurnType(message: Message): string | undefined {
  const value = message.stages?.triage?.output?.turn_type;
  return typeof value === 'string' ? value : undefined;
}

function detailFor(stage: StageName, message: Message, state: StageState): string {
  if (state === 'skipped') return 'skipped';
  const output = message.stages?.[stage]?.output;
  if (!output) return state === 'active' ? 'running…' : '';

  switch (stage) {
    case 'triage': {
      const turnType = typeof output.turn_type === 'string' ? output.turn_type : '';
      const convType = typeof output.conv_type === 'string' ? output.conv_type : '';
      return [turnType, convType].filter(Boolean).join(' · ');
    }
    case 'preprocess': {
      const subs = Array.isArray(output.sub_questions) ? output.sub_questions.length : 0;
      const program = typeof output.program === 'string' ? output.program : '';
      return program || (subs ? `${subs} sub-question${subs === 1 ? '' : 's'}` : '');
    }
    case 'retriever': {
      const answers = Array.isArray(output.answers) ? output.answers.length : 0;
      return answers ? `${answers} value${answers === 1 ? '' : 's'}` : '';
    }
    case 'calculator': {
      const calls = (message.tools ?? []).length;
      return calls ? `${calls} call${calls === 1 ? '' : 's'}` : '';
    }
    default:
      return '';
  }
}

/**
 * The four stages with their state for this turn.
 *
 * "Skipped" is a real, meaningful state and is not the same as "pending": a
 * number-type turn routes straight from triage to the retriever, so preprocess
 * and calculator never run. Showing them struck through says the router chose
 * the short path; showing them greyed out would imply they are still coming.
 */
export function stageViews(message: Message): StageView[] {
  const finished = message.status === 'done' || message.status === 'error';
  const numberPath = triageTurnType(message) === 'number';

  return STAGE_ORDER.map((stage) => {
    const trace = message.stages?.[stage];
    let state: StageState;
    if (trace?.output !== undefined) state = 'done';
    else if (trace?.started) state = 'active';
    else if (numberPath && (stage === 'preprocess' || stage === 'calculator')) state = 'skipped';
    else if (finished) state = 'skipped';
    else state = 'pending';

    return { stage, state, detail: detailFor(stage, message, state), trace };
  });
}

/** Total measured latency across the stages, or `null` if nothing was measured. */
export function totalLatency(message: Message): number | null {
  return sumMeasured(STAGE_ORDER.map((s) => message.stages?.[s]?.metrics?.latency_ms));
}

/** Total measured tokens across the stages, or `null` if nothing was measured. */
export function totalTokens(message: Message): number | null {
  return sumMeasured(STAGE_ORDER.map((s) => message.stages?.[s]?.metrics?.total_tokens));
}

/** The values the retriever pulled out of the filing, in order. */
export function retrievedValues(
  message: Message
): Array<{ question: string; answer: string }> {
  const answers = message.stages?.retriever?.output?.answers;
  if (!Array.isArray(answers)) return [];
  return answers.map((entry) => {
    const row = entry as { question?: unknown; answer?: unknown };
    return {
      question: typeof row.question === 'string' ? row.question : '',
      answer: row.answer === undefined || row.answer === null ? '' : String(row.answer),
    };
  });
}
