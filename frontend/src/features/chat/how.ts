import type { Message } from '../../types';
import { retrievedValues, stageViews } from './stages';

/**
 * The trace, in one sentence.
 *
 * A turn card shows the answer and four stage chips; this is the line between
 * them that says what the stages did, in words a reader who has never seen the
 * pipeline can follow. It is derived from the same stage outputs the chips and
 * the inspector read, so it cannot say something the trace does not — and it
 * says nothing when the trace is empty rather than guessing.
 */
export function howLine(message: Message): string {
  if (message.status !== 'done') return '';
  const views = stageViews(message);
  const byStage = Object.fromEntries(views.map((v) => [v.stage, v]));
  const triage = message.stages?.triage?.output;
  const turnType = typeof triage?.turn_type === 'string' ? triage.turn_type : '';
  const convType = typeof triage?.conv_type === 'string' ? triage.conv_type : '';
  const parts: string[] = [];

  if (turnType === 'number') {
    parts.push(
      `A look-up${convType ? ` in a ${convType} conversation` : ''}: triage sent it straight to the retriever, so preprocess and the calculator never ran.`,
    );
  } else if (turnType === 'program') {
    parts.push(`A computation${convType ? ` in a ${convType} conversation` : ''}.`);
  }

  const pre = message.stages?.preprocess?.output;
  const subs = Array.isArray(pre?.sub_questions) ? pre.sub_questions.length : 0;
  const program = typeof pre?.program === 'string' ? pre.program : '';
  if (byStage.preprocess?.state === 'done' && (subs || program)) {
    parts.push(
      `Preprocess ${subs ? `split it into ${subs} sub-question${subs === 1 ? '' : 's'}` : 'resolved it'}${
        program ? ` and planned ${program}` : ''
      }.`,
    );
  }

  const values = retrievedValues(message)
    .map((v) => v.answer)
    .filter(Boolean);
  if (byStage.retriever?.state === 'done') {
    parts.push(
      values.length
        ? `The retriever found ${values.slice(0, 4).join(', ')}${values.length > 4 ? '…' : ''} in the filing.`
        : 'The retriever returned nothing it could use.',
    );
  }

  const calls = message.tools ?? [];
  if (byStage.calculator?.state === 'done') {
    if (calls.length) {
      const named = calls
        .map((t) => t.tool)
        .filter(Boolean)
        .slice(0, 4)
        .join(', ');
      parts.push(
        `The calculator ran ${calls.length} call${calls.length === 1 ? '' : 's'}${named ? ` (${named})` : ''} to reach the answer.`,
      );
    } else {
      parts.push('The calculator ran but recorded no tool calls.');
    }
  }

  return parts.join(' ');
}
