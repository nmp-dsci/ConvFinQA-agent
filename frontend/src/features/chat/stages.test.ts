import { describe, expect, it } from 'vitest';
import type { Message } from '../../types';
import { EM_DASH, fmtCost, fmtMs, fmtTokens, shortRid, sumMeasured } from './format';
import { retrievedValues, stageViews, totalLatency, totalTokens } from './stages';

function message(patch: Partial<Message> = {}): Message {
  return {
    id: 'm1',
    role: 'assistant',
    text: '',
    status: 'streaming',
    createdAt: 0,
    ...patch,
  };
}

const byName = (views: ReturnType<typeof stageViews>) =>
  Object.fromEntries(views.map((v) => [v.stage, v.state]));

describe('stage states', () => {
  it('marks preprocess and calculator skipped on the number path', () => {
    const states = byName(
      stageViews(
        message({
          status: 'done',
          stages: {
            triage: { started: true, output: { turn_type: 'number', conv_type: 'Type I' } },
            retriever: { started: true, output: { answers: [{ question: 'q', answer: '641' }] } },
          },
        })
      )
    );
    expect(states).toEqual({
      triage: 'done',
      preprocess: 'skipped',
      retriever: 'done',
      calculator: 'skipped',
    });
  });

  it('distinguishes a stage still running from one that never will', () => {
    const streaming = byName(
      stageViews(
        message({
          stages: {
            triage: { started: true, output: { turn_type: 'program' } },
            preprocess: { started: true },
          },
        })
      )
    );
    expect(streaming.preprocess).toBe('active');
    // Not started, and the turn is still going: coming, not skipped.
    expect(streaming.retriever).toBe('pending');

    const failed = byName(
      stageViews(
        message({
          status: 'error',
          stages: { triage: { started: true, output: { turn_type: 'program' } } },
        })
      )
    );
    expect(failed.retriever).toBe('skipped');
  });

  it('sums only the stages that were actually measured', () => {
    const m = message({
      status: 'done',
      stages: {
        triage: { started: true, output: {}, metrics: { latency_ms: 100, total_tokens: 10 } },
        retriever: { started: true, output: {}, metrics: { latency_ms: 250 } },
      },
    });
    expect(totalLatency(m)).toBe(350);
    expect(totalTokens(m)).toBe(10);
  });

  it('returns null — never zero — when nothing was measured', () => {
    // Every replayed demo turn is this case: the pack carries `metrics: {}`.
    const m = message({
      status: 'done',
      stages: { triage: { started: true, output: {}, metrics: {} } },
    });
    expect(totalLatency(m)).toBeNull();
    expect(totalTokens(m)).toBeNull();
    expect(sumMeasured([null, undefined])).toBeNull();
  });

  it('reads the retriever’s values for the filing highlight', () => {
    const m = message({
      stages: {
        retriever: {
          started: true,
          output: {
            answers: [
              { question: 'cash from operations in 2008', answer: 641 },
              { question: 'cash from operations in 2009', answer: '868.0' },
            ],
          },
        },
      },
    });
    expect(retrievedValues(m)).toEqual([
      { question: 'cash from operations in 2008', answer: '641' },
      { question: 'cash from operations in 2009', answer: '868.0' },
    ]);
  });
});

describe('formatters', () => {
  it('renders an unmeasured value as an em dash, not zero', () => {
    expect(fmtMs(null)).toBe(EM_DASH);
    expect(fmtTokens(undefined)).toBe(EM_DASH);
    expect(fmtCost(null)).toBe(EM_DASH);
    // A genuine zero still reads as zero.
    expect(fmtTokens(0)).toBe('0');
  });

  it('shortens a report id for the 210 px sessions pane', () => {
    expect(shortRid('Double_MAR/2010/page_55.pdf')).toBe('MAR · 2010 · p.55');
    expect(shortRid('Single_VLO/2011/page_126.pdf-1')).toBe('VLO · 2011 · p.126');
    expect(shortRid('something-else')).toBe('something-else');
  });
});
