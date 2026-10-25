import { describe, expect, it } from 'vitest';
import { boardRows } from './SectionPaper';
import type { SystemData } from './useSystemData';
import type { RuntimeArm, RuntimeComparison } from '../admin/api';

/**
 * Our rows on the leaderboard come from `/eval/campaigns → runtime_comparison` only: both
 * champions and, since s25, the open agent as a challenger named for itself.
 */

function arm(overrides: Partial<RuntimeArm>): RuntimeArm {
  return {
    version: null,
    run_name: null,
    accuracy: null,
    by_turn_type: null,
    panel: null,
    cost: null,
    wall: null,
    ...overrides,
  };
}

function data(runtimes: RuntimeComparison | null | undefined): SystemData {
  return { runtimes, gateQuestions: 349, gateReports: 100 } as unknown as SystemData;
}

const RUNTIMES: RuntimeComparison = {
  pipeline: arm({ version: 'v8', accuracy: 0.816619, program_accuracy: 0.392405 }),
  agent_sdk: arm({ version: 'sdk_v1', accuracy: 0.905444, model: 'claude-sonnet-5' }),
  open_agent: arm({
    version: 'sdk_v1',
    agent_name: 'llm_react_v1',
    accuracy: 0.876791,
    program_accuracy: 0.392405,
    model: 'deepseek-flash',
  }),
  gate: null,
};

describe('our rows on the leaderboard', () => {
  it('puts the open agent on the board under its own name, ranked by accuracy', () => {
    const ours = boardRows(data(RUNTIMES)).filter((r) => r.ours);
    expect(ours.map((r) => [r.system, r.exeText])).toEqual([
      ['sdk-v1 · one Claude session (claude-sonnet-5)', '90.5'],
      ['llm_react_v1 · one open-model session (deepseek-flash)', '87.7'],
      ['llm-v8 · four DeepSeek agents', '81.7'],
    ]);
  });

  it('keeps an unscored open arm on the board and says why, rather than dropping it', () => {
    const { open_agent: _, ...older } = RUNTIMES;
    const row = boardRows(data(older)).find((r) => r.key === 'ours-open_agent');
    expect(row?.exe).toBeNull();
    expect(row?.exeText).toMatch(/not scored/);
  });
});
