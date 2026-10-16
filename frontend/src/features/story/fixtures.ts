/**
 * A test fixture shaped like the committed `story.json` of 2026-10-15: four days of passes, one of
 * them a rate-limited incomplete pass, the champion line v2 → v8 → sdk_v1, the universal-champion
 * record restating sdk_v1, and two train draws. Test-only; nothing in the app imports it.
 */
import type { CampaignsResponse, ChampionStep, TestPass, TrainPass } from '../admin/api';

const DAY = 86_400_000;
const T0 = Date.UTC(2026, 8, 3, 15, 0); // 3 Sep 2026, 15:00 UTC — the first test pass

export function pass(over: Partial<TestPass> & Pick<TestPass, 'run_name' | 'version'>): TestPass {
  return {
    at_ms: T0,
    date: '2026-09-04',
    run_id: null,
    runtime: 'pipeline',
    model: 'deepseek-v4-flash',
    effort: null,
    accuracy: 0.8,
    n_scored: 349,
    n_questions: 349,
    complete: true,
    cost: null,
    champion_run: false,
    label: null,
    outcome: 'rejected',
    reason: '',
    vs: null,
    ...over,
  };
}

export const PASSES: TestPass[] = [
  // Deliberately out of order: the model must sort by start time.
  pass({
    run_name: 'evalloop-test100-v8',
    version: 'v8',
    at_ms: T0 + 3 * 3_600_000,
    accuracy: 0.816619,
    outcome: 'promoted',
    champion_run: true,
    label: 'c01-e03',
    vs: { version: 'v2', model: null, delta_pp: 4.5845, p_value: 0.0404, p_kind: 'one-sided toward better', paired: true, champion_of_the_day: 'v2' },
  }),
  pass({ run_name: 'evalloop-test100-v2', version: 'v2', at_ms: T0, accuracy: 0.770774, outcome: 'champion', champion_run: true }),
  pass({
    run_name: 'evalloop-test100-v6',
    version: 'v6',
    at_ms: T0 + 3_600_000,
    accuracy: 0.793696,
    label: 'c01-e01',
    vs: { version: 'v2', model: null, delta_pp: 2.2923, p_value: 0.191, p_kind: 'one-sided toward better', paired: true },
  }),
  pass({
    run_name: 'sdk-evalloop-test100-sdk_v1·s1-20260905_132411',
    version: 'sdk_v1',
    runtime: 'agent_sdk',
    model: 'claude-sonnet-5',
    effort: 'high',
    at_ms: T0 + DAY + 1_000,
    date: '2026-09-05',
    accuracy: 0.444126,
    complete: false,
    outcome: 'incomplete',
    reason: '176 of 349 turns hit the session limit',
    cost: 5.25,
  }),
  pass({
    run_name: 'sdk-evalloop-test100-sdk_v1·s1-20260905_174627',
    version: 'sdk_v1',
    runtime: 'agent_sdk',
    model: 'claude-sonnet-5',
    effort: 'high',
    at_ms: T0 + DAY + 4 * 3_600_000,
    date: '2026-09-05',
    accuracy: 0.905444,
    outcome: 'promoted',
    champion_run: true,
    label: 's01 cross-runtime',
    cost: 11.83,
    vs: { version: 'v8', model: null, delta_pp: 8.8825, p_value: 0.000286, p_kind: 'one-sided toward better', paired: true },
  }),
  pass({
    run_name: 'sdk-evalloop-test100-sdk_v2·s2-20260905_182812',
    version: 'sdk_v2',
    runtime: 'agent_sdk',
    model: 'claude-sonnet-5',
    effort: 'high',
    at_ms: T0 + DAY + 5 * 3_600_000,
    date: '2026-09-05',
    accuracy: 0.876791,
    label: 's01-e02',
    cost: 11.82,
    vs: { version: 'sdk_v1', model: 'claude-sonnet-5', delta_pp: -2.8653, p_value: 0.917, p_kind: 'one-sided toward better', paired: true },
  }),
  pass({
    run_name: 'sdk-evalloop-test100-sdk_v1·s1-haiku-5-5-high-20261015_175310',
    version: 'sdk_v1',
    runtime: 'agent_sdk',
    model: 'claude-haiku-5-5',
    effort: 'high',
    at_ms: T0 + 42 * DAY,
    date: '2026-10-15',
    accuracy: 0.873926,
    outcome: 'falls_short',
    cost: 0.73,
    vs: { version: 'sdk_v1', model: 'claude-sonnet-5', delta_pp: -3.1519, p_value: 0.0429, p_kind: 'one-sided toward worse', paired: true },
  }),
];

function step(over: Partial<ChampionStep> & Pick<ChampionStep, 'version' | 'event'>): ChampionStep {
  return {
    at: null,
    at_ms: null,
    effective_at_ms: null,
    reason: '',
    runtime: 'pipeline',
    model: null,
    effort: null,
    accuracy: null,
    run_name: null,
    ...over,
  };
}

export const STEPS: ChampionStep[] = [
  step({ event: 'promote', version: 'v2', at: '2026-08-28T00:15:19', effective_at_ms: T0 - 7 * DAY, accuracy: 0.770774, run_name: 'evalloop-test100-v2' }),
  step({ event: 'promote', version: 'v3_1', at: '2026-09-02T04:59:09', effective_at_ms: T0 - DAY }), // legacy: no gate accuracy
  step({ event: 'promote', version: 'v2', at: '2026-09-03T14:00:00', effective_at_ms: T0 - 3_600_000, accuracy: 0.770774, run_name: 'evalloop-test100-v2' }),
  step({ event: 'promote', version: 'v8', at: '2026-09-03T19:33:24', effective_at_ms: T0 + 3 * 3_600_000, accuracy: 0.816619, run_name: 'evalloop-test100-v8' }),
  step({
    event: 'promote_sdk',
    version: 'sdk_v1',
    runtime: 'agent_sdk',
    model: 'claude-sonnet-5',
    effort: 'high',
    at: '2026-09-05T08:01:16',
    // Written after sdk_v2's pass ran; the line steps at the pass that measured it.
    at_ms: T0 + DAY + 7 * 3_600_000,
    effective_at_ms: T0 + DAY + 4 * 3_600_000,
    accuracy: 0.905444,
    run_name: 'sdk-evalloop-test100-sdk_v1·s1-20260905_174627',
  }),
  step({
    event: 'universal_champion',
    version: 'sdk_v1',
    runtime: 'agent_sdk',
    model: 'claude-sonnet-5',
    effort: 'high',
    at: '2026-10-15T04:57:39',
    effective_at_ms: T0 + DAY + 4 * 3_600_000,
    accuracy: 0.905444,
    run_name: 'sdk-evalloop-test100-sdk_v1·s1-20260905_174627',
  }),
];

export const TRAIN: TrainPass[] = [
  {
    at_ms: T0 + DAY + 4.5 * 3_600_000,
    date: '2026-09-05',
    run_name: 'sdk-evalloop-train100-sdk_v1·s1-20260905_180924',
    version: 'sdk_v1',
    runtime: 'agent_sdk',
    model: 'claude-sonnet-5',
    effort: 'high',
    accuracy: 0.885,
    n_questions: 314,
    train_seed: '2027',
    stop_at_first_wrong: true,
    campaign: 's01',
    complete: true,
  },
  {
    at_ms: T0 + 42 * DAY - 3_600_000,
    date: '2026-10-15',
    run_name: 'sdk-evalloop-train100-sdk_v1·s1-haiku-5-5-high-20261015_180335',
    version: 'sdk_v1',
    runtime: 'agent_sdk',
    model: 'claude-haiku-5-5',
    effort: 'high',
    accuracy: 0.8806,
    n_questions: 310,
    train_seed: '2027',
    stop_at_first_wrong: true,
    campaign: null,
    complete: true,
  },
];

export const CAMPAIGNS: CampaignsResponse = {
  champion: 'sdk_v1',
  champion_accuracy: 0.905444,
  champion_panel: {},
  rule: 'r',
  generated_at: '2026-10-15T09:00:00Z',
  split: { name: 'eval_loop_v2', gate_questions: 349, gate_reports: 100 },
  campaigns: [],
  experiments: [],
  champion_track: [],
  test_timeline: PASSES,
  champion_steps: STEPS,
  train_passes: TRAIN,
};
