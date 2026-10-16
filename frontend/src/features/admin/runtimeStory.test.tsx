import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  ARMS,
  PAPER_HUMAN,
  formatDay,
  formatP,
  formatWall,
  modelRows,
  modelShortName,
  progression,
  runtimeVerdict,
  sliceEffect,
  sliceRows,
} from './runtimeStory';
import { ArmCard, ModelSwapTable, SliceTable, VerdictBanner } from './Runtimes';
import { ProgressionChart, SLOT_CHARS, columnLines } from './ProgressionChart';
import type {
  CampaignExperiment,
  ChampionPoint,
  Challengers,
  RuntimeArm,
  RuntimeComparison,
  SdkModelComparison,
  TestPass,
} from './api';

/**
 * What these tests guard is the page's honesty, not its layout.
 *
 * The Runtimes page makes a recommendation, so the properties worth pinning are
 * the ones that would let it make the wrong one quietly: a banner that keeps
 * saying "adopt" after the gate stops promoting, an arm with no run rendering as
 * 0.0% instead of "not yet run", a slice where nothing moved reading as a small
 * win, and a chart drawing a column for a stage that was never scored.
 *
 * Rendering goes through `react-dom/server`, which needs no DOM and no
 * testing-library: the components are pure functions of their props by
 * construction — the page component is the only thing that fetches — so static
 * markup is enough to assert on and the runner stays the project's `vitest` in
 * its existing Node environment.
 */

// ---------------------------------------------------------------------------
// Fixtures — shaped like the committed story, with the figures made obvious
// ---------------------------------------------------------------------------

const EMPTY_ARM: RuntimeArm = {
  version: null,
  run_name: null,
  accuracy: null,
  by_turn_type: null,
  panel: null,
  cost: null,
  wall: null,
  program_accuracy: null,
};

const PIPELINE_ARM: RuntimeArm = {
  version: 'v8',
  run_name: 'evalloop-test100-v8·t2p2r5c2',
  accuracy: 0.816619,
  by_turn_type: { number: 0.954955, program: 0.752101 },
  panel: {
    triage: 0.922636,
    preprocess: null,
    retriever: 0.768072,
    calculator: 0.751055,
  },
  cost: null,
  wall: 343.55,
  program_accuracy: 0.392405,
};

const SDK_ARM: RuntimeArm = {
  version: 'sdk_v1',
  run_name: 'sdk-evalloop-test100-sdk_v1·s1',
  accuracy: 0.905444,
  by_turn_type: { number: 0.954955, program: 0.882353 },
  panel: {
    triage: 0.974212,
    preprocess: 0.875536,
    retriever: 0.785693,
    calculator: 0.881857,
  },
  cost: 27.624373,
  wall: 786.8,
  program_accuracy: 0.409283,
};

function comparison(overrides: Partial<RuntimeComparison> = {}): RuntimeComparison {
  return {
    pipeline: PIPELINE_ARM,
    agent_sdk: SDK_ARM,
    gate: {
      delta_pp: 8.8825,
      p_value: 0.000286,
      ci: [0.042017, 0.137143],
      fixed: 38,
      broken: 7,
      candidate_version: 'sdk_v1',
      promoted: true,
      gate_id: 'g-20260905080150-00fce8a3',
      by_turn_type: {
        program: {
          n: 238,
          baseline_accuracy: 0.752101,
          candidate_accuracy: 0.882353,
          delta_pp: 13.0252,
          fixed: 35,
          broken: 4,
          cluster_p_one_sided: 0.000047,
        },
        number: {
          n: 111,
          baseline_accuracy: 0.954955,
          candidate_accuracy: 0.954955,
          delta_pp: 0,
          fixed: 3,
          broken: 3,
          cluster_p_one_sided: 0.5,
        },
      },
    },
    ...overrides,
  };
}

const SPLIT = { name: 'eval_loop_v2', gate_reports: 100, gate_questions: 349 };

const TRACK: ChampionPoint[] = [
  {
    version: 'v2',
    at: 1,
    accuracy: 0.770774,
    panel: {},
    moved_by: null,
    target_agent: null,
  },
  {
    version: 'v8',
    at: 2,
    accuracy: 0.816619,
    panel: {},
    moved_by: 'c01-e03',
    target_agent: 'retriever',
  },
];

function experiment(patch: Partial<CampaignExperiment> = {}): CampaignExperiment {
  return {
    label: 's01-e02',
    campaign: 's01',
    target_agent: 'calculator/wrong-format',
    target_class: 'calculator/wrong-format',
    runtime: 'agent_sdk',
    edits: [],
    baseline_version: 'sdk_v1',
    candidate_version: 'sdk_v2',
    promoted: false,
    at: 3,
    accuracy_delta: -0.028653,
    cluster_p_one_sided: 0.917241,
    delta_ci_lo: -0.073171,
    delta_ci_hi: 0.008523,
    n_compared: 349,
    fixed: 6,
    broken: 16,
    accuracy_baseline: 0.905444,
    accuracy_candidate: 0.876791,
    panel_baseline: {},
    panel_candidate: {},
    summary_of_changes: '',
    rationale: '',
    diff: '',
    ...patch,
  };
}

const BASELINE_EXPERIMENT = experiment({
  label: 'P3 diagnostic: unoptimised SDK vs optimised pipeline',
  target_agent: 'baseline',
  target_class: 'baseline',
  baseline_version: 'v8',
  candidate_version: 'sdk_v1',
  promoted: true,
  accuracy_delta: 0.088825,
  accuracy_baseline: 0.816619,
  accuracy_candidate: 0.905444,
});

// ---------------------------------------------------------------------------
// The verdict
// ---------------------------------------------------------------------------

describe('the recommendation is derived from the gate, not written down', () => {
  it('recommends the Agent SDK runtime when the gate promoted a positive delta', () => {
    const verdict = runtimeVerdict(comparison(), SPLIT);
    expect(verdict.recommendation).toBe('adopt-agent-sdk');
    expect(verdict.headline).toMatch(/Claude Agent SDK/);
    expect(verdict.candidateVersion).toBe('sdk_v1');
    expect(verdict.gateQuestions).toBe(349);
  });

  it('holds on the pipeline the moment the gate stops promoting', () => {
    const rejected = comparison();
    const verdict = runtimeVerdict(
      { ...rejected, gate: { ...rejected.gate!, promoted: false } },
      SPLIT,
    );
    expect(verdict.recommendation).toBe('stay-on-pipeline');
    expect(verdict.headline).toMatch(/four-agent pipeline \(v8\)/);
  });

  it('refuses to recommend anything when no cross-runtime gate exists', () => {
    const verdict = runtimeVerdict({ ...comparison(), gate: null }, SPLIT);
    expect(verdict.recommendation).toBe('not-yet-run');
    expect(verdict.deltaPp).toBeNull();
    expect(verdict.pValue).toBeNull();
  });

  it('does not read a promoted flag as a recommendation when the delta is not positive', () => {
    const c = comparison();
    const verdict = runtimeVerdict({ ...c, gate: { ...c.gate!, delta_pp: 0 } }, SPLIT);
    expect(verdict.recommendation).toBe('stay-on-pipeline');
  });
});

describe('the verdict banner reflects gate.promoted', () => {
  it('states the move when the gate promoted', () => {
    const html = renderToStaticMarkup(
      <VerdictBanner verdict={runtimeVerdict(comparison(), SPLIT)} />,
    );
    expect(html).toContain('data-recommendation="adopt-agent-sdk"');
    expect(html).toMatch(/Move the runtime to the Claude Agent SDK/);
    expect(html).toContain('+8.88pp');
    expect(html).toContain('349 q / 100 conv');
  });

  it('states the opposite when it did not, on the same payload', () => {
    const c = comparison();
    const html = renderToStaticMarkup(
      <VerdictBanner
        verdict={runtimeVerdict({ ...c, gate: { ...c.gate!, promoted: false } }, SPLIT)}
      />,
    );
    expect(html).toContain('data-recommendation="stay-on-pipeline"');
    expect(html).not.toMatch(/Move the runtime/);
    expect(html).toMatch(/Keep the four-agent pipeline/);
  });

  it('says nothing has been run when there is no gate, and prints no zeros', () => {
    const html = renderToStaticMarkup(
      <VerdictBanner verdict={runtimeVerdict({ ...comparison(), gate: null }, SPLIT)} />,
    );
    expect(html).toContain('data-recommendation="not-yet-run"');
    expect(html).not.toMatch(/0\.00pp/);
    expect(html).toMatch(/no cross-runtime gate has been run/);
  });
});

// ---------------------------------------------------------------------------
// Absent arms
// ---------------------------------------------------------------------------

describe('an arm with no run', () => {
  it('renders "not yet run" rather than a zero', () => {
    const html = renderToStaticMarkup(
      <ArmCard desc={ARMS.agent_sdk} arm={EMPTY_ARM} recommended={false} />,
    );
    expect(html).toContain('data-present="false"');
    expect(html).toMatch(/not yet run/);
    expect(html).not.toMatch(/0\.0%/);
    expect(html).not.toMatch(/\$0\.00/);
  });

  it('renders every figure it does have, and marks only the unscored ones', () => {
    const html = renderToStaticMarkup(
      <ArmCard desc={ARMS.pipeline} arm={PIPELINE_ARM} recommended={false} />,
    );
    expect(html).toContain('data-present="true"');
    expect(html).toContain('81.7%');
    // Program accuracy is the check on the headline and must be shown beside it.
    expect(html).toContain('39.2%');
    // Preprocess has no score on this run: absent, never 0%.
    expect(html).toContain('not scored');
    expect(html).not.toMatch(/>0\.0%</);
    // Wall clock reads as time, cost as absent rather than free.
    expect(html).toContain('6 min');
    expect(html).toMatch(/not yet run/);
  });
});

// ---------------------------------------------------------------------------
// Slices
// ---------------------------------------------------------------------------

describe('the per-slice split', () => {
  it('puts the slice carrying the gain first', () => {
    const rows = sliceRows(comparison().gate);
    expect(rows.map((r) => r.key)).toEqual(['program', 'number']);
  });

  it('calls a zero delta no effect, and a significant gain significant', () => {
    const rows = sliceRows(comparison().gate);
    expect(rows[0].effect).toBe('significant');
    expect(rows[1].effect).toBe('no-effect');
  });

  it('keeps "no effect" and "not significant" apart', () => {
    expect(sliceEffect(0, 0.5)).toBe('no-effect');
    expect(sliceEffect(-2.87, 0.917)).toBe('not-significant');
    expect(sliceEffect(13.02, 0.000047)).toBe('significant');
    // Net positive but above alpha is not a win — that is the whole gate.
    expect(sliceEffect(4.2, 0.19)).toBe('not-significant');
  });

  it('marks the no-effect slice plainly in the table', () => {
    const html = renderToStaticMarkup(<SliceTable rows={sliceRows(comparison().gate)} />);
    expect(html).toContain('data-slice="number"');
    expect(html).toContain('data-effect="no-effect"');
    expect(html).toMatch(/no effect/);
    expect(html).toContain('data-effect="significant"');
    expect(html).toContain('+13.03pp');
  });

  it('says so rather than drawing an empty table when the gate has no split', () => {
    const html = renderToStaticMarkup(<SliceTable rows={sliceRows(null)} />);
    expect(html).toMatch(/not yet run/);
    expect(html).not.toContain('<table');
  });
});

const SWAP: SdkModelComparison = {
  version: 'sdk_v1',
  reference_model: 'claude-sonnet-5',
  models: [
    { ...SDK_ARM, model: 'claude-sonnet-5' },
    {
      ...SDK_ARM,
      model: 'claude-haiku-4-5-20251001',
      run_name: 'sdk-evalloop-test100-sdk_v1·s1-haiku-4-5',
      accuracy: 0.85,
      by_turn_type: { number: 0.95, program: 0.8 },
    },
  ],
  pairs: [
    {
      baseline_model: 'claude-sonnet-5',
      candidate_model: 'claude-haiku-4-5-20251001',
      baseline_run: SDK_ARM.run_name,
      candidate_run: 'sdk-evalloop-test100-sdk_v1·s1-haiku-4-5',
      n_compared: 349,
      delta_pp: -5.5,
      p_value: 0.003,
      ci: [-0.09, -0.02],
      fixed: 5,
      broken: 24,
      significant: true,
      by_turn_type: null,
    },
  ],
};

// ---------------------------------------------------------------------------
// The progression
// ---------------------------------------------------------------------------

describe('the progression', () => {
  it('reads the pipeline ends off the champion track and the SDK stages off the arm', () => {
    const stages = progression(TRACK, comparison(), [BASELINE_EXPERIMENT, experiment()], SWAP);
    expect(stages.map((s) => s.key)).toEqual([
      'pipeline_raw',
      'pipeline_optimised',
      'sdk_distilled',
      'sdk_optimised',
      'sdk_model_swap',
    ]);
    expect(stages.map((s) => s.version)).toEqual([
      'v2',
      'v8',
      'sdk_v1',
      'sdk_v2',
      'sdk_v1 · haiku-4-5',
    ]);
    expect(stages.every((s) => s.present)).toBe(true);
    expect(stages[4].accuracy).toBe(0.85);
    expect(stages[4].note).toMatch(/scoring pass/);
    // The cross-runtime read-out is not an optimisation attempt, so the last
    // stage must be the rejected rewrite rather than the distillation again.
    expect(stages[3].promoted).toBe(false);
    expect(stages[3].accuracy).toBeCloseTo(0.876791, 6);
  });

  it('marks the SDK stages absent when that arm has never run', () => {
    const stages = progression(TRACK, { ...comparison(), agent_sdk: EMPTY_ARM }, []);
    expect(stages.filter((s) => s.present).map((s) => s.key)).toEqual([
      'pipeline_raw',
      'pipeline_optimised',
    ]);
    expect(stages[2].accuracy).toBeNull();
    expect(stages[3].note).toMatch(/no optimisation experiment/);
    expect(stages[4].present).toBe(false);
    expect(stages[4].note).toMatch(/one model only/);
  });
});

describe('the progression chart', () => {
  it('draws one series per present stage and no column for an absent one', () => {
    const stages = progression(TRACK, comparison(), [BASELINE_EXPERIMENT, experiment()], SWAP);
    const html = renderToStaticMarkup(
      <ProgressionChart points={stages} pipelineBaseline={PIPELINE_ARM.accuracy} />,
    );
    for (const stage of stages) expect(html).toContain(`data-stage="${stage.key}"`);
    expect(html.match(/data-present="true"/g) ?? []).toHaveLength(5);
    expect(html.match(/data-present="false"/g) ?? []).toHaveLength(0);
    // Both reference lines, and the published one labelled as published.
    expect(html).toContain('data-reference="human"');
    expect(html).toContain('data-reference="incumbent"');
    expect(html).toMatch(/published, not measured here/);
    // An accessible name, not just a picture.
    expect(html).toContain('<title id="progression-title">');
    expect(html).toContain('<desc id="progression-desc">');
  });

  it('degrades a missing series to "not yet run" instead of a bar at zero', () => {
    const stages = progression(TRACK, { ...comparison(), agent_sdk: EMPTY_ARM }, []);
    const html = renderToStaticMarkup(
      <ProgressionChart points={stages} pipelineBaseline={PIPELINE_ARM.accuracy} />,
    );
    expect(html.match(/data-present="true"/g) ?? []).toHaveLength(2);
    expect(html.match(/data-present="false"/g) ?? []).toHaveLength(3);
    expect(html).toMatch(/not yet run/);
  });

  it('says so rather than drawing axes when nothing has been scored at all', () => {
    const stages = progression(
      [],
      { ...comparison(), pipeline: EMPTY_ARM, agent_sdk: EMPTY_ARM },
      [],
    );
    const html = renderToStaticMarkup(<ProgressionChart points={stages} pipelineBaseline={null} />);
    expect(html).not.toContain('<svg');
    expect(html).toMatch(/not yet run/);
  });
});

// ---------------------------------------------------------------------------
// The published ceiling, and two formatters
// ---------------------------------------------------------------------------

describe('the paper figures are read from the paper module, with their citation', () => {
  it('carries the human ceiling as a ratio and names where it came from', () => {
    expect(PAPER_HUMAN.exe).toBeCloseTo(0.8944, 4);
    expect(PAPER_HUMAN.prog).toBeCloseTo(0.8634, 4);
    expect(PAPER_HUMAN.citation).toMatch(/EMNLP 2022/);
    expect(PAPER_HUMAN.citation).toMatch(/Table/);
  });
});

describe('formatters keep absence and smallness distinguishable', () => {
  it('never prints a p-value as 0.000', () => {
    expect(formatP(0.000286)).toMatch(/e−4$/);
    expect(formatP(0.917241)).toBe('0.917');
    expect(formatP(null)).toBe('—');
  });

  it('reads wall clock at the scale a human uses', () => {
    expect(formatWall(786.8)).toBe('13 min');
    expect(formatWall(44)).toBe('44s');
    expect(formatWall(null)).toBe('—');
  });
});

// ---------------------------------------------------------------------------
// The model swap: one prompt, two models
// ---------------------------------------------------------------------------

describe('the model swap', () => {
  const swap = SWAP;

  it('shortens a model id the way the run name does', () => {
    expect(modelShortName('claude-sonnet-5')).toBe('sonnet-5');
    expect(modelShortName('claude-haiku-4-5-20251001')).toBe('haiku-4-5');
  });

  it('puts the reference first and reads the paired verdict off the pair', () => {
    const rows = modelRows(swap);
    expect(rows.map((r) => r.shortName)).toEqual(['sonnet-5', 'haiku-4-5']);
    expect(rows[0].effect).toBe('reference');
    expect(rows[0].deltaPp).toBeNull();
    expect(rows[1].effect).toBe('worse');
    expect(rows[1].deltaPp).toBe(-5.5);
    expect(rows[1].effectLabel).toMatch(/significantly worse/);
  });

  it('is empty with one model — a figure is not a comparison', () => {
    expect(modelRows({ ...swap, models: [swap.models[0]], pairs: [] })).toEqual([]);
    expect(modelRows(null)).toEqual([]);
    const html = renderToStaticMarkup(<ModelSwapTable rows={[]} />);
    expect(html).toMatch(/not yet run/);
    expect(html).not.toContain('<table');
  });

  it('marks a pair whose CSV was never read as not measured, not as a zero', () => {
    const rows = modelRows({ ...swap, pairs: [] });
    expect(rows[1].effect).toBe('not-measured');
    const html = renderToStaticMarkup(<ModelSwapTable rows={rows} />);
    expect(html).toContain('data-effect="not-measured"');
    expect(html).not.toMatch(/\+0\.00pp/);
  });

  it('renders one row per model with the model id visible', () => {
    const html = renderToStaticMarkup(<ModelSwapTable rows={modelRows(swap)} />);
    expect(html).toContain('data-model="sonnet-5"');
    expect(html).toContain('data-model="haiku-4-5"');
    expect(html).toContain('claude-haiku-4-5-20251001');
    expect(html).toContain('-5.50pp');
  });

  it('keys agents on model and effort, and shows the s19 verdict with the train pairing', () => {
    const agents: SdkModelComparison = {
      ...swap,
      reference_effort: 'high',
      models: [
        { ...swap.models[0], effort: 'high' },
        { ...swap.models[1], model: 'claude-haiku-5-5', effort: 'high' },
        { ...swap.models[1], model: 'claude-haiku-5-5', effort: 'low' },
      ],
      pairs: [
        {
          ...swap.pairs[0],
          candidate_model: 'claude-haiku-5-5',
          candidate_effort: 'high',
          delta_pp: -0.5,
          significant: false,
          train: {
            by_seed: [],
            pooled: {
              n_draws: 2,
              n_conversations: 200,
              baseline_full: 132,
              candidate_full: 130,
              p_worse: 0.4,
            },
          },
          verdict: {
            outcome: 'matches',
            reason: 'within margin',
            margin_pp: 3,
            checks: [],
          },
        },
      ],
    };
    const rows = modelRows(agents);
    expect(rows.map((r) => [r.shortName, r.effort])).toEqual([
      ['sonnet-5', 'high'],
      ['haiku-5-5', 'high'],
      ['haiku-5-5', 'low'],
    ]);
    expect(rows[1].verdict).toBe('matches');
    expect(rows[2].effect).toBe('not-measured');
    const html = renderToStaticMarkup(<ModelSwapTable rows={rows} />);
    expect(html).toContain('data-verdict="matches"');
    expect(html).toContain('matches the champion');
    expect(html).toContain('132 → 130');
  });

  it('prints the interval in points, like the delta beside it (ci is stored as a fraction)', () => {
    const html = renderToStaticMarkup(<ModelSwapTable rows={modelRows(swap)} />);
    // fixture ci [-0.09, -0.02] → -9.0pp … -2.0pp, never -0.09pp
    expect(html).toContain('-9.0pp');
    expect(html).toContain('-2.0pp');
    expect(html).not.toContain('-0.09pp');
  });
});

// ---------------------------------------------------------------------------
// The progression with every challenger agent (s21)
// ---------------------------------------------------------------------------

const CHAMPION_AGENT = {
  version: 'sdk_v1',
  runtime: 'agent_sdk' as const,
  model: 'claude-sonnet-5',
  effort: 'high',
};

function pair(
  version: string,
  model: string,
  outcome: 'beats' | 'matches' | 'falls_short',
  at: number,
  accuracy: number,
  run: string,
): Challengers['pairs'][number] {
  return {
    version,
    baseline_version: 'sdk_v1',
    baseline_model: 'claude-sonnet-5',
    candidate_model: model,
    baseline_effort: 'high',
    candidate_effort: 'high',
    baseline_run: 'sdk-evalloop-test100-sdk_v1·s1-20260905_174627',
    candidate_run: run,
    n_compared: 349,
    delta_pp: (accuracy - 0.905444) * 100,
    p_value: 0.04,
    ci: [-0.068, 0.003],
    fixed: 10,
    broken: 21,
    significant: false,
    at,
    candidate_accuracy: accuracy,
    verdict: { outcome, reason: `${outcome}: …`, margin_pp: 3, checks: [] },
  };
}

// Out of order on purpose: the points must come out in the order they ran.
const CHALLENGERS: Challengers = {
  champion: {
    version: 'sdk_v1',
    model: 'claude-sonnet-5',
    effort: 'high',
    run_name: 'sdk-evalloop-test100-sdk_v1·s1-20260905_174627',
    accuracy: 0.905444,
  },
  margin_pp: 3,
  pairs: [
    pair(
      'sdk_v3',
      'claude-haiku-5-5',
      'matches',
      30,
      0.896,
      'sdk-evalloop-test100-sdk_v3·s3-haiku-5-5-high-20261015_205249',
    ),
    pair(
      'sdk_v1',
      'claude-haiku-4-5-20251001',
      'falls_short',
      10,
      0.873926,
      'sdk-evalloop-test100-sdk_v1·s1-haiku-4-5-20260906_090517',
    ),
    pair(
      'sdk_v1',
      'claude-haiku-5-5',
      'falls_short',
      20,
      0.873926,
      'sdk-evalloop-test100-sdk_v1·s1-haiku-5-5-high-20261015_175310',
    ),
  ],
};

function pass(
  run_name: string,
  version: string,
  date: string,
  extra: Partial<TestPass> = {},
): TestPass {
  return {
    at_ms: null,
    date,
    run_id: null,
    run_name,
    version,
    runtime: version.startsWith('sdk') ? 'agent_sdk' : 'pipeline',
    model: version.startsWith('sdk') ? 'claude-sonnet-5' : 'deepseek-v4-flash',
    effort: version.startsWith('sdk') ? 'high' : null,
    accuracy: 0.8,
    n_scored: 349,
    n_questions: 349,
    complete: true,
    cost: null,
    champion_run: false,
    label: null,
    outcome: 'scored',
    reason: '',
    vs: null,
    ...extra,
  };
}

const TIMELINE: TestPass[] = [
  pass('evalloop-test100-v2·t2p2r2c2-20260904_032533', 'v2', '2026-09-04', {
    champion_run: true,
  }),
  pass('evalloop-test100-v8·t2p2r5c2-20260904_052739', 'v8', '2026-09-04', {
    outcome: 'promoted',
  }),
  pass('sdk-evalloop-test100-sdk_v1·s1', 'sdk_v1', '2026-09-05', {
    outcome: 'promoted',
  }),
  pass('sdk-evalloop-test100-sdk_v2·s2-20260905_182812', 'sdk_v2', '2026-09-05', {
    outcome: 'rejected',
  }),
  pass(CHALLENGERS.pairs[1].candidate_run as string, 'sdk_v1', '2026-09-06', {
    model: 'claude-haiku-4-5-20251001',
  }),
  pass(CHALLENGERS.pairs[2].candidate_run as string, 'sdk_v1', '2026-10-15', {
    model: 'claude-haiku-5-5',
  }),
  pass(CHALLENGERS.pairs[0].candidate_run as string, 'sdk_v3', '2026-10-15', {
    model: 'claude-haiku-5-5',
  }),
];

const SDK_ARM_AGENT: RuntimeArm = {
  ...SDK_ARM,
  model: 'claude-sonnet-5',
  effort: 'high',
};

function withChallengers(champion = CHAMPION_AGENT) {
  return progression(
    TRACK,
    comparison({ agent_sdk: SDK_ARM_AGENT }),
    [BASELINE_EXPERIMENT, experiment()],
    SWAP,
    { challengers: CHALLENGERS, timeline: TIMELINE, championAgent: champion },
  );
}

describe('the progression with every challenger agent (s21)', () => {
  it('draws one dated point per challenger, in the order they ran', () => {
    const stages = withChallengers();
    expect(stages.map((s) => s.key)).toEqual([
      'pipeline_raw',
      'pipeline_optimised',
      'sdk_distilled',
      'sdk_optimised',
      'challenger',
      'challenger',
      'challenger',
    ]);
    expect(new Set(stages.map((s) => s.id)).size).toBe(stages.length);
    const challengers = stages.slice(4);
    expect(challengers.map((s) => [s.version, s.model])).toEqual([
      ['sdk_v1', 'claude-haiku-4-5-20251001'],
      ['sdk_v1', 'claude-haiku-5-5'],
      ['sdk_v3', 'claude-haiku-5-5'],
    ]);
    expect(challengers.map((s) => s.date)).toEqual(['2026-09-06', '2026-10-15', '2026-10-15']);
    expect(stages[0].date).toBe('2026-09-04');
    expect(stages[2].date).toBe('2026-09-05');
    // The single model-swap stage is gone: the challengers replace it.
    expect(stages.some((s) => s.key === 'sdk_model_swap')).toBe(false);
  });

  it('tones each bar by what the record says, not by its position', () => {
    const stages = withChallengers();
    const tone = Object.fromEntries(stages.map((s) => [s.id, s.tone]));
    expect(tone.sdk_distilled).toBe('current');
    expect(tone.sdk_optimised).toBe('fail');
    expect(tone['challenger:sdk_v1|claude-haiku-4-5-20251001|high']).toBe('fail');
    expect(tone['challenger:sdk_v1|claude-haiku-5-5|high']).toBe('fail');
    expect(tone['challenger:sdk_v3|claude-haiku-5-5|high']).toBe('ok');
    expect(tone.pipeline_optimised).toBe('ok');
    expect(tone.pipeline_raw).toBe('neutral');
  });

  it('names a challenger with its own prompt as one, not as a model swap', () => {
    const stages = withChallengers();
    const v3 = stages.find((s) => s.version === 'sdk_v3');
    const swap = stages.find((s) => s.model === 'claude-haiku-5-5' && s.version === 'sdk_v1');
    expect(v3?.stage).toBe('new prompt');
    expect(v3?.note).toMatch(/a prompt the loop wrote for claude-haiku-5-5/);
    expect(swap?.stage).toBe('model swap');
  });

  it('moves the accent to a challenger the registry promoted', () => {
    const stages = withChallengers({
      ...CHAMPION_AGENT,
      version: 'sdk_v3',
      model: 'claude-haiku-5-5',
    });
    expect(stages.find((s) => s.version === 'sdk_v3')?.tone).toBe('current');
    // The distilled prompt on Sonnet still passed its gate; it just no longer serves.
    expect(stages.find((s) => s.key === 'sdk_distilled')?.tone).toBe('ok');
  });

  it('draws a falls-short challenger hatched in red, never in the pass colour', () => {
    const html = renderToStaticMarkup(
      <ProgressionChart points={withChallengers()} pipelineBaseline={PIPELINE_ARM.accuracy} />,
    );
    const failBars = html.match(/class="bar [a-z_]+ t-fail"[^>]*style="fill:url\(#hatch-/g) ?? [];
    // sdk_v2 rejected, and both Haiku swaps falling short.
    expect(failBars).toHaveLength(3);
    expect(html.match(/data-tone="fail"/g) ?? []).toHaveLength(3 * 2); // svg column + row
    expect(html).toContain('✕ falls short');
    expect(html).toContain('✕ rejected');
    expect(html).toContain('● champion');
    expect(html).toContain('● matches');
  });

  it('keeps every column label inside its slot, however many challengers there are', () => {
    const stages = withChallengers();
    for (const p of stages) {
      for (const line of columnLines(p)) expect(line.length).toBeLessThanOrEqual(SLOT_CHARS);
    }
    const many = [...stages, ...stages.slice(4), ...stages.slice(4)];
    const html = renderToStaticMarkup(<ProgressionChart points={many} pipelineBaseline={null} />);
    const width = Number(/viewBox="0 0 (\d+)/.exec(html)?.[1]);
    // The chart widens a slot per column instead of squeezing them.
    expect((width - 52 - 200) / many.length).toBeGreaterThanOrEqual(112);
  });

  it('dates and labels every column, with the pipeline effort said to be absent', () => {
    const [raw, , distilled] = withChallengers();
    expect(columnLines(raw)).toEqual([
      'raw',
      'llm-v2',
      'deepseek ×4',
      '—',
      '4 Sep 2026',
      '○ baseline',
    ]);
    expect(columnLines(distilled)).toEqual([
      'distilled',
      'sdk-v1',
      'sonnet-5',
      'high',
      '5 Sep 2026',
      '● champion',
    ]);
  });

  it('turns a falls-short model swap red on the pre-s21 record too', () => {
    const swap: SdkModelComparison = {
      ...SWAP,
      pairs: SWAP.pairs.map((p) => ({
        ...p,
        verdict: {
          outcome: 'falls_short',
          reason: 'falls_short: …',
          margin_pp: 3,
          checks: [],
        },
      })),
    };
    const stages = progression(TRACK, comparison(), [BASELINE_EXPERIMENT, experiment()], swap);
    expect(stages[4].key).toBe('sdk_model_swap');
    expect(stages[4].tone).toBe('fail');
  });

  it('formats a run day as written, whatever the timezone', () => {
    expect(formatDay('2026-09-06')).toBe('6 Sep 2026');
    expect(formatDay('2026-10-15', false)).toBe('15 Oct');
    expect(formatDay(null)).toBe('—');
    expect(formatDay('not a date')).toBe('—');
  });
});
