import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  campaignHeadline,
  campaignNames,
  championAgentLabel,
  championOf,
  experimentRows,
  runAgent,
  runsForSplit,
  trackDates,
} from './campaignStory';
import { ChampionChart, ExperimentTable } from './Experiments';
import type { CampaignExperiment, CampaignsResponse, ChampionPoint, TestPass } from './api';

/**
 * The Campaigns page's claims, held to the payload: the headline is the champion
 * agent against the pipeline champion, both arms share the experiment table,
 * every run says its split and agent, and the track is dated by each version's
 * own pass rather than by the experiment that promoted it.
 */

function experiment(overrides: Partial<CampaignExperiment>): CampaignExperiment {
  return {
    label: 'c01-e03',
    campaign: 'c01',
    target_agent: 'retriever',
    baseline_version: 'v2',
    candidate_version: 'v8',
    promoted: true,
    at: null,
    accuracy_delta: 0.045845,
    cluster_p_one_sided: 0.040428,
    delta_ci_lo: 0.004,
    delta_ci_hi: 0.088,
    n_compared: 349,
    fixed: 34,
    broken: 18,
    accuracy_baseline: 0.770774,
    accuracy_candidate: 0.816619,
    panel_baseline: {},
    panel_candidate: {},
    summary_of_changes: '',
    rationale: '',
    diff: '',
    ...overrides,
  };
}

const TRACK: ChampionPoint[] = [
  {
    version: 'v2',
    at: 1788463659667,
    accuracy: 0.770774,
    panel: {},
    moved_by: null,
    target_agent: null,
  },
  {
    version: 'v8',
    at: 1788463659667,
    accuracy: 0.816619,
    panel: {},
    moved_by: 'c01-e03',
    target_agent: 'retriever',
  },
];

function pass(version: string, date: string, extra: Partial<TestPass> = {}): TestPass {
  return {
    at_ms: null,
    date,
    run_id: null,
    run_name: `evalloop-test100-${version}`,
    version,
    runtime: 'pipeline',
    model: 'deepseek-v4-flash',
    effort: null,
    accuracy: 0.8,
    n_scored: 349,
    n_questions: 349,
    complete: true,
    cost: null,
    champion_run: true,
    label: null,
    outcome: 'champion',
    reason: '',
    vs: null,
    ...extra,
  };
}

const DATA = {
  champion: 'sdk_v1',
  champion_agent: {
    version: 'sdk_v1',
    runtime: 'agent_sdk',
    model: 'claude-sonnet-5',
    effort: 'high',
  },
  champion_accuracy: 0.905444,
  champion_panel: {},
  pipeline_champion: 'v8',
  pipeline_champion_accuracy: 0.816619,
  rule: '',
  generated_at: '',
  split: {},
  campaigns: [
    {
      name: 'c01',
      n_experiments: 3,
      n_promoted: 1,
      n_remaining: 2,
      cap: 5,
      runtime: 'pipeline',
    },
  ],
  sdk_campaigns: [
    {
      name: 'h01',
      n_experiments: 1,
      n_promoted: 0,
      n_remaining: 1,
      cap: 2,
      runtime: 'agent_sdk',
    },
    {
      name: 's01',
      n_experiments: 2,
      n_promoted: 1,
      n_remaining: 0,
      cap: 2,
      runtime: 'agent_sdk',
    },
  ],
  experiments: [
    experiment({ label: 'c01-e01', candidate_version: 'v6', promoted: false }),
    experiment({}),
  ],
  sdk_experiments: [
    experiment({
      label: 's01-e02',
      campaign: 's01',
      target_agent: 'retriever/wrong-period',
      target_class: 'retriever/wrong-period',
      baseline_version: 'sdk_v1',
      candidate_version: 'sdk_v2',
      promoted: false,
      runtime: 'agent_sdk',
      accuracy_baseline: 0.905444,
      accuracy_candidate: 0.876791,
    }),
  ],
  champion_track: TRACK,
  test_timeline: [
    pass('v2', '2026-09-04'),
    pass('v8', '2026-09-04', { outcome: 'promoted', champion_run: true }),
  ],
} as unknown as CampaignsResponse;

describe('the campaigns headline', () => {
  it('leads with the champion agent and sets it against the pipeline champion', () => {
    expect(championAgentLabel(DATA)).toBe('sdk-v1 · sonnet-5 · high');
    expect(campaignHeadline(DATA)).toEqual({
      nExperiments: 3,
      nPromoted: 1,
      champion: 'sdk-v1 · sonnet-5 · high',
      championAccuracy: 0.905444,
      pipelineChampion: 'v8',
      pipelineAccuracy: 0.816619,
    });
  });

  it('says there is no champion agent rather than naming the pipeline one', () => {
    expect(championAgentLabel({ ...DATA, champion_agent: null })).toBeNull();
  });
});

describe('the experiment table', () => {
  it('puts both arms in one list, each row naming its runtime', () => {
    const rows = experimentRows(DATA);
    expect(rows.map((r) => [r.label, r.runtime])).toEqual([
      ['c01-e01', 'pipeline'],
      ['c01-e03', 'pipeline'],
      ['s01-e02', 'agent_sdk'],
    ]);
    expect(campaignNames(DATA)).toEqual(['c01', 'h01', 's01']);
  });

  it('marks each arm’s own champion: the bundle for the pipeline, the agent’s prompt for the session', () => {
    const [, v8, sdkV2] = experimentRows(DATA);
    expect(championOf(v8, DATA)).toBe('v8');
    expect(championOf(sdkV2, DATA)).toBe('sdk_v1');
  });

  it('renders a runtime column and a failure class as the session row’s target', () => {
    const rows = experimentRows(DATA);
    const html = renderToStaticMarkup(
      <ExperimentTable experiments={rows} championFor={(r) => championOf(r, DATA)} />,
    );
    expect(html).toContain('<th>Runtime</th>');
    expect(html.match(/data-runtime="agent_sdk"/g) ?? []).toHaveLength(1);
    expect(html.match(/data-runtime="pipeline"/g) ?? []).toHaveLength(2);
    expect(html).toContain('retriever/wrong-period');
    expect(html).toContain('90.5% → 87.7%');
  });
});

describe('the runs table', () => {
  const run = (kind: string, params: Record<string, string>) => ({
    kind,
    params,
  });

  it('reads the agent off an sdk run’s own params, not the fingerprint’s lm_mini', () => {
    expect(
      runAgent(
        run('evalloop', {
          split: 'test',
          runtime: 'agent_sdk',
          sdk_model: 'claude-haiku-5-5',
          sdk_effort: 'high',
          lm_mini: 'deepseek-v4-flash',
        }),
      ),
    ).toEqual({
      split: 'test',
      runtime: 'agent_sdk',
      model: 'claude-haiku-5-5',
      effort: 'high',
      effortWhy: null,
    });
  });

  it('reads an open-agent run as its own runtime and model, not the pipeline’s', () => {
    expect(
      runAgent(
        run('evalloop', {
          split: 'test',
          runtime: 'open_agent',
          open_model: 'deepseek-flash',
          open_effort: 'high',
          lm_mini: 'deepseek-v4-flash',
        }),
      ),
    ).toEqual({
      split: 'test',
      runtime: 'open_agent',
      model: 'deepseek-flash',
      effort: 'high',
      effortWhy: null,
    });
  });

  it('keeps an open-agent experiment’s runtime and judges it against the champion agent', () => {
    const data = {
      ...DATA,
      sdk_experiments: [
        ...(DATA.sdk_experiments ?? []),
        { ...(DATA.sdk_experiments ?? [])[0], label: 'o01-e01', runtime: 'open_agent' },
      ],
    } as CampaignsResponse;
    const open = experimentRows(data).find((r) => r.label === 'o01-e01');
    expect(open?.runtime).toBe('open_agent');
  });

  it('says why an effort is absent rather than guessing one', () => {
    const old = runAgent(
      run('evalloop', {
        split: 'test',
        runtime: 'agent_sdk',
        sdk_model: 'claude-sonnet-5',
      }),
    );
    expect(old.effort).toBeNull();
    expect(old.effortWhy).toMatch(/predates the sdk_effort param/);
    const pipeline = runAgent(run('evalloop', { split: 'train', lm_mini: 'deepseek-v4-flash' }));
    expect(pipeline).toMatchObject({
      runtime: 'pipeline',
      model: 'deepseek-v4-flash',
      effort: null,
    });
    expect(pipeline.effortWhy).toMatch(/Claude agents only/);
    // A run that answered nothing has no agent at all.
    expect(runAgent(run('smoke', { lm_mini: 'deepseek-v4-flash' }))).toMatchObject({
      split: null,
      runtime: null,
      model: null,
    });
  });

  it('defaults to the split that promotes, and keeps the rest one filter away', () => {
    const runs = [
      run('evalloop', { split: 'test' }),
      run('evalloop', { split: 'train' }),
      run('smoke', {}),
    ];
    expect(runsForSplit(runs, 'test')).toHaveLength(1);
    expect(runsForSplit(runs, 'train')).toHaveLength(1);
    expect(runsForSplit(runs, '')).toHaveLength(3);
  });
});

describe('the pipeline champion’s track', () => {
  it('dates each point by its own pass, not by the experiment that promoted it', () => {
    expect(trackDates(TRACK, DATA.test_timeline)).toEqual(['2026-09-04', '2026-09-04']);
    expect(trackDates(TRACK, [])).toEqual([null, null]);
    const html = renderToStaticMarkup(
      <ChampionChart track={TRACK} dates={['2026-09-03', '2026-09-04']} />,
    );
    expect(html).toContain('3 Sep 2026');
    expect(html).toContain('4 Sep 2026');
  });

  it('prints the dash for a point with no pass on record', () => {
    const html = renderToStaticMarkup(<ChampionChart track={TRACK} />);
    expect(html.match(/data-date="true">—</g) ?? []).toHaveLength(2);
  });
});
