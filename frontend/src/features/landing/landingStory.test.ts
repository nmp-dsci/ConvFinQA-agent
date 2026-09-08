import { describe, expect, it } from 'vitest';
import type { CampaignsResponse } from '../admin/api';
import { judgeSentence, landingStory } from './landingStory';

function campaigns(over: Partial<CampaignsResponse> = {}): CampaignsResponse {
  return {
    champion: 'v8',
    champion_accuracy: 0.8166,
    champion_panel: {},
    rule: 'r1',
    generated_at: '2026-09-08T00:00:00Z',
    split: { gate_questions: 349, gate_reports: 100 },
    campaigns: [],
    experiments: [],
    champion_track: [],
    runtime_comparison: {
      pipeline: {
        version: 'v8',
        run_name: null,
        accuracy: 0.8166,
        by_turn_type: null,
        panel: null,
        cost: null,
        wall: null,
      },
      agent_sdk: {
        version: 'sdk_v1',
        run_name: null,
        accuracy: 0.9054,
        by_turn_type: null,
        panel: null,
        cost: 27.62,
        wall: null,
      },
      gate: {
        delta_pp: 8.88,
        p_value: 0.000286,
        ci: [0.042, 0.137],
        promoted: true,
        candidate_version: 'sdk_v1',
      },
    },
    ...over,
  };
}

describe('the landing headline', () => {
  it('claims human-expert accuracy only when the sdk arm is at or above the paper figure', () => {
    const story = landingStory(campaigns(), false);
    expect(story.headline.emphasis).toBe('human-expert');
    expect(story.measured).toBe(true);
    expect(story.caveat).toMatch(/contamination not excluded/);
  });

  it('says how far below the human figure it is when it is below', () => {
    const c = campaigns();
    c.runtime_comparison!.agent_sdk.accuracy = 0.85;
    const story = landingStory(c, false);
    expect(story.headline.before).toMatch(/within/);
    expect(story.headline.emphasis).toMatch(/pp$/);
    expect(story.headline.emphasis).not.toMatch(/^\+/);
  });

  it('falls back to the plain description when nothing has been scored', () => {
    const story = landingStory(undefined, false);
    expect(story.headline.emphasis).toBe('dependent');
    expect(story.caveat).toBeNull();
    expect(story.proof[0].value).toBe('—');
    expect(story.proof[2].label).toMatch(/no cross-runtime gate/);
  });

  it('prints the three proofs with their baselines', () => {
    const story = landingStory(campaigns(), false);
    expect(story.proof.map((p) => p.value)).toEqual(['90.5%', '89.4%', '+8.9pp']);
    expect(story.proof[0].label).toContain('349 unseen questions');
    expect(story.proof[1].label).toMatch(/human expert · paper/);
    expect(story.proof[2].label).toMatch(/paired · p=3e-4/);
  });

  it('states the sealed holdout with the split size', () => {
    expect(landingStory(campaigns(), false).holdout).toMatch(
      /349 questions across 100 conversations.*never been opened/,
    );
  });

  it('describes the replay in demo mode', () => {
    expect(landingStory(campaigns(), true).lede).toMatch(/holds no API key/);
    expect(landingStory(campaigns(), false).lede).toMatch(/streams its stages live/);
  });
});

describe('the judge sentence', () => {
  const verdict = {
    split: 'test',
    version: 'judge_j1',
    baseline_accuracy: 0.9054,
    high_band_accuracy: 0.9146,
    delta_pp: 0.91,
    high_band_accuracy_ci: [0.8787, 0.9522] as [number, number],
    significant: false,
    meets_target: false,
    coverage: 0.9054,
    failure_capture: 0.1818,
    n_withheld: 33,
    n_false_alarms: 27,
    n_failures_caught: 6,
    n_wrong: 33,
    error_target: 0.01,
    recommendation: 'advisory only',
  };

  it('says the band fails to separate when not significant', () => {
    const s = judgeSentence(campaigns({ judge: { dataset: null, champion: 'judge_j1', versions: [], gates: [], verdict } }));
    expect(s?.headline).toBe('no effect');
    expect(s?.body).toMatch(/fails to separate from it/);
    expect(s?.body).toMatch(/withholds 33 answers to remove 6/);
  });

  it('says it separates when significant', () => {
    const s = judgeSentence(
      campaigns({
        judge: { dataset: null, champion: 'judge_j1', versions: [], gates: [], verdict: { ...verdict, significant: true } },
      }),
    );
    expect(s?.headline).toMatch(/separates from it/);
    expect(s?.body).not.toMatch(/fails to separate/);
  });

  it('is null with no verdict', () => {
    expect(judgeSentence(campaigns())).toBeNull();
  });
});
