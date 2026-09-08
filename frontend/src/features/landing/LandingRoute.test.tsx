import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TooltipProvider } from '../../components/ui/tooltip';
import { RightPane } from './LandingRoute';
import type { BoardData } from './useBoardData';
import type { CampaignsResponse, JudgeVerdict } from '../admin/api';

/**
 * s13: the landing HUD's judge tile leads with the verdict, same as the
 * admin banner and the published write-up — so a later judge that clears
 * the bar must not still read "not significant" beside a positive delta.
 */

function verdict(over: Partial<JudgeVerdict> = {}): JudgeVerdict {
  return {
    split: 'test',
    version: 'judge_j1',
    baseline_accuracy: 0.9054,
    high_band_accuracy: 0.9146,
    delta_pp: 0.91,
    high_band_accuracy_ci: [0.8787, 0.9522],
    significant: false,
    meets_target: false,
    coverage: 0.9054,
    failure_capture: 0.1818,
    n_withheld: 33,
    n_false_alarms: 27,
    n_failures_caught: 6,
    n_wrong: 33,
    error_target: 0.01,
    recommendation: 'advisory only — do not gate on it',
    ...over,
  };
}

function boardWithVerdict(v: JudgeVerdict | null): BoardData {
  const campaigns: CampaignsResponse = {
    champion: 'v11',
    champion_accuracy: 0.8,
    champion_panel: {},
    rule: 'r1',
    generated_at: '2026-09-08T00:00:00Z',
    split: {},
    campaigns: [],
    experiments: [],
    champion_track: [],
    judge: {
      dataset: null,
      champion: 'judge_j1',
      versions: [],
      gates: [],
      verdict: v,
    },
  };

  return {
    health: null,
    isDemo: false,
    champion: 'v11',
    servingChampion: 'sdk_v1',
    servingRuntime: 'agent_sdk',
    campaigns,
    versions: undefined,
    championVersion: undefined,
    championHoldout: undefined,
    versionHoldouts: undefined,
    previousVersion: undefined,
    metricsSource: 'serving',
    metrics: null,
    metricsGeneratedAt: undefined,
    metricsWindow: undefined,
    traceCaptureEnabled: undefined,
    metricsLoading: false,
    gate: undefined,
    gateCandidate: undefined,
    recorded: undefined,
    recordedLoading: false,
    loading: false,
    error: null,
  };
}

function renderJudgeTile(v: JudgeVerdict | null): string {
  // The readiness strip inside the pane reads /eval/readiness through
  // react-query; with no fetch it stays in its loading skeleton, which is fine.
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return renderToStaticMarkup(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <TooltipProvider>
          <RightPane board={boardWithVerdict(v)} />
        </TooltipProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('the landing judge card', () => {
  it('says the band fails to separate from the unjudged baseline when not significant', () => {
    const html = renderJudgeTile(verdict());
    expect(html).toContain('no effect');
    expect(html).toContain('fails to separate from it');
    expect(html).toContain('data-significant="false"');
  });

  it('says the band separates from the unjudged baseline when significant', () => {
    const html = renderJudgeTile(verdict({ significant: true }));
    expect(html).toContain('separates from it');
    expect(html).not.toContain('fails to separate from it');
    expect(html).toContain('data-significant="true"');
  });

  it('states the sealed holdout under the tiles instead of an empty tile', () => {
    const html = renderJudgeTile(verdict());
    expect(html).toContain('never been opened');
    expect(html).not.toContain('hud-tile-out-of-sample-accuracy');
  });
});
