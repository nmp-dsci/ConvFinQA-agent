/**
 * Optimise › Campaigns, as data shaping.
 *
 * The page's claims are derived here from `/eval/campaigns` and
 * `/admin/experiments`, so a test can hold them to the payload: the headline
 * names the champion *agent* (prompt, model, effort) and sets it against the
 * pipeline champion, both arms' experiments share one table, and every run
 * says which split and which agent it is.
 */

import type { CampaignExperiment, CampaignsResponse, ChampionPoint, TestPass } from './api';
import type { ExperimentRun } from '../../types';
import { NO_VALUE } from '../landing/format';
import { modelShortName } from './runtimeStory';

/** `open_agent` (s25): the sdk prompt on an open model in an open harness. */
export type Runtime = 'pipeline' | 'agent_sdk' | 'open_agent';

/** One experiment row with the arm it belongs to, read off the list it came from. */
export interface ExperimentRow extends CampaignExperiment {
  runtime: Runtime;
}

/**
 * Both arms' experiments in one list, pipeline then single-session, each in its
 * recorded order. The server keeps them apart (`experiments` is the pipeline's,
 * `sdk_experiments` the session's) because one targets a subagent and the
 * other a failure class; the row's `runtime` is what tells them apart here.
 */
export function experimentRows(data: CampaignsResponse | undefined): ExperimentRow[] {
  return [
    ...(data?.experiments ?? []).map((e) => ({
      ...e,
      runtime: 'pipeline' as const,
    })),
    ...(data?.sdk_experiments ?? []).map((e) => ({
      ...e,
      // A session-runtime experiment is Claude's unless its row says open_agent.
      runtime: e.runtime === 'open_agent' ? ('open_agent' as const) : ('agent_sdk' as const),
    })),
  ];
}

/** Every campaign's name, both arms, in the order the server listed them. */
export function campaignNames(data: CampaignsResponse | undefined): string[] {
  return [...(data?.campaigns ?? []), ...(data?.sdk_campaigns ?? [])].map((c) => c.name);
}

/** The champion's version for the row's own arm: the agent for the session, the bundle for the pipeline. */
export function championOf(row: ExperimentRow, data: CampaignsResponse | undefined): string | null {
  return row.runtime !== 'pipeline'
    ? (data?.champion_agent?.version ?? data?.champion ?? null)
    : (data?.pipeline_champion ?? null);
}

/** `sdk-v1 · sonnet-5 · high` — the champion as the agent it is, or null before there is one. */
export function championAgentLabel(data: CampaignsResponse | undefined): string | null {
  const agent = data?.champion_agent;
  if (!agent) return null;
  const head = agent.version.replace(/^sdk_v/, 'sdk-v').replace(/^v(\d)/, 'llm-v$1');
  return `${head} · ${modelShortName(agent.model)} · ${agent.effort}`;
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

export interface RunAgent {
  /** `train` / `test` / `holdout`; null for a run that scored no split (a smoke, a gate). */
  split: string | null;
  /** null for a run that answered nothing (diagnose, propose, gate, smoke). */
  runtime: Runtime | null;
  model: string | null;
  effort: string | null;
  /** Why `effort` is absent, when it is. */
  effortWhy: string | null;
}

/**
 * Which split and which agent a run is, from its own params.
 *
 * Only an eval-loop run (`kind = evalloop`) answered questions. Every run
 * carries the bundle fingerprint, `lm_mini` included, so that param says
 * nothing about an sdk run: its model is `sdk_model`. Effort is a param since
 * 2026-10-15; an older sdk run has none, and the page says so rather than
 * guessing — `story.json` dates it against the low pin.
 */
export function runAgent(run: Pick<ExperimentRun, 'kind' | 'params'>): RunAgent {
  const p = run.params ?? {};
  const split = p.split || null;
  if (run.kind !== 'evalloop') {
    return { split, runtime: null, model: null, effort: null, effortWhy: null };
  }
  if (p.runtime === 'open_agent') {
    return {
      split,
      runtime: 'open_agent',
      model: p.open_model || null,
      effort: p.open_effort || null,
      effortWhy: p.open_effort ? null : 'run has no open_effort param',
    };
  }
  if (p.runtime === 'agent_sdk') {
    return {
      split,
      runtime: 'agent_sdk',
      model: p.sdk_model || null,
      effort: p.sdk_effort || null,
      effortWhy: p.sdk_effort ? null : 'run predates the sdk_effort param (2026-10-15)',
    };
  }
  return {
    split,
    runtime: 'pipeline',
    model: p.lm_mini || null,
    effort: null,
    effortWhy: 'effort applies to the Claude agents only',
  };
}

/** The split chips the run table offers, test first: it is the split that promotes. */
export const SPLIT_FILTERS = ['test', 'train', ''] as const;
export type SplitFilter = (typeof SPLIT_FILTERS)[number];

export function splitFilterLabel(filter: SplitFilter): string {
  return filter === 'test'
    ? 'test — promotes'
    : filter === 'train'
      ? 'train — writes versions, never promotes them'
      : 'every run';
}

export function runsForSplit<T extends Pick<ExperimentRun, 'kind' | 'params'>>(
  runs: T[],
  filter: SplitFilter,
): T[] {
  return filter ? runs.filter((r) => runAgent(r).split === filter) : runs;
}

// ---------------------------------------------------------------------------
// The headline and the track
// ---------------------------------------------------------------------------

export interface CampaignHeadline {
  nExperiments: number;
  nPromoted: number;
  champion: string | null;
  championAccuracy: number | null;
  pipelineChampion: string | null;
  pipelineAccuracy: number | null;
}

/** The numbers the h1 is made of, across both arms. */
export function campaignHeadline(data: CampaignsResponse | undefined): CampaignHeadline {
  const rows = experimentRows(data);
  return {
    nExperiments: rows.length,
    nPromoted: rows.filter((e) => e.promoted).length,
    champion: championAgentLabel(data),
    championAccuracy: data?.champion_accuracy ?? null,
    pipelineChampion: data?.pipeline_champion ?? null,
    pipelineAccuracy: data?.pipeline_champion_accuracy ?? null,
  };
}

/**
 * The day each track point's version ran its gate-split pass, from
 * `test_timeline`. The track's own `at` is the time of the experiment that
 * promoted, and both ends of a one-promotion track carry it — v2 and v8 read
 * as the same moment — so the date shown is the pass's own.
 */
export function trackDates(
  track: ChampionPoint[],
  timeline: TestPass[] | undefined,
): Array<string | null> {
  const passes = (timeline ?? []).filter((r) => r.runtime === 'pipeline' && r.complete);
  return track.map((point) => {
    const own =
      passes.find((r) => r.version === point.version && r.champion_run) ??
      passes.find((r) => r.version === point.version);
    return own?.date ?? null;
  });
}

/** A run cell for an absent value: the dash, and why. */
export const ABSENT = NO_VALUE;
