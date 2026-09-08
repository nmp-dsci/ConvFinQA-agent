import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { HudTile } from './HudTile';
import { LampStrip } from './LampStrip';
import { RecordedConversations } from './RecordedConversations';
import { ProgressionChart } from '../admin/ProgressionChart';
import { versionLabel } from '../admin/lib';
import { progression } from '../admin/runtimeStory';
import { NO_VALUE, formatLatency, formatPercent, formatPointsDelta, formatUsd } from './format';
import { judgeSentence, landingStory } from './landingStory';
import { useBoardData } from './useBoardData';
import type { BoardData } from './useBoardData';
import { useStore } from '../../store';
import { ReadinessStrip } from '../readiness/Readiness';

/**
 * The front door at `/`.
 *
 * Two panes, and the split is the argument. On the left, the outcome in one
 * sentence with its baseline and its caveat, and the doors into the product.
 * On the right, the record: the five-stage progression on a zero baseline —
 * the star — then four tiles with a baseline on every number, and the judge
 * tried last.
 *
 * Three rules this file must not break:
 *
 *  1. **No number is written here.** Every figure comes from a query in
 *     `useBoardData`; every sentence comes from `landingStory`, a pure
 *     function with a test per branch. If a read fails or has nothing, the
 *     tile says so.
 *  2. **`null` is not `0`.** `/metrics/production` returns `null` with
 *     `n_measured: 0` for latency and cost until someone pays for a metered
 *     eval run. An empty tile prints an em dash and the reason; it never
 *     prints a zero and it never draws a flat line.
 *  3. **Every figure is the gate split, and the page says so.** The holdout
 *     has never been opened; the sentence under the tiles states it, rather
 *     than an empty tile implying it.
 */
export function LandingRoute() {
  const board = useBoardData(3);

  return (
    <div data-testid="landing-board" className="h-full overflow-y-auto overflow-x-hidden bg-ground">
      <div
        className={cn(
          'mx-auto grid w-full max-w-[1200px] gap-8 px-4 py-8',
          'lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)] lg:gap-10 lg:px-8 lg:py-12',
        )}
      >
        <LeftPane board={board} />
        <RightPane board={board} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Left — the outcome, its baseline, its caveat, and the way in
// ---------------------------------------------------------------------------

const PROOF_TONE = {
  text: 'text-text',
  violet: 'text-violet',
  good: 'text-good',
  bad: 'text-bad',
  faint: 'text-faint',
} as const;

function LeftPane({ board }: { board: BoardData }) {
  const { isDemo, recorded, recordedLoading, campaigns } = board;
  const nReports = useStore((s) => s.reports.length);
  const story = landingStory(campaigns, isDemo);

  return (
    <section className="min-w-0">
      <p className="mono-caps">
        multi-turn financial QA · the ConvFinQA benchmark
        {nReports > 0 ? ` · ${nReports} filings` : ''}
      </p>

      <h1 className="type-display mt-3" data-testid="landing-headline">
        {story.headline.before}
        {story.headline.emphasis && (
          <span className="text-amber">{story.headline.emphasis}</span>
        )}
        {story.headline.after}
      </h1>

      <p className="type-lede mt-4 max-w-[56ch]">{story.lede}</p>

      <dl data-testid="landing-proof" className="mt-5 flex flex-wrap gap-x-7 gap-y-3">
        {story.proof.map((p) => (
          <div key={p.key} className="min-w-0" data-proof={p.key}>
            <dd className={cn('type-hud', PROOF_TONE[p.tone])}>{p.value}</dd>
            <dt className="type-meta mt-1 max-w-[22ch]">{p.label}</dt>
          </div>
        ))}
      </dl>

      <div className="mt-6 flex flex-wrap items-center gap-2">
        <Link
          to="/chat"
          data-testid="landing-enter"
          className="type-body inline-flex items-center gap-1.5 rounded-md bg-amber px-4 py-2 font-medium text-amber-ink transition-[transform,box-shadow,opacity] duration-[var(--dur)] hover:-translate-y-px hover:opacity-95 hover:shadow-[var(--lift)]"
        >
          Open a conversation
          <ArrowRight className="size-4" aria-hidden />
        </Link>
        <Link
          to="/admin/system"
          className="type-body rounded-md border border-line-2 px-4 py-2 text-text transition-colors hover:border-amber-line hover:bg-panel-2"
        >
          How it was built
        </Link>
      </div>

      {story.caveat && (
        <p data-testid="landing-caveat" className="type-meta mt-3 max-w-[60ch] text-faint">
          {story.caveat}
        </p>
      )}

      <div className="mt-8 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="mono-caps">
          {isDemo ? 'replay a recorded conversation' : 'open a recorded conversation'}
        </span>
        <Link
          to="/chat"
          data-testid="landing-cta"
          className="type-meta whitespace-nowrap text-amber underline decoration-amber-line underline-offset-4 hover:decoration-amber"
        >
          or start from any filing →
        </Link>
      </div>

      <div className="mt-2">
        <RecordedConversations conversations={recorded} loading={recordedLoading} isDemo={isDemo} />
      </div>

      {board.error && (
        <p className="type-meta mt-5 rounded-md border border-dashed border-bad/50 bg-panel p-2.5 text-bad">
          A board read failed: {board.error}. The tiles show what did load; nothing has been
          substituted.
        </p>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Right — the record
// ---------------------------------------------------------------------------

export function RightPane({ board }: { board: BoardData }) {
  const { health, isDemo, metricsSource, campaigns, metrics, metricsLoading, traceCaptureEnabled } =
    board;

  // `served` even in the demo when the board is reading the recorded development
  // serving turns rather than this container's own replays — see `metricsSource`.
  const sourceWord = isDemo && metricsSource === 'demo' ? 'replayed' : 'served';
  const noMetrics = !metrics;
  const noTurns = Boolean(metrics && metrics.n_turns === 0);

  /**
   * Why a metrics tile is empty — never the same sentence for two reasons.
   *
   * These figures are all-time, over every turn the store holds, so an empty
   * tile means this source has never carried one. It used to say "in the last
   * 24 h", which was a window the numbers above it were never computed over and
   * which read as "quiet lately" when the truth was "never".
   */
  function metricsReason(what: string): string {
    if (noMetrics) return '/metrics/production returned nothing for this deployment';
    if (noTurns) return `no turns ${sourceWord} on this deployment yet`;
    return `${what} not yet measured — awaiting a metered eval run`;
  }

  const comparison = campaigns?.runtime_comparison ?? null;
  const stages = progression(
    campaigns?.champion_track,
    comparison,
    campaigns?.sdk_experiments ?? [],
    campaigns?.sdk_model_comparison,
  );
  const sdkArm = comparison?.agent_sdk ?? null;
  const gate = comparison?.gate ?? null;
  const story = landingStory(campaigns, isDemo);
  const judge = judgeSentence(campaigns);

  /**
   * How far the champion has moved across the campaign, on the gate split.
   * A campaign with no promotion yet has fewer than two track entries and this
   * is null — inventing a move from rejected challengers would report a change
   * that never shipped.
   */
  const campaignMove = (() => {
    const track = campaigns?.champion_track ?? [];
    if (track.length < 2) return null;
    const first = track[0];
    const last = track[track.length - 1];
    if (first.accuracy == null || last.accuracy == null) return null;
    return {
      from: first.version,
      delta: last.accuracy - first.accuracy,
      nPromoted: (campaigns?.experiments ?? []).filter((e) => e.promoted).length,
    };
  })();

  return (
    <section className="min-w-0">
      <LampStrip board={board} />

      {/* --- The star: the record as one figure ------------------------- */}
      <div
        data-testid="landing-star"
        className="mt-3 rounded-md border border-line bg-panel p-3.5 sm:p-4"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <span className="mono-caps">
            optimising the ConvFinQA agent · accuracy on the unseen gate split
          </span>
          <Link
            to="/admin/runtimes"
            className="type-meta text-amber underline decoration-amber-line underline-offset-4 hover:decoration-amber"
          >
            read the comparison →
          </Link>
        </div>
        <p className="type-small mt-1.5 text-muted">
          Every version scored on the same 349 held-back questions, in the order the work happened —
          from the raw four-agent pipeline to the single-session agent serving today.
        </p>
        <div className="mt-3">
          <ProgressionChart
            points={stages}
            pipelineBaseline={comparison?.pipeline?.accuracy ?? null}
            compact
          />
        </div>
        <ReadinessStrip className="mt-4 border-t border-line pt-3" />
      </div>

      {/* --- Four numbers, each with its baseline ----------------------- */}
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        <HudTile
          label="gate accuracy"
          value={formatPercent(campaigns?.champion_accuracy)}
          loading={!campaigns && board.loading}
          reason="no gate run recorded for the pipeline champion — run a cycle, then `convfinqa-evalloop story`"
          tone="plain"
          to="/admin/experiments"
          drill="/admin/experiments"
          meta={
            campaigns?.champion_accuracy != null && (
              <>
                four-agent pipeline ·{' '}
                <span className="type-num">{versionLabel(campaigns.champion)}</span>
                {campaignMove && (
                  <>
                    <br />
                    <span className={cn('type-num', campaignMove.delta >= 0 ? 'text-good' : 'text-bad')}>
                      {formatPointsDelta(campaignMove.delta)}
                    </span>{' '}
                    vs {versionLabel(campaignMove.from)} · {campaignMove.nPromoted} promotion
                    {campaignMove.nPromoted === 1 ? '' : 's'} in{' '}
                    {(campaigns.experiments ?? []).length} tries
                  </>
                )}
              </>
            )
          }
        />

        <HudTile
          label="runtime accuracy"
          value={formatPercent(sdkArm?.accuracy)}
          loading={!campaigns && board.loading}
          reason="the Claude Agent SDK arm has no run on the gate split yet"
          tone={gate?.promoted ? 'good' : 'plain'}
          to="/admin/runtimes"
          drill="/admin/runtimes"
          meta={
            sdkArm?.accuracy != null && (
              <>
                one Claude session ·{' '}
                <span className="type-num">{sdkArm.version ? versionLabel(sdkArm.version) : '—'}</span>
                {gate?.ci?.[0] != null && gate?.ci?.[1] != null && (
                  <>
                    <br />
                    95% CI{' '}
                    <span className="type-num">
                      {formatPointsDelta(gate.ci[0])} … {formatPointsDelta(gate.ci[1])}
                    </span>{' '}
                    vs {comparison?.pipeline?.version ? versionLabel(comparison.pipeline.version) : 'the pipeline'}, paired
                  </>
                )}
              </>
            )
          }
        />

        <HudTile
          label="cost per turn"
          value={formatUsd(metrics?.cost_usd.per_turn)}
          loading={metricsLoading}
          reason={metricsReason('token cost')}
          tone="amber"
          to="/admin/traces"
          drill="/admin/traces"
          meta={
            metrics && (
              <>
                all time · <span className="type-num">{formatUsd(metrics.cost_usd.total)}</span>{' '}
                over <span className="type-num">{metrics.cost_usd.n_measured}</span> turns priced
              </>
            )
          }
        />

        <HudTile
          label="p50 latency"
          value={formatLatency(metrics?.latency_ms.p50)}
          loading={metricsLoading}
          reason={metricsReason('latency')}
          tone="info"
          to="/admin/traces"
          drill="/admin/traces"
          meta={
            metrics && (
              <>
                p95 <span className="type-num">{formatLatency(metrics.latency_ms.p95)}</span> ·{' '}
                <span className="type-num">{metrics.latency_ms.n_measured}</span> turns measured
                {metrics.n_turns > 0 && (
                  <>
                    {' '}
                    · <span className="type-num">{metrics.n_turns}</span> {sourceWord}
                  </>
                )}
              </>
            )
          }
        />
      </div>

      <p data-testid="landing-holdout" className="type-meta mt-2.5 text-faint">
        {story.holdout}
      </p>

      {/* --- Tried last -------------------------------------------------- */}
      <Link
        to="/admin/runtimes"
        data-testid="landing-judge"
        data-significant={judge ? String(judge.significant) : 'none'}
        className="group mt-3 block rounded-md border border-line bg-panel p-3.5 transition-colors hover:border-amber-line hover:bg-panel-2"
      >
        <div className="flex items-baseline justify-between gap-2">
          <span className="mono-caps">tried last · llm-as-judge</span>
          <span className={cn('type-hud', judge?.significant ? 'text-good' : 'text-text')}>
            {judge ? judge.headline : NO_VALUE}
          </span>
        </div>
        <p className="type-small mt-2 text-muted">
          {judge
            ? judge.body
            : 'No confidence judge has been scored on the gate split yet.'}
          <span className="type-meta ml-1 text-amber group-hover:underline">read the measurement →</span>
        </p>
      </Link>

      <details className="mt-3 rounded-md border border-dashed border-line-2 bg-panel/60 p-3">
        <summary className="mono-caps cursor-pointer select-none">where these numbers come from</summary>
        <SourceNote board={board} />
      </details>

      {traceCaptureEnabled === false && (
        <p className="type-meta mt-3 text-faint">
          Trace capture is off on this deployment, so the cost and latency tiles will not move.
        </p>
      )}
      {health && health.runtime !== 'agent_sdk' && (
        <p className="type-meta mt-3 text-faint">
          This deployment still serves the four-agent pipeline; the runtime accuracy above is the
          challenger’s.
        </p>
      )}
    </section>
  );
}

/**
 * Where every number above came from. Accuracy is recomputed from committed
 * CSVs and is reproducible on any machine; the operational figures are
 * development traffic, and in the demo they are replays of recordings.
 * Neither has ever seen production load.
 */
function SourceNote({ board }: { board: BoardData }) {
  const { isDemo, metricsSource, metricsGeneratedAt, health } = board;

  return (
    <div className="mt-2 space-y-2">
      <p className="type-small text-muted">
        <span className="text-text">accuracy</span> — both arms on the fixed gate split of the
        committed split manifest, read from <span className="type-num">evaluation/story.json</span>,
        which the optimisation loop builds from its own runs. No API calls, no cached figure to
        drift. The legacy 770-question corpus is a different population under a retired protocol
        and lives on <span className="type-num">/admin/evaluations</span>, labelled as such.
      </p>
      <p className="type-small text-muted">
        <span className="text-text">latency and cost</span> —{' '}
        {isDemo && metricsSource === 'demo'
          ? 'turns this deployment replayed from conversations recorded in development. Replay timing is not production latency, so recorded and served turns are counted in separate source groups and never summed.'
          : isDemo
            ? 'the development turns these conversations were recorded from, shipped with the image and read all-time. Not this deployment’s own replays — those carry no timing at all, and the two are separate source groups that are never summed.'
            : 'every turn this development process has served, all time. Development traffic on one machine — not production load, and not a benchmark.'}{' '}
        Source group “<span className="type-num">{metricsSource}</span>”
        {metricsGeneratedAt && (
          <>
            {' '}
            · read <span className="type-num">{new Date(metricsGeneratedAt).toLocaleString()}</span>
          </>
        )}
        .
      </p>
      {health && (
        <p className="type-small break-words text-muted">
          <span className="text-text">bundle</span> — every figure is attributable to{' '}
          <span className="type-num">{health.bundle_id}</span>: prompts{' '}
          <span className="type-num">{health.bundle.prompts_version}</span>, models{' '}
          <span className="type-num">{health.bundle.lm_mini}</span> /{' '}
          <span className="type-num">{health.bundle.lm_max}</span>, dataset{' '}
          <span className="type-num">{health.bundle.dataset_hash}</span>, code{' '}
          <span className="type-num">{health.bundle.code_sha}</span>.
        </p>
      )}
    </div>
  );
}

export default LandingRoute;
