import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { HudTile } from './HudTile';
import { LampStrip } from './LampStrip';
import { PipelineStrip } from './PipelineStrip';
import { RecordedConversations } from './RecordedConversations';
import {
  NO_VALUE,
  formatCount,
  formatLatency,
  formatPercent,
  formatPointsDelta,
  formatUsd,
} from './format';
import { useBoardData } from './useBoardData';
import type { BoardData } from './useBoardData';

/**
 * The status board — the public front door at `/`.
 *
 * Two panes, and the split is the argument. On the left, what the system does
 * and three doors straight into it. On the right, whether it works, how fast,
 * what it costs — with the source of every figure printed underneath, because
 * the whole point of this project is that a number you cannot trace is not
 * evidence.
 *
 * Three rules this file must not break:
 *
 *  1. **No number is written here.** Every figure comes from a query in
 *     `useBoardData`. If a read fails or has nothing, the tile says so.
 *  2. **`null` is not `0`.** `/metrics/production` returns `null` with
 *     `n_measured: 0` for latency, tokens and cost until someone pays for a
 *     metered eval run. Half these tiles are legitimately empty today. An
 *     empty tile prints an em dash and the reason; it never prints a zero and
 *     it never draws a flat line.
 *  3. **Gate accuracy and out-of-sample accuracy are two tiles.** The gate
 *     split is the loop's own evidence and is what the campaign gates on; it
 *     is not out-of-sample, because every challenger of the campaign has been
 *     measured against it. The second tile is empty on purpose and says why:
 *     the holdout is unallocated during a campaign and has never been opened.
 *     Filling it with the gate figure — or with the legacy corpus's holdout
 *     number, which belongs to a rolled-back version — would be the board
 *     lying about generalisation, which is the one thing it exists not to do.
 */
export function LandingRoute() {
  const board = useBoardData(3);

  return (
    <div data-testid="landing-board" className="h-full overflow-y-auto overflow-x-hidden bg-ground">
      <div
        className={cn(
          'mx-auto grid w-full max-w-[1200px] gap-7 px-4 py-8',
          'lg:grid-cols-[minmax(0,1.02fr)_minmax(0,1fr)] lg:gap-9 lg:px-8 lg:py-12',
        )}
      >
        <LeftPane board={board} />
        <RightPane board={board} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Left — the system, in one line, and three ways in
// ---------------------------------------------------------------------------

function LeftPane({ board }: { board: BoardData }) {
  const { isDemo, recorded, recordedLoading } = board;

  return (
    <section className="min-w-0">
      <p className="mono-caps">
        Multi-turn financial QA · four-agent pipeline · single-session challenger measured
      </p>

      <h1 className="type-display mt-3">
        A system that answers <span className="text-amber">dependent</span> questions about SEC
        filings — and shows its work.
      </h1>

      <p className="type-lede mt-4 max-w-[54ch]">
        {isDemo
          ? 'Step into a recorded conversation. Every turn replays the real stage events, tool calls and timings captured in development — this deployment holds no API key and makes no model calls.'
          : 'Step into a conversation. Every turn runs the four agents live and streams each stage — triage, preprocess, retriever, calculator — as it resolves, with the gold answer beside its own.'}
      </p>

      <div className="mt-7 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
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
        <RecordedConversations
          conversations={recorded}
          loading={recordedLoading}
          isDemo={isDemo}
        />
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-2">
        <Link
          to="/chat"
          data-testid="landing-enter"
          className="rounded-md bg-amber px-3.5 py-2 type-body font-medium text-amber-ink transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber"
        >
          Open the chat
        </Link>
        <Link
          to="/admin"
          className="rounded-md border border-line-2 bg-panel px-3.5 py-2 type-body text-text transition-colors hover:border-amber-line hover:bg-panel-2"
        >
          Admin portal
        </Link>
        <Link
          to="/admin/system"
          className="rounded-md border border-line-2 bg-panel px-3.5 py-2 type-body text-text transition-colors hover:border-amber-line hover:bg-panel-2"
        >
          System
        </Link>
      </div>

      {board.error && (
        <p className="type-meta mt-5 rounded-md border border-dashed border-bad/50 bg-panel p-2.5 text-bad">
          A board read failed: {board.error}. The tiles below show what did load; nothing has been
          substituted.
        </p>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Right — does it work, how fast, how much
// ---------------------------------------------------------------------------

function RightPane({ board }: { board: BoardData }) {
  const {
    health,
    isDemo,
    campaigns,
    metrics,
    metricsSource,
    metricsLoading,
    metricsWindowHours,
    traceCaptureEnabled,
  } = board;

  const series = metrics?.series ?? [];
  // A bucket with no turns has no cost and no errors *to measure* — plotting a
  // zero there would draw the same line as a measured zero. Nulls keep the
  // hole visible.
  const measured = <T,>(pick: (s: (typeof series)[number]) => T): Array<T | null> =>
    series.map((s) => (s.n_turns > 0 ? pick(s) : null));

  const window = metricsWindowHours ?? 24;
  const sourceWord = isDemo ? 'replayed' : 'served';

  const noMetrics = !metrics;
  const noTurns = Boolean(metrics && metrics.n_turns === 0);

  /** Why a metrics tile is empty — never the same sentence for two reasons. */
  function metricsReason(what: string): string {
    if (noMetrics) return '/metrics/production returned nothing for this deployment';
    if (noTurns) return `no turns ${sourceWord} in the last ${window} h`;
    return `${what} not yet measured — awaiting a metered eval run`;
  }

  /**
   * How far the champion has moved across the campaign, on the gate split.
   *
   * The track holds only versions a promotion actually moved the champion to,
   * so its first entry is the campaign's starting champion and its last is the
   * current one. A campaign with no promotion yet has fewer than two entries
   * and this is null — there is no move to report, and inventing one from the
   * rejected challengers would be reporting a change that never shipped.
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

  // The runtime decision, read from the same story the Runtimes page renders:
  // the SDK arm on the model it was gated with, and the same prompt on the
  // second model. Absent until those runs exist — never a zero.
  const sdkArm = campaigns?.runtime_comparison?.agent_sdk ?? null;
  const sdkGate = campaigns?.runtime_comparison?.gate ?? null;
  const swap = campaigns?.sdk_model_comparison ?? null;
  const swapArm = swap?.models?.find((m) => m.model !== swap.reference_model) ?? null;
  const swapPair = swap?.pairs?.[0] ?? null;
  const shortModel = (model: string | null | undefined) =>
    (model ?? '')
      .replace(/^claude-/, '')
      .replace(/-\d{8}$/, '');

  return (
    <section className="min-w-0">
      <p className="mono-caps break-words">
        {health
          ? `champion ${health.champion ?? '—'} · bundle ${health.bundle_id} · ${health.bundle.lm_mini} · code ${health.bundle.code_sha}`
          : 'reading deployment…'}
      </p>

      <div className="mt-3">
        <LampStrip board={board} />
      </div>

      <div className="mt-4 grid gap-2 sm:grid-cols-2">
        {/* --- Does it work ------------------------------------------------ */}
        <HudTile
          label="gate accuracy"
          value={formatPercent(campaigns?.champion_accuracy)}
          loading={!campaigns && board.loading}
          reason="no gate run recorded for the champion — run a cycle, then `convfinqa-evalloop story`"
          tone="plain"
          to="/admin/campaigns"
          drill="/admin/campaigns"
          meta={
            campaigns?.champion_accuracy != null && (
              <>
                <span className="type-num">{campaigns.champion}</span> ·{' '}
                <span className="type-num">{String(campaigns.split?.gate_questions ?? '—')}</span>{' '}
                questions across{' '}
                <span className="type-num">{String(campaigns.split?.gate_reports ?? '—')}</span>{' '}
                conversations, fixed for the campaign
                {campaignMove && (
                  <>
                    <br />
                    <span className={cn('type-num', campaignMove.delta >= 0 ? 'text-good' : 'text-bad')}>
                      {formatPointsDelta(campaignMove.delta)}
                    </span>{' '}
                    vs {campaignMove.from} over {campaignMove.nPromoted} promotion
                    {campaignMove.nPromoted === 1 ? '' : 's'}
                  </>
                )}
              </>
            )
          }
        />

        <HudTile
          label="out-of-sample accuracy"
          value="—"
          reason="the holdout is unallocated during a campaign and has never been opened"
          tone="good"
          to="/admin/campaigns"
          drill="/admin/campaigns"
          meta={
            <>
              no confirmatory run yet ·{' '}
              <span className="type-num">
                {campaigns?.experiments?.length ?? 0}
              </span>{' '}
              challenger{(campaigns?.experiments?.length ?? 0) === 1 ? '' : 's'} have now been
              measured against the gate split, so part of the figure beside this is selection
            </>
          }
        />

        {/* --- The runtime decision ---------------------------------------- */}
        <HudTile
          label="single-session runtime"
          value={formatPercent(sdkArm?.accuracy)}
          loading={!campaigns && board.loading}
          reason="the Claude Agent SDK arm has no run on the gate split yet"
          tone={sdkGate?.promoted ? 'good' : 'plain'}
          to="/admin/runtimes"
          drill="/admin/runtimes"
          meta={
            sdkArm?.accuracy != null && (
              <>
                <span className="type-num">{sdkArm.version ?? '—'}</span> on{' '}
                <span className="type-num">{shortModel(sdkArm.model) || 'claude'}</span> · same{' '}
                {String(campaigns?.split?.gate_questions ?? '—')} questions
                {sdkGate?.delta_pp != null && (
                  <>
                    <br />
                    <span className={cn('type-num', sdkGate.delta_pp >= 0 ? 'text-good' : 'text-bad')}>
                      {formatPointsDelta(sdkGate.delta_pp / 100)}
                    </span>{' '}
                    vs {campaigns?.champion ?? 'the champion'}, paired ·{' '}
                    {sdkGate.promoted ? 'recommended runtime' : 'not promoted'}
                  </>
                )}
              </>
            )
          }
        />

        <HudTile
          label="same prompt, smaller model"
          value={formatPercent(swapArm?.accuracy)}
          loading={!campaigns && board.loading}
          reason="the sdk champion has been scored on one model only"
          tone="plain"
          to="/admin/runtimes"
          drill="/admin/runtimes"
          meta={
            swapArm?.accuracy != null && (
              <>
                <span className="type-num">{shortModel(swapArm.model)}</span> · scoring pass, no
                optimisation
                {swapPair?.delta_pp != null && (
                  <>
                    <br />
                    <span className={cn('type-num', swapPair.delta_pp >= 0 ? 'text-good' : 'text-bad')}>
                      {formatPointsDelta(swapPair.delta_pp / 100)}
                    </span>{' '}
                    vs {shortModel(swap?.reference_model)} · cost{' '}
                    <span className="type-num">{formatUsd(swapArm.cost)}</span> vs{' '}
                    <span className="type-num">{formatUsd(sdkArm?.cost)}</span> per pass
                  </>
                )}
              </>
            )
          }
        />

        {/* --- How fast, how much ------------------------------------------ */}
        <HudTile
          label="p50 latency"
          value={formatLatency(metrics?.latency_ms.p50)}
          loading={metricsLoading}
          reason={metricsReason('latency')}
          tone="info"
          series={series.map((s) => s.p50_latency_ms)}
          to="/admin/traces"
          drill="/admin/traces"
          meta={
            metrics && (
              <>
                p95 <span className="type-num">{formatLatency(metrics.latency_ms.p95)}</span> ·{' '}
                <span className="type-num">{metrics.latency_ms.n_measured}</span> turns measured
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
          series={measured((s) => s.cost_usd)}
          to="/admin/traces"
          drill="/admin/traces"
          meta={
            metrics && (
              <>
                <span className="type-num">{formatUsd(metrics.cost_usd.total)}</span> total ·{' '}
                <span className="type-num">{metrics.cost_usd.n_measured}</span> turns priced
              </>
            )
          }
        />

        <HudTile
          label="turns served"
          value={metrics ? formatCount(metrics.n_turns) : NO_VALUE}
          loading={metricsLoading}
          reason={metricsReason('turn count')}
          tone="plain"
          series={series.map((s) => s.n_turns)}
          to="/admin/traces"
          drill="/admin/traces"
          meta={
            metrics && (
              <>
                last <span className="type-num">{window}</span> h · source group “
                <span className="type-num">{metricsSource}</span>”
                {metrics.accuracy.n_scored > 0 && (
                  <>
                    <br />
                    <span className="type-num">
                      {formatPercent(metrics.accuracy.accuracy)}
                    </span>{' '}
                    correct over <span className="type-num">{metrics.accuracy.n_scored}</span>{' '}
                    scored
                  </>
                )}
              </>
            )
          }
        />

        <HudTile
          label="error rate"
          value={formatPercent(metrics?.errors.error_rate)}
          loading={metricsLoading}
          reason={metricsReason('error rate')}
          tone={metrics && metrics.errors.n_errors > 0 ? 'bad' : 'good'}
          series={measured((s) => s.n_errors)}
          to="/admin/traces"
          drill="/admin/traces"
          meta={
            metrics && (
              <>
                <span className="type-num">{metrics.errors.n_errors}</span> errors over{' '}
                <span className="type-num">{metrics.n_turns}</span> turns
              </>
            )
          }
        />
      </div>

      <SourceNote board={board} />

      <div className="mt-3">
        <PipelineStrip />
      </div>

      {traceCaptureEnabled === false && (
        <p className="type-meta mt-3 text-faint">
          Trace capture is off on this deployment, so the four operational tiles will not move.
        </p>
      )}
    </section>
  );
}

/**
 * Where every number above came from, in the reader's line of sight.
 *
 * This is the single most important paragraph on the page. Accuracy is
 * recomputed from committed CSVs and is reproducible on any machine; the
 * operational figures are development traffic, and in the demo they are
 * replays of recordings. Neither has ever seen production load. Saying so here
 * costs three lines and is the difference between a portfolio piece and a
 * claim that is not true.
 */
function SourceNote({ board }: { board: BoardData }) {
  const { isDemo, metricsSource, metricsGeneratedAt, metricsWindowHours, health } = board;
  const window = metricsWindowHours ?? 24;

  return (
    <div className="mt-3 rounded-md border border-dashed border-line-2 bg-panel/60 p-3">

      <div className="mono-caps mb-1.5">sources</div>
      <p className="type-meta">
        <span className="text-text">gate accuracy</span> — the champion's accuracy on the fixed
        gate split of the committed split manifest, read from{' '}
        <span className="type-num">evaluation/story.json</span>, which the optimisation loop builds
        from its own runs. No API calls, no cached figure to drift. It is deliberately not the
        legacy 770-question corpus figure: that is a different population under a retired scoring
        protocol, and it lives on{' '}
        <span className="type-num">/admin/evaluations</span> where it is labelled as such.
      </p>
      <p className="type-meta mt-1.5">
        <span className="text-text">latency, cost, turns, errors</span> —{' '}
        {isDemo
          ? `turns this deployment replayed from conversations recorded in development. Replay timing is not production latency: a turn recorded at 6.7 s and replayed in 2 s is a 6.7 s turn, so recorded and served turns are counted in separate source groups and never summed.`
          : `turns this development process has served in the last ${window} h. Development traffic on one machine — not production load, and not a benchmark.`}{' '}
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
        <p className="type-meta mt-1.5 break-words">
          <span className="text-text">bundle</span> — every figure above is attributable to{' '}
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
