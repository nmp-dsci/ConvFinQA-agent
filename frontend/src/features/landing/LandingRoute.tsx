import { Link } from 'react-router-dom';
import { Figure, Status } from '@/kit/ui';
import { HudTile } from './HudTile';
import { LampStrip } from './LampStrip';
import { RecordedConversations } from './RecordedConversations';
import { ProgressionChart } from '../admin/ProgressionChart';
import { versionLabel } from '../admin/lib';
import { progression, runtimeVerdict } from '../admin/runtimeStory';
import { NO_VALUE, formatLatency, formatPercent, formatPointsDelta, formatUsd } from './format';
import { judgeSentence, landingStory } from './landingStory';
import type { ProofStat } from './landingStory';
import { useBoardData } from './useBoardData';
import type { BoardData } from './useBoardData';
import { useStore } from '../../store';
import { ReadinessStrip } from '../readiness/Readiness';
import { shippedCount, useReadiness } from '../readiness/api';
import './landing.css';

/**
 * The front door at `/`, as the house Overview: one slide flow down a single column.
 *
 * Eyebrow → the outcome as a claim with its baseline and its caveat → the lead → the one way in →
 * four tiles with a baseline on every number → the lamps → then numbered sections, each headed by a
 * claim computed from the data it shows: the progression, the readiness scorecard, the recorded
 * conversations, and the judge tried last.
 *
 * Three rules this file must not break:
 *
 *  1. **No number is written here.** Every figure comes from a query in `useBoardData`; every
 *     sentence comes from `landingStory`, a pure function with a test per branch. If a read fails
 *     or has nothing, the tile says so.
 *  2. **`null` is not `0`.** `/metrics/production` returns `null` with `n_measured: 0` for latency
 *     and cost until someone pays for a metered eval run. An empty tile prints an em dash and the
 *     reason; it never prints a zero and it never draws a flat line.
 *  3. **Every figure is the gate split, and the page says so.** The holdout has never been opened;
 *     the sentence under the tiles states it, rather than an empty tile implying it.
 */
export function LandingRoute() {
  const board = useBoardData(3);
  return <LandingBoard board={board} />;
}

/** The whole page for a board read — split from the route so a test can render it from a fixture. */
export function LandingBoard({ board }: { board: BoardData }) {
  return (
    <div data-testid="landing-board" className="landing">
      <Opening board={board} />
      <TheRecord board={board} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// The opening slide — the outcome, its baseline, its caveat, and the way in
// ---------------------------------------------------------------------------

/** A proof figure's colour: passed/failed only where the figure is a verdict; the baseline is ink. */
const PROOF_TONE: Record<ProofStat['tone'], string> = {
  text: '',
  violet: 'base',
  good: 'ok',
  bad: 'fail',
  faint: 'absent',
};

function Opening({ board }: { board: BoardData }) {
  const { isDemo, campaigns } = board;
  const nReports = useStore((s) => s.reports.length);
  const story = landingStory(campaigns, isDemo);

  return (
    <header className="page-head">
      <p className="label">
        multi-turn financial QA · the ConvFinQA benchmark
        {nReports > 0 ? ` · ${nReports} filings` : ''}
      </p>

      <h1 data-testid="landing-headline">
        {story.headline.before}
        {story.headline.emphasis && <em>{story.headline.emphasis}</em>}
        {story.headline.after}
      </h1>

      {story.caveat && (
        <p data-testid="landing-caveat" className="caveat">
          {story.caveat}
        </p>
      )}

      <p className="lead">{story.lede}</p>

      <dl data-testid="landing-proof" className="proof">
        {story.proof.map((p) => (
          <div key={p.key} data-proof={p.key}>
            <dd className={PROOF_TONE[p.tone]}>{p.value}</dd>
            <dt>{p.label}</dt>
          </div>
        ))}
      </dl>

      <div className="row cta">
        <Link to="/agent/chat" data-testid="landing-enter" className="btn">
          Ask a filing →
        </Link>
        <Link to="/agent">How it was built</Link>
      </div>

      {board.error && (
        <p className="callout err">
          A board read failed: {board.error}. The tiles show what did load; nothing has been
          substituted.
        </p>
      )}
    </header>
  );
}

// ---------------------------------------------------------------------------
// The record — tiles, lamps, then the numbered claims
// ---------------------------------------------------------------------------

function TheRecord({ board }: { board: BoardData }) {
  const {
    health,
    isDemo,
    metricsSource,
    campaigns,
    metrics,
    metricsLoading,
    traceCaptureEnabled,
    recorded,
    recordedLoading,
  } = board;

  // `served` even in the demo when the board is reading the recorded development
  // serving turns rather than this container's own replays — see `metricsSource`.
  const sourceWord = isDemo && metricsSource === 'demo' ? 'replayed' : 'served';
  const noMetrics = !metrics;
  const noTurns = Boolean(metrics && metrics.n_turns === 0);

  /**
   * Why a metrics tile is empty — never the same sentence for two reasons.
   *
   * These figures are all-time, over every turn the store holds, so an empty
   * tile means this source has never carried one.
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
  const gate = comparison?.gate ?? null;
  const story = landingStory(campaigns, isDemo);
  const judge = judgeSentence(campaigns);
  const { gateQuestions } = runtimeVerdict(comparison, campaigns?.split);

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
    <>
      {/* --- Four numbers, each with its baseline ----------------------- */}
      <div className="kpis">
        <HudTile
          label="champion accuracy"
          value={formatPercent(campaigns?.champion_accuracy)}
          loading={!campaigns && board.loading}
          reason="no gate run recorded for the champion — run it on the gate split, then `convfinqa-evalloop story`"
          tone={gate?.promoted ? 'good' : 'plain'}
          to="/optimise/runtimes"
          drill="optimise · runtimes"
          meta={
            campaigns?.champion_accuracy != null && (
              <>
                champion · <span className="num">{versionLabel(campaigns.champion)}</span>
                {campaigns.champion_agent && (
                  <>
                    {' '}
                    · <span className="mono">{campaigns.champion_agent.model}</span> · effort{' '}
                    {campaigns.champion_agent.effort}
                  </>
                )}
                {gate?.ci?.[0] != null && gate?.ci?.[1] != null && (
                  <>
                    <br />
                    95% CI{' '}
                    <span className="num">
                      {formatPointsDelta(gate.ci[0])} … {formatPointsDelta(gate.ci[1])}
                    </span>{' '}
                    vs{' '}
                    {comparison?.pipeline?.version
                      ? versionLabel(comparison.pipeline.version)
                      : 'the pipeline'}
                    , paired
                  </>
                )}
              </>
            )
          }
        />

        <HudTile
          label="pipeline accuracy"
          value={formatPercent(campaigns?.pipeline_champion_accuracy)}
          loading={!campaigns && board.loading}
          reason="no gate run recorded for the pipeline champion — run a cycle, then `convfinqa-evalloop story`"
          tone="plain"
          to="/optimise"
          drill="optimise · the campaigns"
          meta={
            campaigns?.pipeline_champion_accuracy != null && (
              <>
                four-agent pipeline · <span className="num">{versionLabel(campaigns.pipeline_champion)}</span>
                {campaignMove && (
                  <>
                    <br />
                    <span className={`num ${campaignMove.delta >= 0 ? 'v-ok' : 'v-fail'}`}>
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
          label="cost per turn"
          value={formatUsd(metrics?.cost_usd.per_turn)}
          loading={metricsLoading}
          reason={metricsReason('token cost')}
          tone="amber"
          to="/runs/traces"
          drill="runs · traces"
          meta={
            metrics && (
              <>
                all time · <span className="num">{formatUsd(metrics.cost_usd.total)}</span> over{' '}
                <span className="num">{metrics.cost_usd.n_measured}</span> turns priced
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
          to="/runs/traces"
          drill="runs · traces"
          meta={
            metrics && (
              <>
                p95 <span className="num">{formatLatency(metrics.latency_ms.p95)}</span> ·{' '}
                <span className="num">{metrics.latency_ms.n_measured}</span> turns measured
                {metrics.n_turns > 0 && (
                  <>
                    {' '}
                    · <span className="num">{metrics.n_turns}</span> {sourceWord}
                  </>
                )}
              </>
            )
          }
        />
      </div>

      <p data-testid="landing-holdout" className="holdout">
        {story.holdout}
      </p>

      <LampStrip board={board} />

      {/* --- 01 · the star: the record as one figure -------------------- */}
      <section className="doc-section" data-testid="landing-star">
        <p className="label">01 · progression</p>
        <h2>{progressionClaim(stages, gateQuestions)}</h2>
        <Figure
          title="accuracy on the unseen gate split, in the order the work happened"
          caption={
            <>
              Every version scored on the same {gateQuestions ?? 'fixed'} held-back questions, from
              the raw four-agent pipeline to the single-session agent serving today.{' '}
              <Link to="/optimise/runtimes">Read the comparison →</Link>
            </>
          }
          source={<code>evaluation/story.json</code>}
        >
          <ProgressionChart
            points={stages}
            pipelineBaseline={comparison?.pipeline?.accuracy ?? null}
            compact
          />
        </Figure>
      </section>

      {/* --- 02 · readiness ---------------------------------------------- */}
      <section className="doc-section">
        <p className="label">02 · readiness</p>
        <h2>
          <ReadinessClaim />
        </h2>
        <div className="body">
          <ReadinessStrip />
        </div>
      </section>

      {/* --- 03 · recorded conversations --------------------------------- */}
      <section className="doc-section">
        <p className="label">03 · recorded conversations</p>
        <h2>{recordedClaim(recorded, isDemo)}</h2>
        <p>
          {isDemo
            ? 'Each row replays a whole conversation recorded in development, through the same events a live turn emits.'
            : 'Each row opens a whole conversation on its filing, with the gold answer beside every turn.'}{' '}
          <Link to="/agent/chat" data-testid="landing-cta">
            Or start from any filing →
          </Link>
        </p>
        <RecordedConversations conversations={recorded} loading={recordedLoading} isDemo={isDemo} />
      </section>

      {/* --- Tried last -------------------------------------------------- */}
      <section className="doc-section">
        <p className="label">04 · tried last</p>
        <h2>
          {judge
            ? judge.significant
              ? 'An LLM judge separates right answers from wrong ones on the gate split'
              : 'An LLM judge was measured and did not earn the right to gate'
            : 'The confidence judge has not been measured yet'}
        </h2>
        <Link
          to="/scoring"
          data-testid="landing-judge"
          data-significant={judge ? String(judge.significant) : 'none'}
          className="card link judge"
        >
          <div className="judge-head">
            <span className="label">tried last · llm-as-judge</span>
            <span className="verdict">
              {judge ? (
                <Status tone={judge.significant ? 'ok' : 'no'}>{judge.headline}</Status>
              ) : (
                NO_VALUE
              )}
            </span>
          </div>
          <p>
            {judge ? judge.body : 'No confidence judge has been scored on the gate split yet.'}{' '}
            <span className="go">read the measurement →</span>
          </p>
        </Link>
      </section>

      <details className="sources">
        <summary>Where these numbers come from</summary>
        <SourceNote board={board} />
      </details>

      <div className="notes">
        {traceCaptureEnabled === false && (
          <p>Trace capture is off on this deployment, so the cost and latency tiles will not move.</p>
        )}
        {health && health.runtime !== 'agent_sdk' && (
          <p>
            This deployment still serves the four-agent pipeline; the runtime accuracy above is the
            challenger’s.
          </p>
        )}
      </div>
    </>
  );
}

/** Section 01's claim: the move from the first scored stage to the last, on its denominator. */
function progressionClaim(
  stages: ReturnType<typeof progression>,
  gateQuestions: number | null,
): string {
  const scored = stages.filter((s) => s.present && s.accuracy != null);
  const on = gateQuestions != null ? ` on the same ${gateQuestions} questions` : '';
  if (scored.length < 2) {
    return scored.length === 1
      ? `One version scored on the gate split so far, at ${formatPercent(scored[0].accuracy)}${on}`
      : 'No version has been scored on the gate split yet';
  }
  const first = scored[0];
  const last = scored[scored.length - 1];
  const best = scored.reduce((a, b) => ((b.accuracy ?? 0) > (a.accuracy ?? 0) ? b : a));
  return `From ${formatPercent(first.accuracy)} to ${formatPercent(best.accuracy)}${on}, across ${scored.length} scored stages${
    best !== last ? `; the last, ${last.stage}, reads ${formatPercent(last.accuracy)}` : ''
  }`;
}

/** Section 02's claim, from the same committed scorecard the strip renders. */
function ReadinessClaim() {
  const data = useReadiness().data;
  if (!data) return <>Production readiness, scored against the portfolio’s nine dimensions</>;
  const { shipped, total } = shippedCount(data.rows);
  return (
    <>
      {shipped} of {total} production-readiness dimensions shipped, each linked to its proof
    </>
  );
}

/** Section 03's claim: how the recorded run did on the conversations it offers. */
function recordedClaim(
  conversations: BoardData['recorded'],
  isDemo: boolean,
): string {
  if (!conversations || conversations.length === 0) {
    return 'Open any conversation and watch every stage of the answer';
  }
  const scored = conversations.filter((c) => c.nCorrect !== null);
  const n = conversations.length;
  if (scored.length === 0) {
    return `${n} conversation${n === 1 ? '' : 's'} to open, each a door into the chat`;
  }
  const right = scored.reduce((a, c) => a + (c.nCorrect ?? 0), 0);
  const turns = scored.reduce((a, c) => a + c.nTurns, 0);
  return `${n} conversation${n === 1 ? '' : 's'} ${isDemo ? 'to replay' : 'to open'}: ${right} of ${turns} turns right in the recorded run`;
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
    <>
      <p>
        <b>accuracy</b> — both arms on the fixed gate split of the committed split manifest, read
        from <code>evaluation/story.json</code>, which the optimisation loop builds from its own
        runs. No API calls, no cached figure to drift. The legacy 770-question corpus is a different
        population under a retired protocol and lives on <code>/admin/evaluations</code>, labelled
        as such.
      </p>
      <p>
        <b>latency and cost</b> —{' '}
        {isDemo && metricsSource === 'demo'
          ? 'turns this deployment replayed from conversations recorded in development. Replay timing is not production latency, so recorded and served turns are counted in separate source groups and never summed.'
          : isDemo
            ? 'the development turns these conversations were recorded from, shipped with the image and read all-time. Not this deployment’s own replays — those carry no timing at all, and the two are separate source groups that are never summed.'
            : 'every turn this development process has served, all time. Development traffic on one machine — not production load, and not a benchmark.'}{' '}
        Source group “<code>{metricsSource}</code>”
        {metricsGeneratedAt && (
          <>
            {' '}
            · read <span className="num">{new Date(metricsGeneratedAt).toLocaleString()}</span>
          </>
        )}
        .
      </p>
      {health && (
        <p className="wrap-any">
          <b>bundle</b> — every figure is attributable to <code>{health.bundle_id}</code>: prompts{' '}
          <code>{health.bundle.prompts_version}</code>, models <code>{health.bundle.lm_mini}</code> /{' '}
          <code>{health.bundle.lm_max}</code>, dataset <code>{health.bundle.dataset_hash}</code>, code{' '}
          <code>{health.bundle.code_sha}</code>.
        </p>
      )}
    </>
  );
}

export default LandingRoute;
