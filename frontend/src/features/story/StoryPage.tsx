import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Figure } from '@/kit/ui';
import { getCampaigns } from '../admin/api';
import type { CampaignsResponse } from '../admin/api';
import { versionLabel } from '../admin/lib';
import { AdminPage, ErrorNote, LoadingRows, Panel } from '../admin/ui';
import { formatPercent } from '../landing/format';
import { PassLedger } from './PassLedger';
import { TimelineChart } from './TimelineChart';
import {
  NO_FILTERS,
  agentLabel,
  challengerPhrase,
  formatDay,
  formatP,
  formatPp,
  ledgerRows,
  stepHistory,
  storySummary,
  timelineModel,
} from './timeline';
import type { LedgerFilters, StepSegment, StorySummary, TimelineModel } from './timeline';

/**
 * Optimise › Story — the first page of the slot (s21, owner's D4): every agent ever scored on the
 * gate split, test passes only, against the champion of its day.
 *
 * The other Optimise pages each tell one chapter — the pipeline's campaigns, the runtime
 * comparison, the rule research — and until this page none of them told the whole of it: the
 * campaign track had two points, the runtime figure had five bars and no dates, and Haiku 4.5 was on
 * no chart at all. Every sentence here is computed from `story.json` (`test_timeline`,
 * `champion_steps`); nothing is counted by hand.
 */
export default function StoryPage() {
  const query = useQuery({
    // The key the Campaigns, Runtimes and Scoring pages use: one payload backs all of them.
    queryKey: ['eval-campaigns'],
    queryFn: () => getCampaigns(),
    staleTime: 60_000,
  });
  return <StoryView data={query.data} loading={query.isLoading} error={query.error} />;
}

/** The page for a payload — split from the route so a test can render it from a fixture. */
export function StoryView({
  data,
  loading = false,
  error = null,
}: {
  data: CampaignsResponse | undefined;
  loading?: boolean;
  error?: unknown;
}) {
  const [filters, setFilters] = useState<LedgerFilters>(NO_FILTERS);
  const model = timelineModel(data?.test_timeline, data?.champion_steps);
  const summary = storySummary(model);
  const gateQuestions = typeof data?.split?.gate_questions === 'number' ? (data.split.gate_questions as number) : null;

  return (
    <AdminPage
      testId="optimise-story"
      eyebrow="optimise · the story"
      title={<StoryClaim summary={summary} />}
      sub={storyLead(summary, gateQuestions)}
    >
      {error ? <ErrorNote error={error} /> : null}
      {loading && !data ? (
        <LoadingRows rows={4} />
      ) : (
        <>
          <section className="opt-sec">
            <Figure
              testId="story-figure"
              title="test accuracy of every complete pass, in the order it was scored"
              caption={
                model.line.length ? (
                  <>The champion line, step by step: {stepHistory(model.line)}.</>
                ) : (
                  <>No champion step is on record for these passes.</>
                )
              }
              source={
                <>
                  <code>evaluation/story.json</code> · <code>test_timeline</code>, <code>champion_steps</code>
                </>
              }
            >
              <TimelineChart passes={data?.test_timeline} steps={data?.champion_steps} gateQuestions={gateQuestions} />
            </Figure>
          </section>

          <Panel
            testId="story-moves"
            title={movesClaim(model)}
            endpoint="evaluation/registry.json · history"
            note="Read off the registry’s history and placed at the pass that measured each champion — before its promotion was written."
          >
            <ChampionMoves model={model} />
          </Panel>

          <Panel
            testId="story-challengers"
            title={challengersClaim(summary)}
            endpoint="story.json · challengers"
            note={
              <>
                A challenger on the champion’s own model is judged by the house gate (net positive and
                one-sided cluster p &lt; 0.05); one on another model or effort by the s19 rule (beats,
                or matches within the 3pp margin, with train as a veto).{' '}
                <Link to="/optimise/runtimes">The model-swap verdicts in full →</Link>
              </>
            }
          >
            {summary.challengersSince.length ? (
              <PassLedger rows={ledgerRows(summary.challengersSince)} testId="challenger-ledger" />
            ) : (
              <p className="empty">— no pass has been scored since the champion’s own</p>
            )}
          </Panel>

          <Panel
            testId="story-ledger"
            title={`Every test pass, ${model.passes.length} of them, oldest first`}
            endpoint="/runs"
            to="/runs"
            note="The same rows as Runs › Every test pass; train passes are there, behind a toggle."
          >
            <PassLedger rows={ledgerRows(model.passes)} filters={filters} onFilters={setFilters} />
          </Panel>
        </>
      )}
    </AdminPage>
  );
}

/** The h1: who leads, since when, at what — and what has tried since. One `<em>`, on the number. */
export function StoryClaim({ summary }: { summary: StorySummary }) {
  const c = summary.champion;
  if (!c) {
    return (
      <>
        No champion has been <em>measured</em> on the gate split yet
      </>
    );
  }
  const k = summary.challengersSince.length;
  const tail = !k
    ? '; nothing has challenged it since'
    : `; ${k} challenger${k === 1 ? '' : 's'} since, ${
        summary.nBeat === 0 ? `none has beaten it` : `${summary.nBeat} beat it`
      }`;
  return (
    <>
      {agentLabel(c.step)} has led since {formatDay(summary.since)} at <em>{formatPercent(c.accuracy)}</em>
      {tail}
    </>
  );
}

function storyLead(summary: StorySummary, gateQuestions: number | null): string {
  if (!summary.nPasses) {
    return 'Every test pass on the gate split will be drawn here, oldest first, against the champion of its day. None is on record yet.';
  }
  return `${summary.nPasses} test passes on the ${gateQuestions ?? 'fixed'}-question gate split, oldest first, of ${summary.nVersions} prompt versions on ${summary.nModels} models. The line is the champion of its day and moves only on a promotion. Train passes write the versions and never promote them, so they sit on Runs behind a toggle, not on this axis.`;
}

function movesClaim(model: TimelineModel): string {
  const moves = Math.max(0, model.line.length - 1);
  if (!model.line.length) return 'No champion is on record for these passes';
  return `The champion moved ${moves} time${moves === 1 ? '' : 's'} in ${model.points.length} complete passes`;
}

function challengersClaim(summary: StorySummary): string {
  const list = summary.challengersSince;
  if (!summary.champion) return 'No champion, so no challenger';
  if (!list.length) return `Nothing has challenged ${versionLabel(summary.champion.step.version)} since ${formatDay(summary.since)}`;
  return `Since ${formatDay(summary.since)}: ${list.map(challengerPhrase).join(', ')}`;
}

function ChampionMoves({ model }: { model: TimelineModel }) {
  if (!model.line.length) return <p className="empty">— no champion step falls inside these passes</p>;
  return (
    <ol className="story-steps" data-testid="champion-moves">
      {model.line.map((s, k) => (
        <li key={`${s.step.version}-${s.from}`} data-version={s.step.version}>
          <span className="when">{formatDay(s.pass?.date ?? s.step.at?.slice(0, 10) ?? null)}</span>
          <span className="what">
            <b>{agentLabel(s.step)}</b> · {formatPercent(s.accuracy)}
            <span className="why">{moveReason(s, k)}</span>
            {s.restated.map((r) => (
              <span key={`${r.event}-${r.at}`} className="why">
                {r.event === 'universal_champion'
                  ? `Recorded as the one universal champion on ${formatDay(r.at?.slice(0, 10) ?? null)} — the same agent, named; the line does not move.`
                  : `${r.event} on ${formatDay(r.at?.slice(0, 10) ?? null)} restated the same agent.`}
              </span>
            ))}
          </span>
        </li>
      ))}
    </ol>
  );
}

function moveReason(s: StepSegment, k: number): string {
  const vs = s.pass?.vs;
  if (k === 0) return 'The champion in force when the gate split was first scored.';
  if (vs?.paired && vs.delta_pp != null) {
    return `${s.pass?.label ? `${s.pass.label}: ` : ''}${formatPp(vs.delta_pp)} over ${versionLabel(vs.version)} on the same questions, paired, one-sided p ${formatP(vs.p_value)}.`;
  }
  return s.step.reason || 'Promoted.';
}
