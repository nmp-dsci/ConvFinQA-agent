import { Link } from 'react-router-dom';
import { formatPercent, formatPointsDelta } from '../landing/format';
import { MiniBar } from './charts';
import {
  Callout,
  Field,
  Lamp,
  Live,
  Mono,
  Panel,
  PanelTitle,
  Prose,
  Provenance,
  ScrollX,
  Section,
  Table,
  Td,
  Th,
} from './ui';
import type { SystemData } from './useSystemData';

// ---------------------------------------------------------------------------
// 06 · Evaluation — the two-number rule
// ---------------------------------------------------------------------------

export function EvaluationSection({ data }: { data: SystemData }) {
  const { versions, versionHoldouts, championName } = data;

  const holdoutOf = (version: string) =>
    versionHoldouts?.find((v) => v.version === version) ?? null;

  return (
    <Section
      id="evaluation"
      index={6}
      eyebrow="evaluation"
      title="Two numbers, always, and never their average"
      lede={
        <>
          200 conversations were sampled from the public split and cut 60/40 on a fixed seed. The
          optimizer saw one side; it never saw the other. Every accuracy on this project is
          therefore two numbers — overall, over everything scored, and never-seen, over the side no
          optimizer touched — and they are reported side by side because only the second one
          supports a claim about generalisation.
        </>
      }
    >
      <div className="grid gap-3 sm:grid-cols-3">
        <Panel>
          <PanelTitle>optimizer_train</PanelTitle>
          <p className="type-hud text-info">120</p>
          <p className="type-meta mt-1">
            conversations GEPA and the s7 harness optimised against. Accuracy measured here says
            nothing about generalisation, because the prompts were tuned on it.
          </p>
        </Panel>
        <Panel>
          <PanelTitle>never_seen</PanelTitle>
          <p className="type-hud text-good">80</p>
          <p className="type-meta mt-1">
            309 questions no optimizer ever saw. The only subset that supports a generalisation
            claim, and the one the held-out figure is measured on.
          </p>
        </Panel>
        <Panel>
          <PanelTitle>sampled · the scored set</PanelTitle>
          <p className="type-hud">770</p>
          <p className="type-meta mt-1">
            questions across all 200 conversations. Its accuracy mixes seen and unseen, so it is
            reported as “overall” and never as “held out”.
          </p>
        </Panel>
      </div>
      <Provenance origin="code">
        <Mono>data/loader.py::optimizer_split()</Mono> — <Mono>random.Random(42)</Mono>, shuffle,
        cut at 60%. Membership is inspectable at <Mono>/eval/splits</Mono>, report id by report id,
        so the held-out claim is checkable rather than asserted.
      </Provenance>

      <Callout tone="warn" title="“Held out” means one specific split, and there is a near-miss">
        <p>
          The repository also has a <Mono>train_report_ids</Mono> helper — likewise 60/40, likewise
          seeded 42 — and the two agree on only 78 of the 120 conversations. GEPA ran against{' '}
          <Mono>optimizer_split()</Mono>, so that is the one that defines “never seen”. Reporting a
          holdout figure computed from the other split would be wrong by about a third of the set
          and would look completely plausible.
        </p>
      </Callout>

      <div>
        <PanelTitle>every version with a committed predictions CSV</PanelTitle>
        <ScrollX>
          <Table>
            <thead>
              <tr>
                <Th>version</Th>
                <Th numeric>overall exe</Th>
                <Th numeric>never-seen exe</Th>
                <Th numeric>program</Th>
                <Th>questions</Th>
                <Th>state</Th>
              </tr>
            </thead>
            <tbody>
              {(versions ?? []).map((v) => {
                const holdout = holdoutOf(v.version);
                const isChampion = v.version === championName;
                return (
                  <tr key={v.version}>
                    <Td>
                      <span className="type-num text-text">{v.version}</span>
                    </Td>
                    <Td numeric>
                      {formatPercent(v.exe_acc)}
                      <div className="mt-1">
                        <MiniBar pct={v.exe_acc * 100} />
                      </div>
                    </Td>
                    <Td numeric>
                      <span className={holdout ? 'text-good' : undefined}>
                        <Live
                          value={holdout ? formatPercent(holdout.holdout_accuracy) : null}
                          reason="/admin/experiments did not answer"
                        />
                      </span>
                      {holdout && (
                        <div className="type-meta mt-0.5 font-normal">
                          {holdout.holdout_n_questions} q
                        </div>
                      )}
                    </Td>
                    <Td numeric>
                      {formatPercent(v.prog_acc)}
                      <div className="type-meta mt-0.5 font-normal">
                        {v.n_program_correct}/{v.n_program_turns}
                      </div>
                    </Td>
                    <Td>{v.n_questions}</Td>
                    <Td>
                      <span className="inline-flex items-center gap-1.5">
                        <Lamp state={isChampion ? 'good' : 'idle'} />
                        {isChampion ? 'champion' : 'registered'}
                      </span>
                    </Td>
                  </tr>
                );
              })}
              {(versions ?? []).length === 0 && (
                <tr>
                  <Td className="text-faint">
                    <Live value={null} reason="/admin/versions did not answer" /> no versions —{' '}
                    <Mono>/admin/versions</Mono> did not answer. Start the backend to fill this
                    table; everything else on this page is readable without it.
                  </Td>
                  <Td />
                  <Td />
                  <Td />
                  <Td />
                  <Td />
                </tr>
              )}
            </tbody>
          </Table>
        </ScrollX>
        <Provenance origin="live">
          <Mono>/admin/versions</Mono> recomputes execution and program accuracy from the committed
          CSVs on each request; <Mono>/admin/experiments</Mono> supplies the never-seen split, which{' '}
          <Mono>/admin/versions</Mono> deliberately cannot compute.{' '}
          <Link
            to="/admin/evaluations"
            className="text-amber underline decoration-amber-line underline-offset-4"
          >
            Open the rows behind these figures →
          </Link>
        </Provenance>
      </div>

      <Callout title="Program accuracy is scored, and it is not an accuracy of the same kind">
        <p>
          Roughly 35% against roughly 77% execution. The pipeline answers a turn from the
          conversation’s prior answers — <Mono>divide(132, 111), multiply(#0, 100)</Mono> — where
          gold re-derives from raw table values —{' '}
          <Mono>subtract(243, 111), divide(#0, 111)</Mono>. Same answer, shorter program, counted as
          a mismatch. It is reported because hiding it would be worse, and it is explained here
          because a reader who sees only the number concludes the system is wrong two turns in
          three.
        </p>
      </Callout>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 07 · The promotion contract
// ---------------------------------------------------------------------------

export function PromotionSection({ data }: { data: SystemData }) {
  const { gate, gateCandidate, championName } = data;

  return (
    <Section
      id="promotion"
      index={7}
      eyebrow="the contract"
      title="A challenger is promoted only if it breaks nothing"
      lede={
        <>
          “Beats the champion” is not enough, because a change that fixes twelve number turns and
          breaks nine program turns beats the champion. Promotion requires both: accuracy at least
          as good, <em>and</em> not one question that used to pass and now fails.
        </>
      }
    >
      <div className="grid gap-3 md:grid-cols-3">
        <Panel>
          <PanelTitle>accuracy_ok</PanelTitle>
          <Prose className="type-small">
            The candidate’s accuracy over the rows both versions answered is at least the
            champion’s, to within a floating-point epsilon. Full-frame accuracy is computed and
            reported but never drives the decision — comparing two versions on different populations
            is not a comparison.
          </Prose>
        </Panel>
        <Panel>
          <PanelTitle>no_regressions</PanelTitle>
          <Prose className="type-small">
            Not a single pass→fail flip on any shared question. Every flip is listed by report id
            and question, so a refusal names the turns it is refusing over rather than a percentage.
          </Prose>
        </Panel>
        <Panel>
          <PanelTitle>promotable</PanelTitle>
          <Prose className="type-small">
            Both of the above, and a non-empty comparison. An empty join promotes nothing — a
            candidate with no overlapping rows is unmeasured, not perfect.
          </Prose>
        </Panel>
      </div>
      <Provenance origin="code">
        <Mono>tracking/comparator.py</Mono> joins the two predictions CSVs on{' '}
        <Mono>(report_id, turn_index)</Mono>. <Mono>convfinqa-mlflow promote</Mono> refuses unless
        it passes.
      </Provenance>

      <Panel>
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <span className="mono-caps">the comparator, run right now on this deployment</span>
          {gate && (
            <span className="type-meta inline-flex items-center gap-1.5">
              <Lamp state={gate.promotable ? 'good' : 'bad'} />
              {gate.promotable ? 'promotable' : 'refused'}
            </span>
          )}
        </div>

        {gate ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Field label="candidate" note={`against champion ${gate.baseline_version}`}>
              <span className="type-num">{gate.candidate_version}</span>
            </Field>
            <Field label="accuracy delta" note={`over ${gate.n_compared} shared questions`}>
              <span className={gate.accuracy_delta >= 0 ? 'text-good' : 'text-bad'}>
                {formatPointsDelta(gate.accuracy_delta)}
              </span>
            </Field>
            <Field label="pass → fail flips" note="any single one refuses the promotion">
              <span className={gate.regressions.length === 0 ? 'text-good' : 'text-bad'}>
                {gate.regressions.length}
              </span>
            </Field>
            <Field label="fail → pass flips" note="what the candidate did fix">
              <span className="text-muted">{gate.improvements.length}</span>
            </Field>
            <div className="sm:col-span-2 lg:col-span-4">
              <p className="type-small text-muted">{gate.reason}</p>
            </div>
          </div>
        ) : (
          <p className="type-small text-faint">
            <Live value={null} reason="/admin/compare did not answer" /> no live comparison —{' '}
            {gateCandidate && championName
              ? `${championName} vs ${gateCandidate} could not be read from /admin/compare.`
              : 'there is no registered version to compare against the champion on this deployment.'}
          </p>
        )}
        <Provenance origin="live">
          computed on request from the committed prediction CSVs — the same function the CLI and CI
          call, not a re-implementation.{' '}
          <Link
            to="/admin/experiments"
            className="text-amber underline decoration-amber-line underline-offset-4"
          >
            See the flips →
          </Link>
        </Provenance>
      </Panel>

      <Panel>
        <PanelTitle>the CI gate</PanelTitle>
        <ol className="type-small ml-4 list-decimal space-y-2 text-muted marker:text-faint">
          <li>
            <span className="text-text">Re-score every committed version.</span> Each row’s{' '}
            <Mono>correct</Mono> column is re-derived from its predicted and gold answers with the
            same numeric matcher the evaluator uses. Any disagreement fails the build — a committed
            CSV whose verdicts have drifted from the scorer is a silently wrong artefact, and every
            number on this site is computed from those CSVs.
          </li>
          <li>
            <span className="text-text">Hold the champion to its recorded floor.</span> Accuracy
            recomputed from the CSV must be within <Mono>0.005</Mono> of the figure the registry
            recorded when it was promoted.
          </li>
          <li>
            <span className="text-text">Report the challenger comparison.</span> If a challenger
            alias exists, the comparison and up to twenty pass→fail flips are printed — as
            information. This step does not fail the build; a human decides whether to promote.
          </li>
        </ol>
        <Provenance origin="code">
          <Mono>uv run python -m convfinqa.tracking.gate</Mono>, run by the{' '}
          <Mono>eval-gate</Mono> job in <Mono>.github/workflows/ci.yml</Mono>. Exit 1 on any
          failure.
        </Provenance>
      </Panel>

      <Callout tone="warn" title="The champion did not itself come through this gate">
        <p>
          The registry holds one promotion event: the current champion, registered by the backfill
          that rebuilt the history from committed artefacts, with actor <Mono>backfill</Mono> and no
          comparison attached — the first registered version becomes champion by default. The
          comparator and the gate are real, they run on demand, and the one challenger that exists
          fails them. But no version has yet been promoted <em>through</em> the contract, and this
          page is not going to imply otherwise.
        </p>
      </Callout>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 08 · GEPA and the s7 harness
// ---------------------------------------------------------------------------

export function OptimisationSection({ data }: { data: SystemData }) {
  return (
    <Section
      id="optimisation"
      index={8}
      eyebrow="optimisation"
      title="Two ways the prompts were written by something other than a person"
      lede={
        <>
          No exemplars are ever put in front of these agents. The instructions themselves are the
          thing being optimised — first by GEPA over the whole pipeline, then by a per-case harness
          that diagnoses one failure at a time and only keeps a fix that survives a re-run.
        </>
      }
    >
      <div className="grid gap-3 lg:grid-cols-2">
        <Panel>
          <PanelTitle>GEPA · reflective prompt evolution</PanelTitle>
          <Prose className="type-small">
            The DSPy optimizer proposes instruction rewrites, scores them against the 120
            optimizer-train conversations with a turn-level metric that feeds textual feedback back
            into the next proposal, and keeps what wins. Two modes: a smoke run capped at 120 metric
            calls over 5 validation conversations, which checks the wiring and is not a transferable
            optimization, and a real run over 12 that takes five to nine hours.
          </Prose>
          <div className="mt-3 space-y-1">
            <p className="type-meta">
              artifacts per run, committed to git so results reproduce on another machine:
            </p>
            <p className="type-num text-[11px] text-muted">
              runs/gepa_&lt;mode&gt;_&lt;ts&gt;/config.json · dspy_optimized_runner.json ·
              dspy_gepa_stats.json · dspy_summary.json · dspy_gepa_logs/
            </p>
          </div>
          <Callout tone="broken" title="Not runnable from this checkout today">
            <p>
              The DSPy path does not send the provider’s thinking-mode override, so every GEPA
              command fails with a 400 before it starts. The champion’s prompts came from a real run
              made before the provider changed; the run directories are committed and inspectable,
              and the run cannot currently be repeated.
            </p>
          </Callout>
        </Panel>

        <Panel>
          <PanelTitle>s7 · the auto-research harness</PanelTitle>
          <Prose className="type-small">
            One failing case at a time, through three steps. <span className="text-text">Diagnose</span>{' '}
            — what went wrong, in which stage. <span className="text-text">Route + fix</span> — a
            router picks which of the four agents owns the failure and that agent’s specialist
            proposes one additional rule. <span className="text-text">Verify</span> — the case is
            re-run; a rule that does not fix it, or that moves the regression set, is written to the
            attempts log and not to the rule store.
          </Prose>
          <ul className="type-small mt-3 space-y-1.5 text-muted">
            <li>
              Kept rules land in <Mono>rules_&lt;agent&gt;_&lt;variant&gt;.jsonl</Mono>; every
              attempt, including the rejected ones, lands in{' '}
              <Mono>rule_attempts_&lt;agent&gt;_&lt;variant&gt;.jsonl</Mono>. The rejections are the
              interesting half.
            </li>
            <li>
              The prompt module <Mono>src/convfinqa/prompts/&lt;variant&gt;.py</Mono> is{' '}
              <em>generated</em> from those stores and never hand-edited, so the prompts a run used
              can always be traced back to the cases that produced them.
            </li>
            <li>
              The diagnosis agents run on the larger model; the pipeline being diagnosed runs on the
              small one.
            </li>
          </ul>
          <Provenance origin="committed">
            stores and per-case results under <Mono>evaluation/diagnostics/</Mono>, tracked in git.{' '}
            <Link
              to="/admin/research"
              className="text-amber underline decoration-amber-line underline-offset-4"
            >
              Open the rule stores →
            </Link>
          </Provenance>
        </Panel>
      </div>

      <Callout title="What the harness actually produced, measured honestly">
        <p>
          The variant it generated moved number-selection accuracy up and program accuracy down, for
          a net loss overall — so it was not promoted, and the table in the previous section shows it
          sitting below the champion rather than being quietly dropped. That is the system working:
          an automated prompt writer that is trusted only as far as a held-out re-run confirms it.
          {data.championName && (
            <>
              {' '}
              The champion is <Mono>{data.championName}</Mono>.
            </>
          )}
        </p>
      </Callout>
    </Section>
  );
}
