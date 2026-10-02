import { Link } from 'react-router-dom';
import { Kpi, Status } from '@/kit/ui';
import { versionLabel } from '../admin/lib';
import { formatPercent, formatPointsDelta } from '../landing/format';
import { MiniBar } from './charts';
import {
  Callout,
  Field,
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
import './system.css';

// ---------------------------------------------------------------------------
// Evaluation — the two-number rule
// ---------------------------------------------------------------------------

export function EvaluationSection({ data, index = 1 }: { data: SystemData; index?: number }) {
  const { versions, versionHoldouts, championName } = data;

  const holdoutOf = (version: string) =>
    versionHoldouts?.find((v) => v.version === version) ?? null;

  return (
    <Section
      id="evaluation"
      index={index}
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
      <div className="kpis">
        <Kpi
          n="120"
          b={
            <>
              <span className="label">optimizer_train</span> conversations GEPA and the s7 harness
              optimised against. Accuracy here says nothing about generalisation: the prompts were
              tuned on it.
            </>
          }
        />
        <Kpi
          n="80"
          b={
            <>
              <span className="label">never_seen</span> conversations, 309 questions no optimizer
              ever saw. The only subset that supports a generalisation claim.
            </>
          }
        />
        <Kpi
          n="770"
          b={
            <>
              <span className="label">sampled · the scored set</span> questions across all 200
              conversations. It mixes seen and unseen, so it is “overall”, never “held out”.
            </>
          }
        />
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
                <Th numeric>questions</Th>
                <Th>state</Th>
              </tr>
            </thead>
            <tbody>
              {(versions ?? []).map((v) => {
                const holdout = holdoutOf(v.version);
                const isChampion = v.version === championName;
                return (
                  <tr key={v.version}>
                    <Td className="sub mono">{versionLabel(v.version)}</Td>
                    <Td numeric>
                      {formatPercent(v.exe_acc)}
                      <MiniBar pct={v.exe_acc * 100} />
                    </Td>
                    <Td numeric>
                      <Live
                        value={holdout ? formatPercent(holdout.holdout_accuracy) : null}
                        reason="/admin/experiments did not answer"
                      />
                      {holdout && <span className="path">{holdout.holdout_n_questions} questions</span>}
                    </Td>
                    <Td numeric>
                      {formatPercent(v.prog_acc)}
                      <span className="path">
                        {v.n_program_correct}/{v.n_program_turns}
                      </span>
                    </Td>
                    <Td numeric>{v.n_questions}</Td>
                    <Td>
                      <Status tone={isChampion ? 'ok' : 'no'}>
                        {isChampion ? 'champion' : 'registered'}
                      </Status>
                    </Td>
                  </tr>
                );
              })}
              {(versions ?? []).length === 0 && (
                <tr>
                  <Td className="muted">
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
          <Link to="/runs">Open the rows behind these figures →</Link>
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
// The promotion contract
// ---------------------------------------------------------------------------

export function PromotionSection({ data, index = 2 }: { data: SystemData; index?: number }) {
  const { gate, gateCandidate, championName } = data;

  return (
    <Section
      id="promotion"
      index={index}
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
      <div className="sys-cols">
        <Panel>
          <PanelTitle>accuracy_ok</PanelTitle>
          <Prose className="small">
            The candidate’s accuracy over the rows both versions answered is at least the
            champion’s, to within a floating-point epsilon. Full-frame accuracy is computed and
            reported but never drives the decision — comparing two versions on different populations
            is not a comparison.
          </Prose>
        </Panel>
        <Panel>
          <PanelTitle>no_regressions</PanelTitle>
          <Prose className="small">
            Not a single pass→fail flip on any shared question. Every flip is listed by report id
            and question, so a refusal names the turns it is refusing over rather than a percentage.
          </Prose>
        </Panel>
        <Panel>
          <PanelTitle>promotable</PanelTitle>
          <Prose className="small">
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
        <div className="panel-head">
          <h3>The comparator, run right now on this deployment</h3>
          {gate && (
            <span className="right">
              <Status tone={gate.promotable ? 'ok' : 'err'}>
                {gate.promotable ? 'promotable' : 'refused'}
              </Status>
            </span>
          )}
        </div>

        {gate ? (
          <>
            <div className="sys-fields cols">
              <Field
                label="candidate"
                note={`against champion ${versionLabel(gate.baseline_version)}`}
              >
                <span className="num">{versionLabel(gate.candidate_version)}</span>
              </Field>
              <Field label="accuracy delta" note={`over ${gate.n_compared} shared questions`}>
                <span className={`num ${gate.accuracy_delta >= 0 ? 'v-ok' : 'v-fail'}`}>
                  {formatPointsDelta(gate.accuracy_delta)}
                </span>
              </Field>
              <Field label="pass → fail flips" note="any single one refuses the promotion">
                <span className={`num ${gate.regressions.length === 0 ? 'v-ok' : 'v-fail'}`}>
                  {gate.regressions.length}
                </span>
              </Field>
              <Field label="fail → pass flips" note="what the candidate did fix">
                <span className="num">{gate.improvements.length}</span>
              </Field>
            </div>
            <p className="small muted">{gate.reason}</p>
          </>
        ) : (
          <p className="small muted">
            <Live value={null} reason="/admin/compare did not answer" /> no live comparison —{' '}
            {gateCandidate && championName
              ? `${championName} vs ${gateCandidate} could not be read from /admin/compare.`
              : 'there is no registered version to compare against the champion on this deployment.'}
          </p>
        )}
        <Provenance origin="live">
          computed on request from the committed prediction CSVs — the same function the CLI and CI
          call, not a re-implementation.{' '}
          <Link to="/optimise">See the flips →</Link>
        </Provenance>
      </Panel>

      <Panel>
        <PanelTitle>the CI gate</PanelTitle>
        <ol className="steps">
          <li>
            <b>Re-score every committed version.</b> Each row’s{' '}
            <Mono>correct</Mono> column is re-derived from its predicted and gold answers with the
            same numeric matcher the evaluator uses. Any disagreement fails the build — a committed
            CSV whose verdicts have drifted from the scorer is a silently wrong artefact, and every
            number on this site is computed from those CSVs.
          </li>
          <li>
            <b>Hold the champion to its recorded floor.</b> Accuracy
            recomputed from the CSV must be within <Mono>0.005</Mono> of the figure the registry
            recorded when it was promoted.
          </li>
          <li>
            <b>Report the challenger comparison.</b> If a challenger
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
// GEPA and the s7 harness
// ---------------------------------------------------------------------------

export function OptimisationSection({ data, index = 3 }: { data: SystemData; index?: number }) {
  return (
    <Section
      id="optimisation"
      index={index}
      eyebrow="optimisation"
      title="Prompts written by optimizers, kept only when a re-run confirms"
      lede={
        <>
          No exemplars are ever put in front of these agents. The instructions themselves are the
          thing being optimised — first by GEPA over the whole pipeline, then by a per-case harness
          that diagnoses one failure at a time and only keeps a fix that survives a re-run.
        </>
      }
    >
      <div className="sys-two">
        <Panel>
          <PanelTitle>GEPA · reflective prompt evolution</PanelTitle>
          <Prose className="small">
            The DSPy optimizer proposes instruction rewrites, scores them against the 120
            optimizer-train conversations with a turn-level metric that feeds textual feedback back
            into the next proposal, and keeps what wins. Two modes: a smoke run capped at 120 metric
            calls over 5 validation conversations, which checks the wiring and is not a transferable
            optimization, and a real run over 12 that takes five to nine hours.
          </Prose>
          <p className="small muted">
            Artifacts per run, committed to git so results reproduce on another machine:
          </p>
          <p className="sys-where">
            runs/gepa_&lt;mode&gt;_&lt;ts&gt;/config.json · dspy_optimized_runner.json ·
            dspy_gepa_stats.json · dspy_summary.json · dspy_gepa_logs/
          </p>
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
          <Prose className="small">
            One failing case at a time, through three steps. <b>Diagnose</b>{' '}
            — what went wrong, in which stage. <b>Route + fix</b> — a
            router picks which of the four agents owns the failure and that agent’s specialist
            proposes one additional rule. <b>Verify</b> — the case is
            re-run; a rule that does not fix it, or that moves the regression set, is written to the
            attempts log and not to the rule store.
          </Prose>
          <ul className="sys-list">
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
            <Link to="/optimise/research">Open the rule stores →</Link>
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
