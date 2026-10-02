import type { ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Loading, PageHead } from '@/kit/ui';
import { EvaluationSection, PromotionSection } from '@/features/system/SectionContract';
import { useSystemData } from '@/features/system/useSystemData';
import { getCampaigns } from '@/features/admin/api';
import { ErrorNote } from '@/features/admin/ui';
import { JudgeEvidence } from '@/features/admin/judge';
import './scoring.css';

/**
 * Slot 3 · how a turn is judged (DESIGN.md §2 — DataAgentBench's Validators, tau2-loop's
 * Rubric). Four parts, in the order a number is made: the metric a turn is scored on, the walk
 * that charges a wrong turn to one agent, the gate a challenger must pass, and the confidence
 * judge that was tried as a guard-rail and not adopted. The judge reads the same
 * `/eval/campaigns` payload (and query key) as Optimise › Runtimes.
 */

function DocSection({
  id,
  n,
  label,
  title,
  lead,
  children,
}: {
  id: string;
  n: number;
  label: string;
  title: ReactNode;
  lead?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} data-testid={`scoring-section-${id}`} className="doc-section">
      <p className="label">
        {String(n).padStart(2, '0')} · {label}
      </p>
      <h2>{title}</h2>
      {lead && <p className="lead">{lead}</p>}
      <div className="body">{children}</div>
    </section>
  );
}

function AttributionSection() {
  return (
    <DocSection
      id="attribution"
      n={2}
      label="attribution"
      title="A wrong turn is charged to the first agent whose check fails — or to no agent at all"
      lead={
        <>
          Attribution is derived from gold, not asked of a model: <code>stage_scores.first_fault()</code>{' '}
          walks the pipeline in the order its data depends and returns the first failing check.
        </>
      }
    >
      <ol className="steps">
        <li>
          <b>triage</b> — the turn type, number or program, against gold&rsquo;s.
        </li>
        <li>
          <b>preprocess</b> — its sub-questions and its plan. The plan is symbolic (
          <code>divide(A, B)</code>, each placeholder bound to one of its own sub-questions, in
          order), so it cannot be evaluated until operand coverage is known.
        </li>
        <li>
          <b>retriever</b> — operand coverage: whether its answers hold the operands the gold
          program needs.
        </li>
        <li>
          <b>calculator</b> — the plan run: <code>program_exec.bind_and_execute</code> binds the
          placeholders to the retriever&rsquo;s answers, runs the DSL and compares to gold. A plan
          that cannot be bound returns <code>None</code>, never <code>False</code>, because
          &ldquo;unreadable&rdquo; and &ldquo;wrong&rdquo; are different claims.
        </li>
      </ol>

      <div>
        <p className="label">three verdicts that charge no agent · NON_AGENT</p>
        <dl className="kv sc-kv">
          <dt>gold_suspect</dt>
          <dd>
            Fires only when the document text is actually held — with no text, every operand would
            read as absent.
          </dd>
          <dt>ambiguous</dt>
          <dd>
            The ~7% the record cannot settle: a missing operand that <i>is</i> in the document,
            where the retriever declined a sub-question. The teacher is asked one binary question —
            did any sub-question ask for this number? — and is never shown the gold answer. Yes →
            retriever; no → preprocess; a failed adjudication stays unresolved rather than guessed.
          </dd>
          <dt>unscorable</dt>
          <dd>No agent is charged.</dd>
        </dl>
        <p className="small muted">
          All three are excluded from targeting&rsquo;s numerator <i>and</i> denominator, so a bad
          gold label is not billed to an innocent agent. The teacher is told the attribution, must
          justify it, and may dissent; a dissent is recorded as <code>attribution_disputed</code>.
        </p>
      </div>

      <p className="callout note">
        The walk was rewritten on 2026-09-04 after hand-attributing seven failures and disagreeing
        with the previous rule on five: re-scoring all twelve committed runs moved 37.4% of 554
        first-wrong cases, and cut the retriever&rsquo;s share from 152 to 58. Source:{' '}
        <code>src/convfinqa/evalloop/stage_scores.py</code>.
      </p>
    </DocSection>
  );
}

function JudgeSection() {
  const query = useQuery({
    // The same key Optimise › Runtimes and Experiments use: one payload, one cache entry.
    queryKey: ['eval-campaigns'],
    queryFn: () => getCampaigns(),
    staleTime: 60_000,
  });
  const judge = query.data?.judge ?? null;
  const verdict = judge?.verdict ?? null;
  const adopted = Boolean(verdict?.significant && verdict?.meets_target);
  const title = !verdict
    ? 'A confidence judge was built for the SDK runtime; no verdict is recorded yet'
    : adopted
      ? 'The confidence judge clears the bar, so its band gates what is released'
      : 'The confidence judge did not beat releasing everything, so its band is advisory only';

  return (
    <DocSection
      id="judge"
      n={4}
      label="the confidence judge"
      title={title}
      lead="Tried last, after the runtime decision. A Haiku 4.5 judge reads each finished turn — question, filing, sub-questions, retrieved cells with their cited sources, program, calculator trajectory, answer — and returns a band: high releases the answer, low withholds it. It never sees the gold answer."
    >
      <p className="small muted">
        Trained the way sdk_v1 was: a teacher diagnosed every case of a balanced optimise split with
        gold, and one distil call wrote the prompt. Scored at natural prevalence on a disjoint
        calibrate split, then once on the gate split by the frozen champion.
      </p>
      {query.isLoading ? (
        <Loading what="the judge record" />
      ) : query.error ? (
        <ErrorNote error={query.error} />
      ) : (
        <JudgeEvidence judge={judge} />
      )}
      <p className="small">
        <Link to="/optimise/runtimes">The runtime the judge was calibrated to →</Link>
      </p>
    </DocSection>
  );
}

export default function Scoring() {
  const data = useSystemData();
  return (
    <div data-testid="scoring-route">
      <PageHead
        eyebrow="scoring · how a turn is judged"
        title={
          <>
            A turn counts when its executed answer matches gold, and a challenger moves only on{' '}
            <em>significant</em> unseen evidence
          </>
        }
        lead="Four things decide what a number on this site means: the metric a turn is scored on, the walk that charges a wrong turn to one agent, the gate a challenger must pass, and the confidence judge that was tried as a guard-rail."
      />
      <EvaluationSection data={data} index={1} />
      <AttributionSection />
      <PromotionSection data={data} index={3} />
      <JudgeSection />
    </div>
  );
}
