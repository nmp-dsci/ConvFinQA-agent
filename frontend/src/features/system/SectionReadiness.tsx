import { Link } from 'react-router-dom';
import { ReadinessBlock } from '../readiness/Readiness';
import { Section } from './ui';

/**
 * §13: the production gen-AI rubric, scored against this system.
 *
 * Every earlier section on this page is the evidence; this one is the grade.
 * It reads the same committed file the landing strip and `/admin/readiness`
 * read, so the three cannot disagree.
 */
export function ReadinessSection() {
  return (
    <Section
      id="readiness"
      index={13}
      eyebrow="production readiness"
      title="Nine dimensions, scored honestly — and where each one is proved above"
      lede={
        <>
          The portfolio grades every system on the same nine questions: a versioned golden set, a
          judge, a regression gate, a learning loop, guardrails, tracing, cost control, deploy,
          and a scale path. This is the grade for this one, with each row linking to the page in
          the app that carries its proof.{' '}
          <Link to="/admin/readiness" className="text-amber underline underline-offset-4">
            The full page
          </Link>{' '}
          adds the question each dimension asks and the file paths behind it.
        </>
      }
    >
      <ReadinessBlock />
    </Section>
  );
}
