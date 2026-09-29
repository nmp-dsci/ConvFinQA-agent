import { PageHead } from '@/kit/ui';
import { Callout, Mono, Provenance, Section } from '../system/ui';
import { shippedCount, useReadiness } from './api';
import { ReadinessBlock } from './Readiness';
import './readiness.css';

/**
 * `/agent/readiness` — the production gen-AI rubric, in the product it grades.
 *
 * The portfolio site scores this system against nine dimensions with proof paths into this
 * repository. This page is the same nine rows inside the running thing, each linking to the
 * in-app page that proves it, so a reader who arrived from the matrix can check the claim against
 * the product.
 */
export default function ReadinessRoute() {
  const query = useReadiness();
  const data = query.data;
  const count = data ? shippedCount(data.rows) : null;
  const partial = data ? data.rows.filter((r) => r.status === 'partial').length : 0;
  const designed = data ? data.rows.filter((r) => r.status === 'designed').length : 0;

  const verdict = data
    ? `${count?.shipped} of ${count?.total} dimensions shipped, ${partial} partial, ${designed} designed-not-built, at rung ${data.rung}, measured ${data.measured}. The confidence judge (R2) was tried, measured, and not adopted; the scale path (R9) is design only.`
    : 'The scorecard is a committed file, checked on every pull request; it has not loaded on this deployment.';

  return (
    <div data-testid="readiness-page">
      <PageHead
        eyebrow="agent · production readiness · evaluation/readiness.json"
        title={
          count ? (
            <>
              {count.shipped} of {count.total} production dimensions <em>shipped</em>, each linked to
              its proof
            </>
          ) : (
            <>
              Nine production dimensions, each linked to its <em>proof</em>
            </>
          )
        }
        lead={verdict}
      />

      <Section
        id="rubric"
        index={1}
        eyebrow="the rubric"
        title="How a version learns, then how it runs once live"
        lede="The rubric the portfolio scores every system on. The learning half asks how a version gets better without getting worse; the operating half asks what happens once it is live."
      >
        <ReadinessBlock detailed />
        <Provenance origin="committed">
          <Mono>evaluation/readiness.json</Mono>, served verbatim by <Mono>/eval/readiness</Mono>.
        </Provenance>
        <Callout title="Statuses are judgements; what CI checks is that they still point at real things">
          <p>
            The file is hand-maintained. CI checks that every proof path exists, every linked page is
            served, and the headline score matches the rows. A status that stops being true is a
            human&rsquo;s job to change — the same job as changing it on the portfolio site, where the
            second copy lives.
          </p>
        </Callout>
      </Section>
    </div>
  );
}
