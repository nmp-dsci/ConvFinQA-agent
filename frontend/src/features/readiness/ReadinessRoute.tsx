import { AdminPage, Caveat, Panel } from '../admin/ui';
import { shippedCount, useReadiness } from './api';
import { ReadinessBlock } from './Readiness';

/**
 * `/admin/readiness` — the production gen-AI rubric, in the product it grades.
 *
 * The portfolio site scores this system against nine dimensions with proof
 * paths into this repository. This page is the same nine rows inside the
 * running thing, each linking to the in-app page that proves it, so a reader
 * who arrived from the matrix can check the claim against the product.
 */
export default function ReadinessRoute() {
  const query = useReadiness();
  const data = query.data;
  const count = data ? shippedCount(data.rows) : null;
  const partial = data ? data.rows.filter((r) => r.status === 'partial').length : 0;
  const designed = data ? data.rows.filter((r) => r.status === 'designed').length : 0;

  const verdict = data
    ? `${count?.shipped} of ${count?.total} dimensions shipped, ${partial} partial, ${designed} designed-not-built, at rung ${data.rung}. The confidence judge (R2) was tried, measured, and not adopted; the scale path (R9) is design only.`
    : 'The scorecard is a committed file, checked on every pull request; it has not loaded on this deployment.';

  return (
    <AdminPage
      title="Production readiness"
      sub={verdict}
      testId="readiness-page"
      actions={
        data && (
          <>
            <span className="type-num type-meta rounded-[4px] border border-line-2 px-1.5 py-0.5 text-muted">
              measured {data.measured}
            </span>
            <span className="type-num type-meta rounded-[4px] border border-line-2 px-1.5 py-0.5 text-muted">
              evaluation/readiness.json
            </span>
          </>
        )
      }
    >
      <Panel
        title="nine dimensions, two halves"
        endpoint="/eval/readiness"
        note="The rubric the portfolio scores every system on. The learning half asks how a version gets better without getting worse; the operating half asks what happens once it is live."
      >
        <ReadinessBlock detailed />
        <Caveat>
          Statuses are judgements, not metrics, so the file is hand-maintained. What is checked in
          CI is that it still points at real things: every proof path exists, every linked page is
          served, and the headline score matches the rows. A status that stops being true is a
          human's job to change — the same job as changing it on the portfolio site, where the
          second copy lives.
        </Caveat>
      </Panel>
    </AdminPage>
  );
}
