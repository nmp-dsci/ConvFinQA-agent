import { Link } from 'react-router-dom';
import { Loading } from '@/kit/ui';
import { shippedCount, useReadiness } from './api';
import type { Readiness, ReadinessRow, ReadinessStatus } from './api';
import './readiness.css';

/**
 * The scorecard, in three sizes from one file.
 *
 *  - `ReadinessStrip`: nine small links — glyph and ref — for the Overview. Each links to the
 *    row's proof page.
 *  - `ReadinessBlock`: the two halves the rubric defines, each row a card with its status word,
 *    the 40-word "how" and its proof links.
 *  - `/agent/readiness` (`ReadinessRoute`) adds the question each dimension asks and the proof
 *    paths behind a disclosure.
 *
 * Status is a word and a glyph, never a colour alone; the glyphs ● ◐ ○ — survive greyscale,
 * which is the portfolio's own rule. A designed-not-built row is dashed, the house mark for it.
 */

/** The glyph, coloured by status, with the word beside it. */
function StatusMark({ data, status }: { data: Readiness; status: ReadinessStatus }) {
  const s = data.statuses[status];
  return (
    <span data-status={status} title={s.blurb} className="rd-status">
      <span aria-hidden className={`rd-g ${status}`}>
        {s.glyph}
      </span>{' '}
      {s.label}
    </span>
  );
}

function ScoreLine({ data }: { data: Readiness }) {
  const { shipped, total } = shippedCount(data.rows);
  return (
    <>
      <span className="num">{shipped}</span> / <span className="num">{total}</span> shipped · rung{' '}
      <span className="num">{data.rung}</span> · measured <span className="num">{data.measured}</span>
    </>
  );
}

// ---------------------------------------------------------------------------
// Strip
// ---------------------------------------------------------------------------

export function ReadinessStrip({ className }: { className?: string }) {
  const query = useReadiness();
  const data = query.data;

  if (query.isLoading) {
    return (
      <div className={className} aria-label="loading">
        <Loading what="the readiness scorecard" />
      </div>
    );
  }
  if (!data) {
    return (
      <p className={`small muted${className ? ` ${className}` : ''}`} data-testid="readiness-strip-absent">
        The readiness scorecard did not load — /eval/readiness returned nothing for this deployment.
      </p>
    );
  }

  return (
    <div className={`rd-strip${className ? ` ${className}` : ''}`} data-testid="readiness-strip">
      <div className="rd-head">
        <span className="label">
          production readiness · <ScoreLine data={data} />
        </span>
        <Link to="/agent/readiness">the nine dimensions →</Link>
      </div>
      <ol>
        {data.rows.map((row) => {
          const s = data.statuses[row.status];
          return (
            <li key={row.ref}>
              <Link
                to={row.app[0] ?? '/agent/readiness'}
                data-ref={row.ref}
                data-status={row.status}
                title={`${row.ref} ${row.label} — ${s.label}: ${s.blurb}`}
              >
                <span aria-hidden className={`rd-g ${row.status}`}>
                  {s.glyph}
                </span>
                {row.ref}
                <span className="sr-only">
                  {row.label}: {s.label}
                </span>
              </Link>
            </li>
          );
        })}
      </ol>
      <p className="rd-names">
        {data.rows.map((r) => `${r.ref} ${r.short.replace(/­/g, '').toLowerCase()}`).join(' · ')}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

function RowCard({ data, row, detailed }: { data: Readiness; row: ReadinessRow; detailed: boolean }) {
  return (
    <li
      data-testid={`readiness-${row.ref}`}
      data-status={row.status}
      className={`card rd-card ${row.status}`}
    >
      <div className="rd-card-head">
        <h3>
          <span className="num">{row.ref}</span>
          {row.label}
        </h3>
        <StatusMark data={data} status={row.status} />
      </div>
      {detailed && <p className="q">{row.question}</p>}
      <p>{row.how}</p>
      {row.app.length > 0 && (
        <div className="rd-links">
          {row.app.map((to) => (
            <Link key={to} to={to}>
              {to} →
            </Link>
          ))}
        </div>
      )}
      {detailed && row.proof.length > 0 && (
        <details>
          <summary className="small">
            proof · {row.proof.length} path{row.proof.length === 1 ? '' : 's'} in the repo
          </summary>
          <ul>
            {row.proof.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}

export function ReadinessBlock({ detailed = false }: { detailed?: boolean }) {
  const query = useReadiness();
  const data = query.data;

  if (query.isLoading) {
    return <Loading what="the readiness scorecard" />;
  }
  if (!data) {
    return (
      <p className="empty">
        The readiness scorecard did not load — /eval/readiness returned nothing for this
        deployment. It is a committed file, so a fresh checkout carries it.
      </p>
    );
  }

  return (
    <div data-testid="readiness-block" className="rd-block">
      <p className="small">
        <ScoreLine data={data} />.
      </p>
      <ul className="rd-legend" aria-label="status legend">
        {(Object.keys(data.statuses) as ReadinessStatus[]).map((key) => (
          <li key={key}>
            <StatusMark data={data} status={key} /> — {data.statuses[key].blurb}
          </li>
        ))}
      </ul>
      {data.halves.map((half, i) => {
        const from = i === 0 ? 0 : data.halves[i - 1].upto;
        const rows = data.rows.filter((r) => r.order > from && r.order <= half.upto);
        return (
          <section key={half.label} className="rd-half">
            <h3>
              {half.label}
              <span className="label">{half.range}</span>
            </h3>
            <p className="small muted">{half.blurb}</p>
            <ul className="cards">
              {rows.map((row) => (
                <RowCard key={row.ref} data={data} row={row} detailed={detailed} />
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
