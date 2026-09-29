import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';
import * as api from '../../api';
import { qk } from '../../lib/queryClient';
import type { Message } from '../../types';
import { retrievedValues } from './stages';

/**
 * A cell counts as "pulled" when its number is one of the numbers the
 * retriever returned for this turn.
 *
 * The retriever's output is `{question, answer}` pairs — it names the value it
 * found, never the coordinate it found it at. So this match is by value, and
 * the panel says so: it is evidence, not a claim about which lookup the model
 * performed. Matching on the parsed float rather than the string keeps
 * `1,151` and `1151.0` together without letting `11` match `1151`.
 */
function parseNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string') return null;
  const cleaned = value.replace(/[$,%\s]/g, '').replace(/^\((.*)\)$/, '-$1');
  const parsed = Number.parseFloat(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

function useHighlighted(message: Message | null): Set<number> {
  return useMemo(() => {
    const out = new Set<number>();
    if (!message) return out;
    for (const entry of retrievedValues(message)) {
      const parsed = parseNumber(entry.answer);
      if (parsed !== null) out.add(parsed);
    }
    return out;
  }, [message]);
}

function fmtCell(value: number | string | undefined): string {
  if (value === undefined || value === null || value === '') return '—';
  return typeof value === 'number' ? value.toLocaleString() : String(value);
}

interface Props {
  reportId: string;
  /** The turn whose retrieved values are highlighted. */
  message: Message | null;
  onClose: () => void;
}

export function FilingDrawer({ reportId, message, onClose }: Props) {
  const highlighted = useHighlighted(message);
  const doc = useQuery({
    queryKey: qk.reportDocument(reportId),
    queryFn: () => api.getDocument(reportId),
    staleTime: Infinity,
  });

  const table = doc.data?.table ?? {};
  const rowKeys = Object.keys(table);
  const colKeys = useMemo(() => {
    const cols = new Set<string>();
    for (const row of rowKeys) for (const col of Object.keys(table[row] ?? {})) cols.add(col);
    return Array.from(cols);
  }, [table, rowKeys]);

  let hits = 0;
  for (const row of rowKeys) {
    for (const col of colKeys) {
      const parsed = parseNumber(table[row]?.[col]);
      if (parsed !== null && highlighted.has(parsed)) hits += 1;
    }
  }

  return (
    <div data-testid="document-viewer" className="ch-drawer">
      <div className="ch-head">
        {/* The thread header already names the filing; repeating it here would
            spend the only line this bar has on a fact already on screen. */}
        <span className="label">source document</span>
        <span className="ch-fill ch-trunc small muted">
          what the retriever reads: the page’s text and its table
        </span>
        <button type="button" onClick={onClose} aria-label="Close the filing" className="tog">
          ✕ close
        </button>
      </div>

      {/*
        `tabIndex={0}` on both scroll containers: a region that scrolls but
        holds no focusable child cannot be reached by keyboard at all, so a
        reader who does not use a mouse simply cannot see the bottom of a long
        filing or the right-hand columns of a wide table. axe's
        `scrollable-region-focusable`, and it only fires at narrow widths —
        where it matters most.
      */}
      <div role="region" aria-label="Filing text and table" tabIndex={0} className="ch-drawer-body">
        {doc.isPending && <p className="small muted">Loading the filing…</p>}
        {doc.isError && (
          <p className="callout err">Could not load this filing: {String(doc.error)}</p>
        )}

        {doc.data && (
          <>
            {doc.data.pre_text && (
              <section>
                <p className="label">pre-text</p>
                <p className="ch-doc-text">{doc.data.pre_text}</p>
              </section>
            )}

            <section>
              <div className="ch-legend">
                <span className="label">table</span>
                {highlighted.size > 0 && (
                  <span className="small muted">
                    <span className="ch-swatch" aria-hidden />
                    {hits > 0
                      ? `retrieved — ${hits} cell${hits === 1 ? '' : 's'} carrying a value this turn's retriever returned`
                      : 'retrieved — none: the retriever’s values for this turn are not in the table, it read them from the text'}
                  </span>
                )}
              </div>

              <div role="region" aria-label="Filing table" tabIndex={0} className="tw ch-filing-tw">
                <table className="ch-filing">
                  <thead>
                    <tr>
                      {/*
                        The corner cell. It is empty by design — it sits above
                        the row labels and left of the column labels — but an
                        empty <th> is announced as an unnamed column header, so
                        it carries a name only a screen reader hears.
                      */}
                      <th scope="col" className="rowh">
                        <span className="sr-only">Row label</span>
                        <span aria-hidden>&nbsp;</span>
                      </th>
                      {colKeys.map((col) => (
                        <th key={col} scope="col" className="num">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rowKeys.map((row) => (
                      <tr key={row}>
                        <th scope="row" className="rowh">
                          {row}
                        </th>
                        {colKeys.map((col) => {
                          const raw = table[row]?.[col];
                          const parsed = parseNumber(raw);
                          const hit = parsed !== null && highlighted.has(parsed);
                          return (
                            <td
                              key={col}
                              data-retrieved={hit ? 'true' : undefined}
                              title={hit ? 'Returned by the retriever on this turn' : undefined}
                              className={hit ? 'num ch-hit' : 'num'}
                            >
                              {fmtCell(raw)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {highlighted.size > 0 && (
                <p className="small muted">
                  Highlighted by value: the retriever names the number it found, not the cell it
                  came from, so a coincidental equal value elsewhere in the table lights up too.
                </p>
              )}
            </section>

            {doc.data.post_text && (
              <section>
                <p className="label">post-text</p>
                <p className="ch-doc-text">{doc.data.post_text}</p>
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}
