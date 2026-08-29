import { useQuery } from '@tanstack/react-query';
import { X } from 'lucide-react';
import { useMemo } from 'react';
import { cn } from '@/lib/utils';
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
    <div
      data-testid="document-viewer"
      className="absolute inset-0 z-20 flex min-h-0 flex-col bg-panel"
    >
      <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-b border-line px-3">
        {/* The thread header already names the filing; repeating it here would
            spend the only line this bar has on a fact already on screen. */}
        <span className="mono-caps">source document</span>
        <span className="min-w-0 flex-1 truncate text-[10.5px] text-faint">
          what the retriever reads: the page’s text and its table
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close the filing"
          className="shrink-0 rounded-[4px] border border-line-2 p-1 text-muted transition-colors hover:border-amber-line hover:text-amber"
        >
          <X className="size-3" aria-hidden />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-3 py-3">
        {doc.isPending && <p className="text-[12px] text-faint">Loading the filing…</p>}
        {doc.isError && (
          <p className="text-[12px] text-bad">
            Could not load this filing: {String(doc.error)}
          </p>
        )}

        {doc.data && (
          <div className="space-y-3">
            {doc.data.pre_text && (
              <section>
                <div className="mono-caps mb-1">pre-text</div>
                <p className="text-[12px] leading-relaxed whitespace-pre-wrap text-muted">
                  {doc.data.pre_text}
                </p>
              </section>
            )}

            <section>
              <div className="mb-1 flex flex-wrap items-baseline gap-2">
                <span className="mono-caps">table</span>
                {highlighted.size > 0 && (
                  <span className="text-[10.5px] text-faint">
                    <span className="mr-1 inline-block size-2 translate-y-px rounded-[2px] border border-amber-line bg-amber-soft" />
                    {hits > 0
                      ? `${hits} cell${hits === 1 ? '' : 's'} carrying a value this turn's retriever returned`
                      : 'the retriever’s values for this turn are not in the table — it read them from the text'}
                  </span>
                )}
              </div>

              <div className="overflow-x-auto rounded-md border border-line">
                <table className="min-w-full border-collapse text-[11px]">
                  <thead>
                    <tr>
                      <th className="sticky left-0 z-10 border-b border-line bg-panel-2 px-2 py-1 text-left font-medium">
                        &nbsp;
                      </th>
                      {colKeys.map((col) => (
                        <th
                          key={col}
                          className="border-b border-l border-line bg-panel-2 px-2 py-1 text-left font-medium whitespace-nowrap"
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rowKeys.map((row) => (
                      <tr key={row}>
                        <th className="sticky left-0 z-10 border-b border-line bg-panel px-2 py-1 text-left font-medium text-muted">
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
                              className={cn(
                                'border-b border-l border-line px-2 py-1 text-right font-mono whitespace-nowrap',
                                hit
                                  ? 'bg-amber-soft text-amber shadow-[inset_0_0_0_1px_var(--amber-line)]'
                                  : 'text-muted'
                              )}
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
                <p className="mt-1.5 text-[10px] leading-relaxed text-faint">
                  Highlighted by value: the retriever names the number it found, not the cell it
                  came from, so a coincidental equal value elsewhere in the table lights up too.
                </p>
              )}
            </section>

            {doc.data.post_text && (
              <section>
                <div className="mono-caps mb-1">post-text</div>
                <p className="text-[12px] leading-relaxed whitespace-pre-wrap text-muted">
                  {doc.data.post_text}
                </p>
              </section>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
