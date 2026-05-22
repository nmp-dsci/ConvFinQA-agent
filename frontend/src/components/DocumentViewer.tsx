import { useEffect, useState } from 'react';
import * as api from '../api';
import type { ReportDocument } from '../types';

interface Props {
  reportId: string;
}

function fmtCell(value: number | string | undefined): string {
  if (value === undefined || value === null || value === '') return '—';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString();
  }
  return String(value);
}

function DocumentTable({
  table,
}: {
  table: Record<string, Record<string, number | string>>;
}) {
  const rowKeys = Object.keys(table);
  if (rowKeys.length === 0) {
    return <div className="text-textMuted italic">(no table data)</div>;
  }
  // Use the first row's keys to seed columns; union the rest in case rows differ.
  const colSet = new Set<string>();
  for (const r of rowKeys) {
    for (const c of Object.keys(table[r] ?? {})) colSet.add(c);
  }
  const colKeys = Array.from(colSet);

  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse min-w-full">
        <thead>
          <tr className="bg-panel2/60">
            <th className="text-left px-2 py-1 border border-black/40 font-semibold sticky left-0 bg-panel2/60">
              &nbsp;
            </th>
            {colKeys.map((col) => (
              <th
                key={col}
                className="text-left px-2 py-1 border border-black/40 font-semibold whitespace-nowrap"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rowKeys.map((row) => (
            <tr key={row} className="hover:bg-panel2/30">
              <th className="text-left px-2 py-1 border border-black/40 font-medium text-textMain sticky left-0 bg-panel">
                {row}
              </th>
              {colKeys.map((col) => (
                <td
                  key={col}
                  className="px-2 py-1 border border-black/40 font-mono text-right whitespace-nowrap"
                >
                  {fmtCell(table[row]?.[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DocumentViewer({ reportId }: Props) {
  const [doc, setDoc] = useState<ReportDocument | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDoc(null);
    setError(null);
    setLoading(true);
    api
      .getDocument(reportId)
      .then((d) => {
        if (!cancelled) setDoc(d);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [reportId]);

  return (
    <div
      data-testid="document-viewer"
      className="border-b border-black/40 bg-panel/60 px-4 py-3 max-h-[45vh] overflow-y-auto overflow-x-hidden min-w-0"
    >
      {loading && (
        <div className="text-textMuted italic text-sm">Loading document…</div>
      )}
      {error && (
        <div className="text-danger text-sm">Failed to load document: {error}</div>
      )}
      {doc && (
        <div className="space-y-3 text-sm">
          {doc.pre_text && (
            <section>
              <h3 className="text-textMuted text-xs uppercase tracking-wide mb-1">
                Pre-text
              </h3>
              <p className="whitespace-pre-wrap leading-relaxed">{doc.pre_text}</p>
            </section>
          )}
          <section>
            <h3 className="text-textMuted text-xs uppercase tracking-wide mb-1">
              Table
            </h3>
            <DocumentTable table={doc.table} />
          </section>
          {doc.post_text && (
            <section>
              <h3 className="text-textMuted text-xs uppercase tracking-wide mb-1">
                Post-text
              </h3>
              <p className="whitespace-pre-wrap leading-relaxed">{doc.post_text}</p>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
