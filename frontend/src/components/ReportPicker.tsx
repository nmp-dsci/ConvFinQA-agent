import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../store';

export function ReportPicker() {
  const open = useStore((s) => s.pickerOpen);
  const close = useStore((s) => s.closePicker);
  const reports = useStore((s) => s.reports);
  const reportsLoading = useStore((s) => s.reportsLoading);
  const reportsError = useStore((s) => s.reportsError);
  const loadReports = useStore((s) => s.loadReports);
  const selectReport = useStore((s) => s.selectReport);
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQuery('');
      // Focus on next tick so the modal has mounted.
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return reports.slice(0, 200);
    return reports.filter((r) => r.toLowerCase().includes(q)).slice(0, 200);
  }, [query, reports]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 flex items-start justify-center pt-24"
      data-testid="report-picker"
      role="dialog"
      aria-modal="true"
      onClick={close}
    >
      <div
        className="bg-panel rounded-lg w-[min(640px,90vw)] max-h-[70vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-3 border-b border-black/40">
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search reports..."
            className="w-full bg-panel2 text-textMain placeholder:text-textMuted px-3 py-2 rounded-md outline-none"
            data-testid="report-picker-input"
            onKeyDown={(e) => {
              if (e.key === 'Escape') close();
              if (e.key === 'Enter' && filtered[0]) {
                void selectReport(filtered[0]);
              }
            }}
          />
          <div className="flex items-center justify-between text-xs text-textMuted mt-1">
            <span>
              {reportsLoading
                ? 'Loading reports…'
                : `${filtered.length} of ${reports.length} reports`}
            </span>
            {reportsError && !reportsLoading && (
              <button
                type="button"
                onClick={() => void loadReports()}
                className="text-danger underline hover:opacity-80"
              >
                Retry
              </button>
            )}
          </div>
        </div>
        <ul className="overflow-y-auto" data-testid="report-picker-list">
          {reportsError && reports.length === 0 ? (
            <li className="px-4 py-3 text-sm text-danger">
              Failed to load reports: {reportsError}. Is the API server
              running on the configured port? (vite dev proxies to
              http://127.0.0.1:8765 by default)
            </li>
          ) : reportsLoading && reports.length === 0 ? (
            <li className="px-4 py-3 text-sm text-textMuted italic">
              Loading reports…
            </li>
          ) : filtered.length === 0 ? (
            <li className="px-4 py-3 text-sm text-textMuted">No matches.</li>
          ) : (
            filtered.map((rid) => (
              <li key={rid}>
                <button
                  type="button"
                  className="w-full text-left px-4 py-2 font-mono text-sm hover:bg-panel2 truncate"
                  onClick={() => void selectReport(rid)}
                  aria-label={rid}
                >
                  {rid}
                </button>
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
}
