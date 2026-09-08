import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/dialog';
import { useStore } from '../../store';
import { shortRid } from './format';

/**
 * Filing chooser. Kept as its own dialog rather than folded into ⌘K because it
 * is the one list a visitor has to browse rather than recall — 200+ report ids
 * nobody has memorised. ⌘K offers the same action for people who do.
 *
 * Built on the vendored Dialog so the focus trap, escape, overlay click, the
 * open/close animation and its reduced-motion behaviour come from one place
 * rather than being hand-rolled here.
 */
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
    if (!open) return;
    setQuery('');
    const id = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => window.clearTimeout(id);
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return reports.slice(0, 200);
    return reports.filter((r) => r.toLowerCase().includes(q)).slice(0, 200);
  }, [query, reports]);

  return (
    <Dialog open={open} onOpenChange={(next) => !next && close()}>
      <DialogContent
        data-testid="report-picker"
        showCloseButton={false}
        className="top-24 flex max-h-[70vh] w-[min(640px,90vw)] max-w-none translate-y-0 flex-col gap-0 overflow-hidden border-line-2 bg-panel p-0 sm:max-w-none"
      >
        <DialogTitle className="sr-only">Choose a filing</DialogTitle>
        <DialogDescription className="sr-only">
          Search the report ids and press Enter to open the first match.
        </DialogDescription>
        <div className="border-b border-line p-3">
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search filings…"
            data-testid="report-picker-input"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && filtered[0]) void selectReport(filtered[0]);
            }}
            className="type-body w-full rounded-md border border-line bg-ground px-3 py-2 text-text outline-none focus:border-amber-line"
          />
          <div className="type-meta mt-1 flex items-center justify-between text-faint">
            <span>
              {reportsLoading ? 'Loading filings…' : `${filtered.length} of ${reports.length} filings`}
            </span>
            {reportsError && !reportsLoading && (
              <button
                type="button"
                onClick={() => void loadReports()}
                className="text-bad underline underline-offset-2 hover:opacity-80"
              >
                Retry
              </button>
            )}
          </div>
        </div>

        <ul className="overflow-y-auto" data-testid="report-picker-list">
          {reportsError && reports.length === 0 ? (
            <li className="type-small px-4 py-3 text-bad">
              Could not load the filing list: {reportsError}. Is the API server running? Vite
              proxies to http://127.0.0.1:8765 by default.
            </li>
          ) : reportsLoading && reports.length === 0 ? (
            <li className="type-small px-4 py-3 text-faint">Loading filings…</li>
          ) : filtered.length === 0 ? (
            <li className="type-small px-4 py-3 text-faint">No matches.</li>
          ) : (
            filtered.map((rid) => (
              <li key={rid}>
                <button
                  type="button"
                  aria-label={rid}
                  onClick={() => void selectReport(rid)}
                  title={shortRid(rid)}
                  className="type-num type-small block w-full truncate px-4 py-2 text-left text-text transition-colors hover:bg-panel-2"
                >
                  {/*
                    The row's text is exactly the report id and nothing else.
                    `smoke.spec.ts` reads it with innerText and then asserts the
                    thread header shows the same string, so a second line here —
                    a short form, a badge — silently breaks that comparison.
                  */}
                  {rid}
                </button>
              </li>
            ))
          )}
        </ul>
      </DialogContent>
    </Dialog>
  );
}
