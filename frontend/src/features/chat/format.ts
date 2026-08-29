/**
 * Formatters shared by the three panes.
 *
 * The rule every one of these obeys: `null`/`undefined` means *not measured*
 * and renders as an em dash, never as `0`. The demo pack carries no latency,
 * token or cost figures at all, so a zero here would be a lie in the
 * flattering direction on every replayed turn.
 */

export const EM_DASH = '—';

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return EM_DASH;
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms).toLocaleString()} ms`;
}

export function fmtTokens(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return EM_DASH;
  return n.toLocaleString();
}

export function fmtCost(usd: number | null | undefined): string {
  if (usd === null || usd === undefined || !Number.isFinite(usd)) return EM_DASH;
  return `$${usd.toFixed(4)}`;
}

export function fmtRelative(ms: number): string {
  const sec = Math.floor((Date.now() - ms) / 1000);
  if (sec < 5) return 'just now';
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  return `${day}d ago`;
}

/**
 * `Double_MAR/2010/page_55.pdf-1` → `MAR · 2010 · p.55`.
 *
 * The sessions pane is 210 px wide; a full report id wraps to three lines and
 * tells the reader nothing the short form does not. The full id stays as the
 * row's `title` and is what the thread header shows, so nothing is lost.
 */
export function shortRid(rid: string): string {
  const match = /^(?:Single|Double)_([^/]+)\/(\d{4})\/page_(\d+)/.exec(rid);
  if (match) return `${match[1]} · ${match[2]} · p.${match[3]}`;
  const parts = rid.split('/');
  return parts.length > 1 ? parts.join(' · ') : rid;
}

/** "Today" / "Yesterday" / a date, for grouping the sessions list. */
export function dayGroup(ms: number): string {
  const then = new Date(ms);
  const now = new Date();
  const startOf = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((startOf(now) - startOf(then)) / 86_400_000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  return then.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

/** Sum that stays `null` when nothing at all was measured. */
export function sumMeasured(values: Array<number | null | undefined>): number | null {
  let total = 0;
  let seen = 0;
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      total += value;
      seen += 1;
    }
  }
  return seen === 0 ? null : total;
}

export function asText(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
