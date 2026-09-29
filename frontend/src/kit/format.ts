// nmp-ui-kit 0.1.0 · vendored from nmp-central-ai/ui/kit — do not edit here; change the kit and run `make ui-sync`.
/** Formatting every page shares. An absent value is `—`, never 0, never blank. */

export const NO_VALUE = '—';

export function fmtPct(x: number | null | undefined, digits = 0): string {
  return x == null || Number.isNaN(x) ? NO_VALUE : `${(x * 100).toFixed(digits)}%`;
}
/** A signed difference in percentage points: +1.4pp, −0.3pp. */
export function fmtPp(x: number | null | undefined, digits = 1): string {
  if (x == null || Number.isNaN(x)) return NO_VALUE;
  const v = (x * 100).toFixed(digits);
  return `${x > 0 ? '+' : x < 0 ? '−' : '±'}${v.replace('-', '')}pp`;
}
export function fmtInt(n: number | null | undefined): string {
  return n == null ? NO_VALUE : n.toLocaleString('en-GB');
}
export function fmtK(n: number | null | undefined): string {
  if (n == null) return NO_VALUE;
  return n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n);
}
export function fmtUsd(x: number | null | undefined, digits = 2): string {
  return x == null ? NO_VALUE : `$${x.toFixed(digits)}`;
}
/** Milliseconds as the unit a person reads: 840ms, 12s, 3.4m. */
export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return NO_VALUE;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return ms >= 60000 ? `${(ms / 60000).toFixed(1)}m` : `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)}s`;
}
export function shortSha(sha: string | null | undefined): string {
  return sha ? sha.slice(0, 7) : NO_VALUE;
}
