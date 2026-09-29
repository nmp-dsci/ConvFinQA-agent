// nmp-ui-kit 0.1.0 · vendored from nmp-central-ai/ui/kit — do not edit here; change the kit and run `make ui-sync`.
/**
 * The URL grammar every house app follows (DataAgentBench's `lib/url.ts`, generalised):
 *
 * 1. One id, the same string everywhere — in tables, APIs and URLs.
 * 2. The path names the subject; the query string holds the lens. If dropping a value leaves
 *    nothing to show, it is the subject and goes in the path. If dropping it falls back to a
 *    default (a filter, a compared run, a selected tab) it is a lens.
 * 3. Picking from a list opens the detail in place: a nested route renders into the list page's
 *    <Outlet>, so the address changes and the list stays.
 *
 * Projects add their own `*Path()` builders beside these helpers, in their own `lib/url.ts`.
 */
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';

export type Lens = Record<string, string | number | boolean | null | undefined>;

/** Escape one path segment, leaving `:` and `,` readable. */
export function enc(v: string): string {
  return encodeURIComponent(v).replace(/%3A/gi, ':').replace(/%2C/gi, ',');
}

/** An id whose slashes are separators: each segment escaped, the slashes kept. */
export function encPath(id: string): string {
  return id.split('/').map(enc).join('/');
}

function encQuery(v: string): string {
  return encodeURIComponent(v).replace(/%2F/gi, '/').replace(/%3A/gi, ':').replace(/%2C/gi, ',');
}

/** `?a=1&b=x/y`, skipping empty values; '' when nothing is set. Keys keep their given order. */
export function search(lens?: Lens): string {
  const parts = Object.entries(lens ?? {})
    .filter((e): e is [string, string | number | boolean] => e[1] != null && e[1] !== '' && e[1] !== false)
    .map(([k, v]) => `${encQuery(k)}=${encQuery(String(v))}`);
  return parts.length ? `?${parts.join('&')}` : '';
}

/** The current lens with `patch` applied (null or '' removes a key), as a `?…` string. */
export function patchLens(current: URLSearchParams, patch: Lens): string {
  const next: Lens = Object.fromEntries(current.entries());
  for (const [k, v] of Object.entries(patch)) next[k] = v;
  return search(next);
}

/** The page's lens and a setter that rewrites it in place (replace, not push). */
export function useLens(): [URLSearchParams, (patch: Lens) => void] {
  const [sp] = useSearchParams();
  const nav = useNavigate();
  const { pathname, hash } = useLocation();
  const set = (patch: Lens) => nav({ pathname, search: patchLens(sp, patch), hash }, { replace: true });
  return [sp, set];
}
