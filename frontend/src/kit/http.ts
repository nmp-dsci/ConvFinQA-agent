// nmp-ui-kit 0.1.0 · vendored from nmp-central-ai/ui/kit — do not edit here; change the kit and run `make ui-sync`.
/** The whole data layer a read-only page needs: GET, POST, and a hook. Projects with a richer
 *  client (react-query, SSE) keep theirs; this is the floor, not a ceiling. */
import { useEffect, useState } from 'react';

export async function get<T>(url: string): Promise<T> {
  const r = await fetch(url, { headers: { accept: 'application/json' } });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return (await r.json()) as T;
}

export async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((j as { detail?: string }).detail ?? `${r.status} ${url}`);
  return j as T;
}

/** Fetch `url` (null = do nothing); bump `nonce` to refetch. */
export function useGet<T>(url: string | null, nonce = 0): { data: T | null; error: string | null; loading: boolean } {
  const [state, set] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: !!url });
  useEffect(() => {
    if (!url) return;
    let alive = true;
    set({ data: null, error: null, loading: true });
    get<T>(url)
      .then((d) => alive && set({ data: d, error: null, loading: false }))
      .catch((e: Error) => alive && set({ data: null, error: e.message, loading: false }));
    return () => {
      alive = false;
    };
  }, [url, nonce]);
  return state;
}
