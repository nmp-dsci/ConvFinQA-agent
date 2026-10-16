/**
 * The Runs slot's address rules, apart from its pages so the route table can import them without
 * pulling a lazy page into the first paint.
 */
import { redirect } from 'react-router-dom';
import type { LoaderFunctionArgs } from 'react-router-dom';

/** The legacy page's own lens keys. A link carrying one meant that page, so it goes there. */
export const LEGACY_LENS = ['version', 'baseline', 'candidate', 'flips', 'split', 'report', 'disagree'] as const;

/**
 * `/runs?version=v8` was the legacy versions page's address until s21, when `/runs` became every
 * test pass; a link or bookmark carrying that page's lens lands on `/runs/legacy` with it intact.
 */
export function legacyLensLoader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  if (LEGACY_LENS.some((key) => url.searchParams.has(key))) {
    return redirect(`/runs/legacy${url.search}`);
  }
  return null;
}
