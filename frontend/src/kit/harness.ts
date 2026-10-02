// nmp-ui-kit 1.0.0 · vendored from nmp-central-ai/ui/kit — do not edit here; change the kit and run `make ui-sync`.
/**
 * `land(routes, url)` — drive the real route table in a memory router (loaders and redirects run,
 * nothing renders) and report where an address ends up. `routes.test.tsx` uses it to prove that
 * every address the app ever published still lands.
 */
import { createMemoryRouter, type RouteObject } from 'react-router-dom';

export type Landing = { path: string; search: string; route: string | undefined };

export async function land(routes: RouteObject[], url: string): Promise<Landing> {
  const router = createMemoryRouter(routes, { initialEntries: [url] });
  await new Promise<void>((resolve) => {
    if (router.state.initialized && router.state.navigation.state === 'idle') return resolve();
    const off = router.subscribe((s) => {
      if (s.initialized && s.navigation.state === 'idle') {
        off();
        resolve();
      }
    });
  });
  const { location, matches } = router.state;
  const leaf = matches[matches.length - 1];
  const out = { path: location.pathname, search: location.search, route: leaf?.route.id };
  router.dispose();
  return out;
}
