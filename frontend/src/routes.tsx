/**
 * The route table, as data (ui/DESIGN.md §2): eight slots, each slot's pages nested under it, and
 * a redirect loader for every address this app ever published — the Console's `/admin/*` pages,
 * `/chat/*`, `/debrief`, `/welcome` — so no bookmark, README link or readiness proof path 404s.
 * `routes.test.tsx` lands every one of them.
 *
 * The chat route is a splat (`agent/chat/*`), not `:reportId`: report ids contain slashes
 * (`Single_VLO/2011/page_126.pdf-1`). It is the one `wide` route: full-height, no footer.
 */
import { lazy } from 'react';
import {
  isRouteErrorResponse,
  Link,
  type LoaderFunctionArgs,
  redirect,
  type RouteObject,
  useRouteError,
} from 'react-router-dom';
import { Root, SlotLayout } from './app/Root';
import { ChatRoute } from './features/chat/ChatRoute';
import { LandingRoute } from './features/landing/LandingRoute';
import { legacyLensLoader } from './features/story/lens';

// Everything but the landing board and chat is lazy: the first paint never pays for pages most
// visitors never open. The kit Shell holds the <Suspense> boundary these resolve into.
const Filings = lazy(() => import('./features/admin/DatasetReview'));
const Scoring = lazy(() => import('./pages/Scoring'));
const Leaderboard = lazy(() => import('./pages/Leaderboard'));
const Evaluations = lazy(() => import('./features/admin/Evaluations'));
const RunsPage = lazy(() => import('./features/story/RunsPage'));
const StoryPage = lazy(() => import('./features/story/StoryPage'));
const Traces = lazy(() => import('./features/admin/Traces'));
const TraceDetail = lazy(() => import('./features/admin/TraceDetail'));
const Experiments = lazy(() => import('./features/admin/Experiments'));
const Runtimes = lazy(() => import('./features/admin/Runtimes'));
const Research = lazy(() => import('./features/admin/Research'));
const Agent = lazy(() => import('./pages/Agent'));
const Readiness = lazy(() => import('./features/readiness/ReadinessRoute'));
const Gold = lazy(() => import('./pages/Gold'));

/** An old address → its new one, keeping the query string (lenses survive the move). */
function moved(to: string | ((params: Record<string, string | undefined>) => string)) {
  return ({ request, params }: LoaderFunctionArgs) => {
    const target = typeof to === 'string' ? to : to(params);
    return redirect(`${target}${new URL(request.url).search}`);
  };
}

function RouteError() {
  const error = useRouteError();
  const status = isRouteErrorResponse(error) ? error.status : null;
  const message = isRouteErrorResponse(error) ? error.statusText : error instanceof Error ? error.message : '';
  return (
    <main>
      <p className="label">{status ? `error ${status}` : 'error'}</p>
      <h1>This page failed to load</h1>
      {message && <p className="mono wrap-any">{message}</p>}
      <p>
        <Link to="/">Back to the overview</Link>
      </p>
    </main>
  );
}

function NotFound() {
  return (
    <>
      <p className="label">404</p>
      <h1>No such page</h1>
      <p>
        Every page is one of the eight tabs above. <Link to="/">Back to the overview</Link>.
      </p>
    </>
  );
}

export const routes: RouteObject[] = [
  {
    id: 'root',
    path: '/',
    element: <Root />,
    errorElement: <RouteError />,
    children: [
      { id: 'overview', index: true, element: <LandingRoute /> },
      { id: 'filings', path: 'filings', element: <Filings /> },
      { id: 'scoring', path: 'scoring', element: <Scoring /> },
      { id: 'leaderboard', path: 'leaderboard', element: <Leaderboard /> },
      {
        id: 'runs',
        path: 'runs',
        element: <SlotLayout />,
        children: [
          // Every test pass leads (s21); a link carrying the legacy page's lens goes there.
          { id: 'runs-passes', index: true, loader: legacyLensLoader, element: <RunsPage /> },
          { id: 'runs-traces', path: 'traces', element: <Traces /> },
          { id: 'runs-trace', path: 'traces/:traceId', element: <TraceDetail /> },
          { id: 'runs-legacy', path: 'legacy', element: <Evaluations /> },
        ],
      },
      {
        id: 'optimise',
        path: 'optimise',
        element: <SlotLayout />,
        children: [
          // The whole story leads (s21); the pipeline's campaigns moved one level down.
          { id: 'optimise-story', index: true, element: <StoryPage /> },
          { id: 'optimise-campaigns', path: 'campaigns', element: <Experiments /> },
          { id: 'optimise-runtimes', path: 'runtimes', element: <Runtimes /> },
          { id: 'optimise-research', path: 'research', element: <Research /> },
        ],
      },
      { id: 'agent-chat', path: 'agent/chat/*', handle: { wide: true }, element: <ChatRoute /> },
      {
        id: 'agent',
        path: 'agent',
        element: <SlotLayout />,
        children: [
          { id: 'agent-architecture', index: true, element: <Agent /> },
          { id: 'agent-readiness', path: 'readiness', element: <Readiness /> },
        ],
      },
      { id: 'gold', path: 'gold', element: <Gold /> },

      // ── every address the Console published, redirected ───────────────────────────────
      { id: 'old-welcome', path: 'welcome', loader: moved('/') },
      { id: 'old-chat', path: 'chat/*', loader: moved((p) => `/agent/chat${p['*'] ? `/${p['*']}` : ''}`) },
      { id: 'old-debrief', path: 'debrief', loader: moved('/agent') },
      {
        id: 'old-admin',
        path: 'admin',
        children: [
          { id: 'old-admin-index', index: true, loader: moved('/optimise') },
          { id: 'old-admin-system', path: 'system', loader: moved('/agent') },
          { id: 'old-admin-readiness', path: 'readiness', loader: moved('/agent/readiness') },
          { id: 'old-admin-runtimes', path: 'runtimes', loader: moved('/optimise/runtimes') },
          { id: 'old-admin-experiments', path: 'experiments', loader: moved('/optimise/campaigns') },
          { id: 'old-admin-campaigns', path: 'campaigns', loader: moved('/optimise/campaigns') },
          { id: 'old-admin-research', path: 'research', loader: moved('/optimise/research') },
          { id: 'old-admin-evaluations', path: 'evaluations', loader: moved('/runs/legacy') },
          { id: 'old-admin-dataset', path: 'dataset', loader: moved('/filings') },
          { id: 'old-admin-traces', path: 'traces', loader: moved('/runs/traces') },
          {
            id: 'old-admin-trace',
            path: 'traces/:traceId',
            loader: moved((p) => `/runs/traces/${encodeURIComponent(p.traceId ?? '')}`),
          },
        ],
      },
      { id: 'not-found', path: '*', element: <NotFound /> },
    ],
  },
];
