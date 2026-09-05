import { lazy } from 'react';
import {
  createBrowserRouter,
  isRouteErrorResponse,
  Link,
  Navigate,
  useRouteError,
} from 'react-router-dom';
import { ChatRoute } from '../features/chat/ChatRoute';
import { LandingRoute } from '../features/landing/LandingRoute';
import { Shell } from './Shell';

// Admin and system are lazy so the first paint — landing or chat — never pays
// for pages most visitors never open. `Shell` renders the <Suspense> boundary
// these resolve into.
const AdminOverview = lazy(() => import('../features/admin/AdminOverview'));
const Evaluations = lazy(() => import('../features/admin/Evaluations'));
const Experiments = lazy(() => import('../features/admin/Experiments'));
const Campaigns = lazy(() => import('../features/admin/Campaigns'));
const Traces = lazy(() => import('../features/admin/Traces'));
const TraceDetail = lazy(() => import('../features/admin/TraceDetail'));
const Research = lazy(() => import('../features/admin/Research'));
const DatasetReview = lazy(() => import('../features/admin/DatasetReview'));
const SystemRoute = lazy(() => import('../features/system/SystemRoute'));

function RouteError() {
  const error = useRouteError();
  const status = isRouteErrorResponse(error) ? error.status : null;
  const message =
    isRouteErrorResponse(error) ? error.statusText : error instanceof Error ? error.message : '';

  return (
    <div className="grid h-full place-items-center p-8">
      <div className="max-w-md text-center">
        <div className="mono-caps mb-2">{status ? `error ${status}` : 'error'}</div>
        <h1 className="mb-2 text-lg font-medium text-text">This page failed to load</h1>
        {message && <p className="mb-4 font-mono text-xs break-words text-muted">{message}</p>}
        <Link to="/" className="text-sm text-amber underline underline-offset-4">
          Back to the console
        </Link>
      </div>
    </div>
  );
}

function NotFound() {
  return (
    <div className="grid h-full place-items-center p-8">
      <div className="text-center">
        <div className="mono-caps mb-2">404</div>
        <h1 className="mb-2 text-lg font-medium text-text">No such page</h1>
        <Link to="/" className="text-sm text-amber underline underline-offset-4">
          Back to the console
        </Link>
      </div>
    </div>
  );
}

/**
 * The route table.
 *
 * Two notes on shapes that differ from the plan, both forced by the data:
 *
 *  - The chat conversation route is a splat (`chat/*`), not `chat/:reportId`.
 *    Report ids contain slashes — `Single_VLO/2011/page_126.pdf-1` — so a
 *    single dynamic segment would only ever match the first third of one.
 *  - `/` is the landing board and chat lives at `/chat` only. Phase 0 had it
 *    the other way round to keep the Playwright specs green, but four of the
 *    six were already red at HEAD for the same reason (the old `App.tsx` gated
 *    `/` behind a sessionStorage "entered" flag while every spec opened `/`
 *    and immediately reached for conversation controls). A public demo whose
 *    front door is the product's own chat window has no front door, so the
 *    board takes `/` and Phase 6 updates the specs to enter through the
 *    landing CTA — which is what a visitor does, and why `landing-enter` and
 *    `landing-cta` exist as test ids at all.
 *  - `/welcome` is kept as a redirect rather than deleted: it shipped in
 *    Phase 0 and any link already pointing at it should land on the board.
 */
export const router = createBrowserRouter([
  {
    path: '/',
    element: <Shell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <LandingRoute /> },
      { path: 'welcome', element: <Navigate to="/" replace /> },
      { path: 'chat', element: <ChatRoute /> },
      { path: 'chat/*', element: <ChatRoute /> },
      {
        path: 'admin',
        children: [
          { index: true, element: <AdminOverview /> },
          { path: 'evaluations', element: <Evaluations /> },
          { path: 'dataset', element: <DatasetReview /> },
          { path: 'experiments', element: <Experiments /> },
          { path: 'campaigns', element: <Campaigns /> },
          { path: 'traces', element: <Traces /> },
          { path: 'traces/:traceId', element: <TraceDetail /> },
          { path: 'research', element: <Research /> },
          { path: 'system', element: <SystemRoute /> },
        ],
      },
      { path: 'debrief', element: <Navigate to="/admin/system" replace /> },
      { path: '*', element: <NotFound /> },
    ],
  },
]);
