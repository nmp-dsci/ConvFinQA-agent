import { fileURLToPath, URL } from 'node:url';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const API_BASE = process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8765';

// Every backend path prefix must be listed here or the dev server silently
// answers with the SPA's index.html and the request fails as an HTML-404.
// In production this proxy does not exist at all — FastAPI serves the built
// SPA from the same origin, so there is nothing to keep in sync there.
//
// `/metrics` is listed ahead of the route existing: Phase 1 adds
// `GET /metrics/production`, and a prefix added to the backend without a
// matching entry here is exactly the silent failure this list exists to
// prevent. Until that route lands the proxy simply forwards a 404 as JSON,
// which is what the client already tolerates.
const BACKEND_PREFIXES = [
  '/healthz',
  '/reports',
  '/sessions',
  '/eval',
  '/admin',
  '/traces',
  '/demo',
  '/metrics',
];

// `/admin` is both an API prefix and a UI route prefix, so in dev a browser
// navigating to /admin/evaluations would be proxied to FastAPI, match no admin
// route, fall through to its SPA catch-all and come back as the *built*
// index.html — which references dist asset hashes the dev server does not
// serve. The result is a blank page and a handful of 404s, with nothing
// obviously wrong in either log.
//
// A document request is never an API call, so hand those back to Vite and let
// the client router resolve them. `fetch()` sends `*/*` and the SSE client
// sends `text/event-stream`, so no real API request is caught by this.
function bypassDocumentRequests(req: { headers: Record<string, string | string[] | undefined> }) {
  const dest = req.headers['sec-fetch-dest'];
  const accept = req.headers.accept;
  const isDocument =
    dest === 'document' || (typeof accept === 'string' && accept.includes('text/html'));
  return isDocument ? '/index.html' : undefined;
}

const proxy = Object.fromEntries(
  BACKEND_PREFIXES.map((p) => [p, { target: API_BASE, changeOrigin: false, bypass: bypassDocumentRequests }]),
);

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
});
