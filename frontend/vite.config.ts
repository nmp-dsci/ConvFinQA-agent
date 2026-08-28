import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const API_BASE = process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8765';

// Every backend path prefix must be listed here or the dev server silently
// answers with the SPA's index.html and the request fails as an HTML-404.
// In production this proxy does not exist at all — FastAPI serves the built
// SPA from the same origin, so there is nothing to keep in sync there.
const BACKEND_PREFIXES = [
  '/healthz',
  '/reports',
  '/sessions',
  '/eval',
  '/admin',
  '/traces',
  '/demo',
];

const proxy = Object.fromEntries(BACKEND_PREFIXES.map((p) => [p, API_BASE]));

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
});
