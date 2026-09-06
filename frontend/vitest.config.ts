import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vitest/config';

// Scoped to `src` deliberately: `tests/e2e` holds Playwright specs, which use a
// different runner and would fail on collection here.
//
// The `@` alias is repeated from `vite.config.ts` rather than shared, because
// this config must not pull in the app's plugins: component tests here render
// through `react-dom/server` in a Node environment, with no DOM and no HMR. A
// component that imports `@/lib/utils` would otherwise fail to resolve under the
// test runner while building perfectly well.
export default defineConfig({
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  test: {
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    environment: 'node',
  },
});
