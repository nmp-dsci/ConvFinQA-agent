import { defineConfig } from 'vitest/config';

// Scoped to `src` deliberately: `tests/e2e` holds Playwright specs, which use a
// different runner and would fail on collection here.
export default defineConfig({
  test: {
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    environment: 'node',
  },
});
