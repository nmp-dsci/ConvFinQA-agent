import { defineConfig, devices } from '@playwright/test';

const PORT_BACKEND = process.env.PW_BACKEND_PORT ?? '8765';
const PORT_FRONTEND = process.env.PW_FRONTEND_PORT ?? '4173';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  retries: 0,
  workers: 1,
  timeout: 240_000,
  expect: { timeout: 30_000 },
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT_FRONTEND}`,
    actionTimeout: 30_000,
    navigationTimeout: 30_000,
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: `cd .. && uv run python cli.py serve --port ${PORT_BACKEND}`,
      url: `http://127.0.0.1:${PORT_BACKEND}/healthz`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
    {
      command: `npm run build && npm run preview -- --host 127.0.0.1 --port ${PORT_FRONTEND}`,
      url: `http://127.0.0.1:${PORT_FRONTEND}/`,
      reuseExistingServer: !process.env.CI,
      timeout: 240_000,
      env: { VITE_API_PROXY: `http://127.0.0.1:${PORT_BACKEND}` },
    },
  ],
});
