import { expect, test } from '@playwright/test';

const BACKEND = process.env.PW_BACKEND_URL ?? 'http://127.0.0.1:8765';

test('new user lands on the welcome quick-start', async ({ page }) => {
  await page.addInitScript((apiBase: string) => {
    window.localStorage.clear();
    window.localStorage.setItem('convfinqa.apiBase', apiBase);
  }, BACKEND);
  await page.goto('/');

  const landing = page.getByTestId('landing-screen');
  await expect(landing).toBeVisible();
  await expect(landing).toHaveAttribute('data-variant', 'new');
  await expect(landing).toContainText(/pick a report/i);
  await expect(landing).toContainText(/ask a question/i);
  await expect(landing).toContainText(/run all gold/i);

  await expect(page.getByText(/your conversations will appear here/i)).toBeVisible();

  await page.getByTestId('landing-cta').click();
  await expect(page.getByTestId('report-picker-input')).toBeVisible();
});

test('returning user with no active selection sees compact variant', async ({ page }) => {
  await page.addInitScript((apiBase: string) => {
    window.localStorage.setItem('convfinqa.apiBase', apiBase);
    const now = Date.now();
    window.localStorage.setItem(
      'convfinqa.v1',
      JSON.stringify({
        state: {
          activeReportId: null,
          conversations: {
            'Single_VLO/2011/page_126.pdf-1': {
              reportId: 'Single_VLO/2011/page_126.pdf-1',
              sessionId: null,
              messages: [],
              lastUsedAt: now,
              lastReadAt: now,
              unreadCount: 0,
              isStreaming: false,
            },
          },
          reports: [],
        },
        version: 1,
      })
    );
  }, BACKEND);
  await page.goto('/');

  const landing = page.getByTestId('landing-screen');
  await expect(landing).toBeVisible();
  await expect(landing).toHaveAttribute('data-variant', 'returning');

  await expect(page.getByText('Single_VLO/2011/page_126.pdf-1').first()).toBeVisible();
});
