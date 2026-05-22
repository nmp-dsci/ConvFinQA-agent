import { expect, test } from '@playwright/test';

const BACKEND = process.env.PW_BACKEND_URL ?? 'http://127.0.0.1:8765';
const RID_A = 'Single_VLO/2011/page_126.pdf-1';
const RID_B = 'Single_AES/2003/page_168.pdf-1';

test('unread badge appears on inactive conversation, clears on click', async ({ page }) => {
  await page.addInitScript((apiBase: string) => {
    window.localStorage.setItem('convfinqa.apiBase', apiBase);
  }, BACKEND);
  await page.goto('/');

  // Open A and ask a question, wait for first answer.
  await page.getByTestId('topbar-change-report').click();
  await page.getByTestId('report-picker-input').fill(RID_A);
  await page.getByRole('button', { name: RID_A }).click();

  await page
    .getByTestId('composer-input')
    .fill('what were the value of futures 2013 long?');
  await page.getByTestId('composer-send').click();
  await expect(page.locator('[data-role="assistant-message"][data-final="true"]').first())
    .toBeVisible({ timeout: 90_000 });

  // Open B and start a question.
  await page.getByTestId('topbar-change-report').click();
  await page.getByTestId('report-picker-input').fill(RID_B);
  await page.getByRole('button', { name: RID_B }).click();
  await page
    .getByTestId('composer-input')
    .fill('what was the value of physical contracts 2013 long?');
  await page.getByTestId('composer-send').click();

  // Switch to A while B is mid-stream.
  await page.locator('[data-testid="sidebar-row"][data-rid="' + RID_A + '"]').click();

  // B's row should accumulate an unread badge as its stream finishes.
  const bRow = page.locator(`[data-testid="sidebar-row"][data-rid="${RID_B}"]`);
  await expect(bRow.getByTestId('unread-badge')).toBeVisible({ timeout: 120_000 });

  // Click B → badge clears.
  await bRow.click();
  await expect(bRow.getByTestId('unread-badge')).toHaveCount(0);
});
