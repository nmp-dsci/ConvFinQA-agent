import { expect, test } from '@playwright/test';

const BACKEND = process.env.PW_BACKEND_URL ?? 'http://127.0.0.1:8765';
const RID = 'Single_VLO/2011/page_126.pdf-1';

test('reset conversation drops server-side history', async ({ page, request }) => {
  await page.addInitScript((apiBase: string) => {
    window.localStorage.setItem('convfinqa.apiBase', apiBase);
  }, BACKEND);
  await page.goto('/');

  await page.getByTestId('topbar-change-report').click();
  await page.getByTestId('report-picker-input').fill(RID);
  await page.getByRole('button', { name: RID }).click();

  await page
    .getByTestId('composer-input')
    .fill('what were the value of futures 2013 long?');
  await page.getByTestId('composer-send').click();

  await expect(page.locator('[data-role="assistant-message"][data-final="true"]').first())
    .toBeVisible({ timeout: 90_000 });

  const sidBefore = await page.evaluate(
    ([rid]: [string]) => {
      const raw = window.localStorage.getItem('convfinqa.v1');
      const parsed = raw ? JSON.parse(raw) : null;
      const conv = parsed?.state?.conversations?.[rid];
      return (conv?.sessionId ?? null) as string | null;
    },
    [RID] as [string]
  );
  expect(sidBefore).not.toBeNull();

  const before = await request.get(`${BACKEND}/sessions/${sidBefore}`);
  expect((await before.json()).n_turns).toBeGreaterThan(0);

  page.once('dialog', (d) => d.accept());
  await page.getByTestId('reset-conversation').click();

  await expect(page.locator('[data-role="assistant-message"]')).toHaveCount(0);

  const afterDelete = await request.get(`${BACKEND}/sessions/${sidBefore}`);
  expect(afterDelete.status()).toBe(404);

  await page.getByTestId('composer-input').fill('what is the sum?');
  await page.getByTestId('composer-send').click();
  await expect(page.locator('[data-role="assistant-message"][data-final="true"]').last())
    .toBeVisible({ timeout: 90_000 });

  const sidAfter = await page.evaluate(
    ([rid]: [string]) => {
      const raw = window.localStorage.getItem('convfinqa.v1');
      const parsed = raw ? JSON.parse(raw) : null;
      const conv = parsed?.state?.conversations?.[rid];
      return (conv?.sessionId ?? null) as string | null;
    },
    [RID] as [string]
  );
  expect(sidAfter).not.toBe(sidBefore);

  const after = await request.get(`${BACKEND}/sessions/${sidAfter}`);
  const body = await after.json();
  expect(body.n_turns).toBe(1);
  expect(body.history[0].question).toBe('what is the sum?');
});
