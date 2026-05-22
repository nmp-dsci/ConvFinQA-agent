import { expect, test } from '@playwright/test';

const BACKEND = process.env.PW_BACKEND_URL ?? 'http://127.0.0.1:8765';
const REPORT_A = 'Single_VLO/2011/page_126.pdf-1';
const REPORT_B = 'Single_AES/2003/page_168.pdf-1';

test('runs gold questions for two reports concurrently', async ({ browser }) => {
  const ctxA = await browser.newContext();
  const ctxB = await browser.newContext();
  const a = await ctxA.newPage();
  const b = await ctxB.newPage();

  for (const p of [a, b]) {
    await p.addInitScript((apiBase: string) => {
      window.localStorage.setItem('convfinqa.apiBase', apiBase);
    }, BACKEND);
  }

  await a.goto('/');
  await b.goto('/');

  for (const [page, rid] of [
    [a, REPORT_A],
    [b, REPORT_B],
  ] as const) {
    await page.getByTestId('topbar-change-report').click();
    await page.getByTestId('report-picker-input').fill(rid);
    await page.getByRole('button', { name: rid }).click();
  }

  await Promise.all([
    a.getByTestId('composer-run-all').click(),
    b.getByTestId('composer-run-all').click(),
  ]);

  // Both pages should have at least one assistant bubble within 30s,
  // proving streams are interleaving on the wire.
  await expect
    .poll(
      async () => {
        const ca = await a.locator('[data-role="assistant-message"]').count();
        const cb = await b.locator('[data-role="assistant-message"]').count();
        return ca >= 1 && cb >= 1;
      },
      { timeout: 30_000 }
    )
    .toBeTruthy();

  // Wait for both to fully complete (at least 2 finalized assistant bubbles each).
  for (const page of [a, b]) {
    await expect
      .poll(
        async () =>
          (await page.locator('[data-role="assistant-message"][data-final="true"]').count()) >= 2,
        { timeout: 240_000 }
      )
      .toBeTruthy();
  }

  // Assert ≥80% gold-match on each conversation.
  for (const [page, label] of [
    [a, 'A'],
    [b, 'B'],
  ] as const) {
    const total = await page.locator('[data-gold]').count();
    const matches = await page.locator('[data-gold="match"]').count();
    expect(total, `${label} should have at least one gold-marked answer`).toBeGreaterThan(0);
    expect(matches / total, `${label} accuracy`).toBeGreaterThanOrEqual(0.8);
  }

  await ctxA.close();
  await ctxB.close();
});
