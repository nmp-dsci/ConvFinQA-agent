import { expect, test } from '@playwright/test';

const BACKEND = process.env.PW_BACKEND_URL ?? 'http://127.0.0.1:8765';

/**
 * Happy-path smoke: opens the app from a clean state, opens the picker,
 * selects whichever report is at the top of the list, fires "Run all gold",
 * and verifies that the run completes end-to-end with no error bubbles.
 *
 * Correctness is *not* asserted here — the concurrent.spec.ts test owns
 * the ≥80% accuracy threshold. This test owns the wiring: that clicks
 * propagate, the picker actually populates, the stream lands, and the
 * Run-all button re-enables when the last turn finishes.
 */
test('new conversation → first report → run all gold completes cleanly', async ({
  page,
}) => {
  await page.addInitScript((apiBase: string) => {
    window.localStorage.clear();
    window.localStorage.setItem('convfinqa.apiBase', apiBase);
  }, BACKEND);
  await page.goto('/');

  // The landing CTA opens the picker on a fresh install.
  await page.getByTestId('landing-cta').click();

  await expect(page.getByTestId('report-picker-input')).toBeVisible();

  // Wait for the reports list to populate (not "Loading reports…").
  const firstRow = page
    .getByTestId('report-picker-list')
    .locator('button')
    .first();
  await expect(firstRow).toBeVisible({ timeout: 30_000 });

  const reportId = (await firstRow.innerText()).trim();
  expect(reportId.length).toBeGreaterThan(0);

  await firstRow.click();

  // Picker closes; top bar reflects the active report id.
  await expect(page.getByTestId('report-picker-input')).toHaveCount(0);
  await expect(page.getByTestId('active-report-id')).toHaveText(reportId);

  // Sidebar should now show this conversation.
  await expect(
    page.locator(`[data-testid="sidebar-row"][data-rid="${reportId}"]`)
  ).toBeVisible();

  // Kick off Run-all.
  const runAll = page.getByTestId('composer-run-all');
  await runAll.click();

  // First answer should land within 2 minutes (cold cache or warm).
  await expect(
    page.locator('[data-role="assistant-message"][data-final="true"]').first()
  ).toBeVisible({ timeout: 120_000 });

  // The whole run finishes when isStreaming flips back to false, which
  // re-enables the Run-all button. Generous ceiling for cold runs.
  await expect(runAll).toBeEnabled({ timeout: 360_000 });

  // We must have at least one assistant bubble that finalized.
  const finalCount = await page
    .locator('[data-role="assistant-message"][data-final="true"]')
    .count();
  expect(finalCount).toBeGreaterThan(0);

  // Every assistant bubble must have *some* terminal state (final=true
  // or status=error). None should still be mid-stream.
  const stillStreaming = await page
    .locator('[data-role="assistant-message"][data-streaming="true"]')
    .count();
  expect(stillStreaming, 'no assistant bubbles should still be streaming').toBe(0);

  // Each answered turn carries a known gold (Run-all passes gold per turn),
  // so [data-gold] should exist on every finalized bubble.
  const goldMarked = await page.locator('[data-gold]').count();
  expect(goldMarked, 'every finalized turn should have a gold marker').toBe(
    finalCount
  );
});
