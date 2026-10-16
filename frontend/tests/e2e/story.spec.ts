import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { BACKEND } from './enter';

/**
 * Optimise › Story and Runs › Every test pass (s21): the whole record on the gate split.
 *
 * Nothing here is a fixture count. The chart must draw exactly the complete passes and the ledger
 * must list exactly the passes `/eval/campaigns` serves — read from the same response the pages
 * read — so the spec holds as cycles add passes, and fails the day a page drops one.
 */

interface TimelineBody {
  test_timeline?: { run_name: string; complete: boolean }[];
  train_passes?: unknown[];
}

async function timeline(request: APIRequestContext): Promise<Required<TimelineBody>> {
  const response = await request.get(`${BACKEND}/eval/campaigns`);
  expect(response.ok(), '/eval/campaigns must answer before the story can be judged').toBeTruthy();
  const body = (await response.json()) as TimelineBody;
  return { test_timeline: body.test_timeline ?? [], train_passes: body.train_passes ?? [] };
}

async function open(page: Page, path: string): Promise<void> {
  await page.addInitScript((apiBase: string) => {
    window.localStorage.setItem('convfinqa.apiBase', apiBase);
  }, BACKEND);
  await page.goto(path);
}

test.describe('the story on the gate split', () => {
  test('/optimise draws every complete pass and lists every pass', async ({ page, request }) => {
    const { test_timeline } = await timeline(request);
    const complete = test_timeline.filter((p) => p.complete).length;
    await open(page, '/optimise');
    await expect(page.getByTestId('optimise-story')).toBeVisible();
    const chart = page.getByTestId('story-figure').getByTestId('timeline-chart');
    if (complete === 0) {
      await expect(page.getByTestId('timeline-empty')).toBeVisible();
      return;
    }
    await expect(chart).toBeVisible();
    await expect(chart.getByTestId('timeline-point')).toHaveCount(complete);
    await expect(chart.getByTestId('champion-line')).toBeVisible();
    await expect(page.getByTestId('story-ledger').getByTestId('pass-row')).toHaveCount(test_timeline.length);
    // The campaigns moved one level down and still land.
    await page.goto('/optimise/campaigns');
    await expect(page).toHaveURL(/\/optimise\/campaigns$/);
  });

  test('/runs lists every test pass, and train only behind the toggle', async ({ page, request }) => {
    const { test_timeline, train_passes } = await timeline(request);
    await open(page, '/runs');
    await expect(page.getByTestId('runs-passes')).toBeVisible();
    const ledger = page.getByTestId('runs-ledger');
    await expect(ledger.getByTestId('pass-row')).toHaveCount(test_timeline.length);
    await expect(ledger.getByTestId('train-row')).toHaveCount(0);
    const toggle = page.getByTestId('train-toggle');
    await expect(toggle).toContainText('train: writes versions, never promotes them');
    // The toggle is the URL (`?train=1`): its state lands after the navigation, so wait for it
    // rather than asserting it the instant the click returns.
    await toggle.getByRole('checkbox').click();
    await expect(page).toHaveURL(/[?&]train=1/);
    await expect(toggle.getByRole('checkbox')).toBeChecked();
    await expect(ledger.getByTestId('train-row')).toHaveCount(train_passes.length);
    await expect(ledger.getByTestId('pass-row')).toHaveCount(test_timeline.length);
  });

  test('a link carrying the legacy lens still lands on the legacy page', async ({ page }) => {
    await open(page, '/runs?version=v2');
    await expect(page).toHaveURL(/\/runs\/legacy\?version=v2$/);
    await expect(page.getByTestId('admin-evaluations')).toBeVisible();
  });
});
