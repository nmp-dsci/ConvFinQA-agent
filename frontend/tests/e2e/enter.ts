import { expect, type Page } from '@playwright/test';

/** Where the browser should point its API calls. */
export const BACKEND = process.env.PW_BACKEND_URL ?? 'http://127.0.0.1:8765';

/**
 * Walk in through the front door.
 *
 * `/` is the status board, not the chat. Every spec that needs a conversation
 * therefore has to do what a visitor does: land on the board, then take one of
 * the two ways in. That is the entire reason `landing-enter` and `landing-cta`
 * exist as pinned ids, and pretending `/` is the chat — by deep-linking past
 * the board — would test a route no visitor ever arrives on.
 *
 * Nothing downstream of this changes. The four streaming specs assert exactly
 * what they asserted before; only the first two lines of each differ.
 */
export async function enterChatFromBoard(page: Page): Promise<void> {
  await page.goto('/');

  // The board is what `/` serves now — assert it, so a regression that put
  // chat back at the root fails here with a clear message rather than
  // succeeding by accident.
  await expect(page.getByTestId('landing-board')).toBeVisible();

  await page.getByTestId('landing-enter').click();

  await expect(page).toHaveURL(/\/chat$/);
  // The chat's own empty state. Waiting for it means the store has mounted and
  // the controls the caller is about to click are attached.
  await expect(page.getByTestId('landing-screen')).toBeVisible();
}
