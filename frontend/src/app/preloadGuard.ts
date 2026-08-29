const RELOADED_KEY = 'convfinqa.chunkReloadAt';
const COOLDOWN_MS = 30_000;

/**
 * Recover from a stale lazy chunk exactly once.
 *
 * After a deploy the index HTML in an open tab still points at hashed chunk
 * files that no longer exist, so the first navigation into a lazily-loaded
 * admin route fails to import — a blank screen for a user who did nothing
 * wrong. Vite fires `vite:preloadError` for precisely this, and a reload picks
 * up the new index.
 *
 * The cooldown is the important half. An unconditional reload turns a genuine
 * network failure into an infinite refresh loop, which is a far worse incident
 * than the one being fixed; a second failure within the window is left to the
 * route's error boundary to report honestly instead.
 */
export function installPreloadErrorGuard(): void {
  window.addEventListener('vite:preloadError', (event) => {
    let last = 0;
    try {
      last = Number(window.sessionStorage.getItem(RELOADED_KEY) ?? 0);
    } catch {
      // No session storage: treat as "never reloaded" and allow the one shot.
    }

    if (Date.now() - last < COOLDOWN_MS) return;

    try {
      window.sessionStorage.setItem(RELOADED_KEY, String(Date.now()));
    } catch {
      // Without storage we cannot guarantee "once", so do not reload at all —
      // an error boundary is better than a loop.
      return;
    }

    event.preventDefault();
    window.location.reload();
  });
}
