import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { BACKEND, enterChatFromBoard } from './enter';

/**
 * The status board at `/`, and the chat empty state behind it.
 *
 * The old version of this file described a marketing landing page that no
 * longer exists: `/` is now the board, and the quick-start copy it used to
 * assert moved into the chat's own empty state. Nothing here was dropped to
 * make the suite green — the second describe block below re-asserts every
 * original assertion at the place the copy now lives.
 *
 * The load-bearing test in this file is `null renders as an em dash`. That is
 * the project's honesty rule expressed as a gate: `/metrics/production` returns
 * `null` (not `0`) with `n_measured: 0` for latency, tokens and cost until
 * someone pays for a metered eval run, and a tile that quietly prints `0` for
 * an unmeasured figure is a lie in the flattering direction. This test fails
 * the build if that ever regresses.
 */

const SEED_RID = 'Single_VLO/2011/page_126.pdf-1';

/** The six tiles, by the testid `HudTile` derives from each label. */
const TILES = [
  'hud-tile-gate-accuracy',
  'hud-tile-out-of-sample-accuracy',
  'hud-tile-p50-latency',
  'hud-tile-cost-per-turn',
  'hud-tile-turns-served',
  'hud-tile-error-rate',
] as const;

const NO_VALUE = '—';

async function openBoard(page: Page): Promise<void> {
  await page.addInitScript((apiBase: string) => {
    window.localStorage.clear();
    window.localStorage.setItem('convfinqa.apiBase', apiBase);
  }, BACKEND);
  await page.goto('/');
  await expect(page.getByTestId('landing-board')).toBeVisible();
}

/** What the backend says it is, so the board's claims can be checked against it. */
async function health(request: APIRequestContext): Promise<{
  mode: string;
  champion: string | null;
}> {
  const response = await request.get(`${BACKEND}/healthz`);
  expect(response.ok(), '/healthz must answer before the board can be judged').toBeTruthy();
  return response.json();
}

// ---------------------------------------------------------------------------
// The board
// ---------------------------------------------------------------------------

test.describe('status board at /', () => {
  test('shows the three lamps, six HUD tiles and three recorded conversations', async ({
    page,
    request,
  }) => {
    const { champion } = await health(request);
    await openBoard(page);

    // --- lamps ------------------------------------------------------------
    for (const label of ['mode', 'champion', 'gate']) {
      await expect(page.getByTestId(`lamp-${label}`)).toBeVisible();
      // State is a shape as well as a colour, so every lamp must declare both.
      await expect(page.getByTestId(`lamp-${label}`)).toHaveAttribute(
        'data-shape',
        /^(solid|dashed)$/
      );
      await expect(page.getByTestId(`lamp-${label}`)).toHaveAttribute(
        'data-tone',
        /^(good|amber|bad|info|idle)$/
      );
    }

    // The champion lamp reports what /healthz reports — not a hardcoded name.
    if (champion) {
      await expect(page.getByTestId('lamp-champion')).toContainText(champion);
    }

    // --- six tiles, each a link with somewhere to go ----------------------
    for (const testId of TILES) {
      const tile = page.getByTestId(testId);
      await expect(tile, `${testId} should be on the board`).toBeVisible();
      // "A metric a reader cannot drill into is decoration."
      await expect(tile).toHaveAttribute('href', /^\/admin\//);
    }

    // --- three recorded conversations, each a door into the chat ----------
    const cards = page.getByTestId('recorded-conversation');
    await expect(cards).toHaveCount(3);
    for (let i = 0; i < 3; i++) {
      const rid = await cards.nth(i).getAttribute('data-rid');
      expect(rid, 'each card names the filing it opens').toBeTruthy();
      await expect(cards.nth(i)).toHaveAttribute('href', `/chat/${rid}`);
    }
  });

  test('a metric with no measurement renders an em dash and a reason, never a zero', async ({
    page,
    request,
  }) => {
    // Read the same numbers the board reads, then hold the board to them. This
    // is deliberately not a fixture: it asserts the *rule* — null out, em dash
    // in; measured out, no em dash in — and so it is true of a cold container,
    // of a dev process with traffic, and of the demo deployment alike.
    const { mode } = await health(request);
    const source = mode === 'demo' ? 'demo' : 'serving';
    const metrics = await request
      .get(`${BACKEND}/metrics/production`)
      .then((r) => r.json())
      .then((body) => body.sources[source]);

    const expected: Record<string, number | null> = {
      'hud-tile-p50-latency': metrics.latency_ms.p50,
      'hud-tile-cost-per-turn': metrics.cost_usd.per_turn,
      'hud-tile-error-rate': metrics.errors.error_rate,
    };

    await openBoard(page);

    for (const [testId, value] of Object.entries(expected)) {
      const tile = page.getByTestId(testId);
      const rendered = tile.getByTestId('hud-value');
      await expect(rendered).toBeVisible();

      if (value === null) {
        await expect(tile).toHaveAttribute('data-absent', 'true');
        await expect(rendered, `${testId} is unmeasured and must render an em dash`).toHaveText(
          NO_VALUE
        );
        // …and say why. An empty tile that explains nothing is worse than none.
        const reason = ((await tile.getByTestId('hud-reason').textContent()) ?? '').trim();
        expect(reason.length, `${testId} must state why it is empty`).toBeGreaterThan(0);
        // The whole point: an unmeasured figure must never present as zero.
        const text = ((await rendered.textContent()) ?? '').trim();
        expect(text, `${testId} must not render a null as a zero`).not.toMatch(
          /^[$]?0([.,]0+)?%?$/
        );
      } else {
        await expect(tile).toHaveAttribute('data-absent', 'false');
        await expect(
          rendered,
          `${testId} was measured and must show the figure`
        ).not.toHaveText(NO_VALUE);
      }
    }

    // Gate accuracy comes from committed CSVs, so it is measured on every
    // deployment. Out-of-sample is deliberately unmeasured mid-campaign — the
    // holdout stays sealed until a release opens it — so it renders an em
    // dash with a reason, same as any other unmeasured tile, and the two
    // populations stay two tiles, never an average.
    await expect(
      page.getByTestId('hud-tile-gate-accuracy').getByTestId('hud-value')
    ).not.toHaveText(NO_VALUE);
    await expect(page.getByTestId('hud-tile-gate-accuracy')).toHaveAttribute(
      'data-absent',
      'false'
    );

    const outOfSample = page.getByTestId('hud-tile-out-of-sample-accuracy');
    await expect(outOfSample.getByTestId('hud-value')).toHaveText(NO_VALUE);
    await expect(outOfSample).toHaveAttribute('data-absent', 'true');
  });

  test('an unmeasured metric renders an em dash, not a zero, even with turns served', async ({
    page,
    request,
  }) => {
    // The test above checks the board against whatever the deployment happens
    // to hold, which on a machine with development traffic means the *measured*
    // branch. This one pins the branch that matters: turns were served, so the
    // count is a real `27`, but nothing was timed or priced. A tile that reads
    // `0ms` / `$0.0000` here would be claiming a free, instant system.
    //
    // This is the exact shape `/metrics/production` returns today — the demo
    // pack carries `metrics: {}` on all 174 stage frames — so it is a recorded
    // reality, not a hypothetical.
    const { mode } = await health(request);
    const source = mode === 'demo' ? 'demo' : 'serving';
    const live = await request.get(`${BACKEND}/metrics/production`).then((r) => r.json());

    const blanked = structuredClone(live);
    const group = blanked.sources[source];
    group.n_turns = 27;
    group.latency_ms = { p50: null, p95: null, mean: null, n_measured: 0 };
    group.tokens_per_turn = { p50: null, mean: null, total: 0, n_measured: 0 };
    group.cost_usd = { per_turn: null, total: 0, n_measured: 0 };
    group.errors = { ...group.errors, n_errors: 0, error_rate: null };
    group.series = group.series.map((bucket: Record<string, unknown>) => ({
      ...bucket,
      p50_latency_ms: null,
      cost_usd: 0,
    }));

    await page.route('**/metrics/production*', async (route) => {
      await route.fulfill({ json: blanked });
    });

    await openBoard(page);

    for (const testId of ['hud-tile-p50-latency', 'hud-tile-cost-per-turn', 'hud-tile-error-rate']) {
      const tile = page.getByTestId(testId);
      await expect(tile).toHaveAttribute('data-absent', 'true');
      await expect(tile.getByTestId('hud-value')).toHaveText(NO_VALUE);
      await expect(tile.getByTestId('hud-reason')).toContainText(/measur|no turns|returned nothing/i);
    }

    // A measured zero is a different fact and still prints as a number: 27
    // turns were genuinely served, and the board says so.
    const served = page.getByTestId('hud-tile-turns-served');
    await expect(served).toHaveAttribute('data-absent', 'false');
    await expect(served.getByTestId('hud-value')).toHaveText('27');
  });

  test('the mode lamp separates a live deployment from a keyless replay by shape', async ({
    page,
    request,
  }) => {
    const { mode } = await health(request);
    const demo = mode === 'demo';
    await openBoard(page);

    const lamp = page.getByTestId('lamp-mode');
    // Dashed amber = replayed from a recording; solid green = answered live.
    // Shape, not colour alone, so the distinction survives greyscale.
    await expect(lamp).toHaveAttribute('data-shape', demo ? 'dashed' : 'solid');
    await expect(lamp).toHaveAttribute('data-tone', demo ? 'amber' : 'good');
    await expect(lamp).toContainText(demo ? /replay/i : /live/i);
  });

  test('both CTAs lead into the chat', async ({ page }) => {
    await openBoard(page);

    // The secondary CTA — "or start from any filing".
    await expect(page.getByTestId('landing-cta')).toHaveAttribute('href', '/chat');

    // The primary one, clicked, because that is the path every other spec takes.
    await page.getByTestId('landing-enter').click();
    await expect(page).toHaveURL(/\/chat$/);
    await expect(page.getByTestId('landing-screen')).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// The chat's empty state — where this file's original assertions moved to
// ---------------------------------------------------------------------------

test.describe('chat empty state, entered from the board', () => {
  test('new user lands on the welcome quick-start', async ({ page }) => {
    await page.addInitScript((apiBase: string) => {
      window.localStorage.clear();
      window.localStorage.setItem('convfinqa.apiBase', apiBase);
    }, BACKEND);
    await enterChatFromBoard(page);

    const landing = page.getByTestId('landing-screen');
    await expect(landing).toBeVisible();
    await expect(landing).toHaveAttribute('data-variant', 'new');
    await expect(landing).toContainText(/pick a report/i);
    await expect(landing).toContainText(/ask a question/i);
    await expect(landing).toContainText(/run all gold/i);

    // This copy used to sit on the landing page; it is the sessions pane's
    // empty state now, and it is asserted here rather than deleted.
    await expect(page.getByText(/your conversations will appear here/i)).toBeVisible();

    await page.getByTestId('landing-cta').click();
    await expect(page.getByTestId('report-picker-input')).toBeVisible();
  });

  test('returning user with no active selection sees compact variant', async ({ page }) => {
    await page.addInitScript(
      ([apiBase, rid]: [string, string]) => {
        window.localStorage.setItem('convfinqa.apiBase', apiBase);
        const now = Date.now();
        window.localStorage.setItem(
          'convfinqa.v1',
          JSON.stringify({
            state: {
              activeReportId: null,
              conversations: {
                [rid]: {
                  reportId: rid,
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
      },
      [BACKEND, SEED_RID] as [string, string]
    );
    await enterChatFromBoard(page);

    const landing = page.getByTestId('landing-screen');
    await expect(landing).toBeVisible();
    await expect(landing).toHaveAttribute('data-variant', 'returning');

    await expect(page.getByText(SEED_RID).first()).toBeVisible();
  });
});
