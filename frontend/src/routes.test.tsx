/**
 * The URL grammar, enforced: every address the Console ever published still lands, on the page
 * that replaced it, with its lens (query string) intact; every tab and sub-page lands on itself.
 * Drives the real route table in a memory router (loaders and redirects run; nothing renders).
 */
import { describe, expect, it } from 'vitest';
import { SLOTS, SUBPAGES, slotOf, subpageOf } from './app/nav';
import { land } from './kit/harness';
import { routes } from './routes';

const RID = 'Single_VLO/2011/page_126.pdf-1';

describe('every address the Console published redirects to its slot', () => {
  const cases: [string, string, string][] = [
    ['/welcome', '/', 'overview'],
    ['/chat', '/agent/chat', 'agent-chat'],
    [`/chat/${RID}`, `/agent/chat/${RID}`, 'agent-chat'],
    ['/debrief', '/agent', 'agent-architecture'],
    ['/admin', '/optimise', 'optimise-campaigns'],
    ['/admin/system', '/agent', 'agent-architecture'],
    ['/admin/readiness', '/agent/readiness', 'agent-readiness'],
    ['/admin/runtimes', '/optimise/runtimes', 'optimise-runtimes'],
    ['/admin/experiments', '/optimise', 'optimise-campaigns'],
    ['/admin/campaigns', '/optimise', 'optimise-campaigns'],
    ['/admin/research', '/optimise/research', 'optimise-research'],
    ['/admin/evaluations', '/runs', 'runs-versions'],
    ['/admin/dataset', '/filings', 'filings'],
    ['/admin/traces', '/runs/traces', 'runs-traces'],
    ['/admin/traces/abc123', '/runs/traces/abc123', 'runs-trace'],
  ];
  for (const [from, to, route] of cases) {
    it(`${from} → ${to}`, async () => {
      const got = await land(routes, from);
      expect(got.path).toBe(to);
      expect(got.route).toBe(route);
    });
  }

  it('keeps the lens: a query string survives the move', async () => {
    const got = await land(routes, '/admin/evaluations?version=v8&split=test');
    expect(got.path).toBe('/runs');
    expect(got.search).toBe('?version=v8&split=test');
    const eval_ = await land(routes, '/admin/traces/eval?version=v8&report_id=x&turn_index=2');
    expect(eval_.path).toBe('/runs/traces/eval');
    expect(eval_.search).toBe('?version=v8&report_id=x&turn_index=2');
  });
});

describe('the eight slots', () => {
  it('are the house order', () => {
    expect(SLOTS.map((s) => s.label)).toEqual([
      'Overview',
      'Filings & questions',
      'Scoring',
      'Leaderboard',
      'Runs',
      'Optimise',
      'Agent',
      'Gold review',
    ]);
  });

  it('every tab and every sub-page lands on itself', async () => {
    const targets = [...SLOTS.map((s) => s.to), ...Object.values(SUBPAGES).flatMap((p) => p!.map((x) => x.to))];
    for (const to of targets) {
      const got = await land(routes, to);
      expect(got.path, to).toBe(to);
      expect(got.route, to).not.toBe('not-found');
    }
  });

  it('an unknown address is a 404, not a blank page', async () => {
    expect((await land(routes, '/nope')).route).toBe('not-found');
  });

  it('lights the right tab and chip', () => {
    expect(slotOf('/')?.key).toBe('overview');
    expect(slotOf('/runs/traces/abc')?.key).toBe('runs');
    expect(slotOf(`/agent/chat/${RID}`)?.key).toBe('agent');
    expect(subpageOf('/runs/traces/abc')).toBe('traces');
    expect(subpageOf('/runs')).toBe('versions');
    expect(subpageOf('/optimise/runtimes')).toBe('runtimes');
    expect(subpageOf('/agent/chat')).toBe('chat');
  });
});
