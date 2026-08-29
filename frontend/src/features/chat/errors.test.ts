import { describe, expect, it } from 'vitest';
import { errorCopy } from './errors';

/**
 * The failure vocabulary, held to its own premise.
 *
 * `src/convfinqa/error_codes.py` classifies every failed turn into one of six
 * codes, and the argument for doing that work at all is that each one implies a
 * *different* response for the reader. If two codes end up sharing a headline,
 * or a code silently falls through to "the turn failed", the classification is
 * being thrown away at the last step and the backend may as well have sent
 * prose. These tests are what make that claim checkable.
 */

/** Mirrors `ALL_CODES` in `src/convfinqa/error_codes.py`. */
const CODES = [
  'llm_unavailable',
  'not_available_demo',
  'no_recording',
  'rate_limited',
  'timeout',
  'unknown',
] as const;

describe('errorCopy', () => {
  it('says something different for every code the backend can send', () => {
    const titles = CODES.map((code) => errorCopy(code, 'detail').title);
    expect(new Set(titles).size).toBe(CODES.length);
    for (const title of titles) expect(title.length).toBeGreaterThan(0);
  });

  it('offers a retry only where retrying could plausibly work', () => {
    // Suggesting "try again" for a deployment that holds no API key, or for a
    // question the pack simply does not contain, sends the reader in a loop.
    expect(errorCopy('llm_unavailable', '').retryable).toBe(true);
    expect(errorCopy('rate_limited', '').retryable).toBe(true);
    expect(errorCopy('timeout', '').retryable).toBe(true);
    expect(errorCopy('not_available_demo', '').retryable).toBe(false);
    expect(errorCopy('no_recording', '').retryable).toBe(false);
  });

  it('names the demo gate rather than blaming the question', () => {
    const demo = errorCopy('not_available_demo', 'writes are disabled');
    expect(demo.hint).toMatch(/no API key|makes no live model calls/i);
    const missing = errorCopy('no_recording', 'nothing recorded');
    expect(missing.hint).toMatch(/recorded|suggested/i);
  });

  it('falls back to `unknown` for a code this client has not learned yet', () => {
    // The backend may grow a seventh code before the frontend does. It must
    // degrade to the honest "we could not classify this", not to a blank.
    const future = errorCopy('quota_exhausted', 'over budget');
    expect(future).toEqual(errorCopy('unknown', 'over budget'));
  });

  it('does not blame the server for a stream the reader cancelled', () => {
    const cancelled = errorCopy(undefined, 'aborted');
    expect(cancelled.title).toMatch(/cancel/i);
    expect(cancelled).not.toEqual(errorCopy('unknown', 'aborted'));
  });

  it('treats an uncoded failure as unknown, not as a cancellation', () => {
    expect(errorCopy(undefined, 'ECONNRESET').title).toBe(errorCopy('unknown', '').title);
  });
});
