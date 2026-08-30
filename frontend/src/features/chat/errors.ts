/**
 * Copy for every code `src/convfinqa/error_codes.py` can put on a failed turn.
 *
 * The backend's vocabulary is deliberately short: a code earns its place by
 * implying a *different response*. That only pays off if the UI actually says
 * something different for each one — "Something went wrong" for all six throws
 * away the classification the backend went to the trouble of making.
 *
 * The free-text message is never discarded; it is shown beneath as the detail.
 * The code decides the headline and the suggested next move.
 */
export interface ErrorCopy {
  title: string;
  hint: string;
  /** `true` when retrying the same question could plausibly work. */
  retryable: boolean;
}

const COPY: Record<string, ErrorCopy> = {
  llm_unavailable: {
    title: 'The model provider did not answer',
    hint: 'The upstream API refused or dropped the connection. Nothing about this question is wrong — try it again in a moment.',
    retryable: true,
  },
  not_available_demo: {
    title: 'Not available in the public demo',
    hint: 'This deployment holds no API key and makes no live model calls. It replays conversations recorded in development.',
    retryable: false,
  },
  no_recording: {
    title: 'No recording for that question',
    hint: 'The demo answers from a recorded pack. Pick one of the suggested questions below to watch the pipeline run end to end.',
    retryable: false,
  },
  rate_limited: {
    title: 'Rate limited by the model provider',
    hint: 'Too many calls in too short a window. Waiting a few seconds and asking again is the whole fix.',
    retryable: true,
  },
  timeout: {
    title: 'The turn ran out of time',
    hint: 'A stage exceeded the call budget and was abandoned, so the turn has no answer rather than a half-formed one.',
    retryable: true,
  },
  unknown: {
    title: 'The turn failed',
    hint: 'The server could not classify this failure, so the message below is the whole of what is known.',
    retryable: true,
  },
};

export function errorCopy(code: string | undefined, message: string | undefined): ErrorCopy {
  if (code && COPY[code]) return COPY[code];
  // An aborted stream is the user's own doing and never reaches the server.
  if (message === 'aborted') {
    return {
      title: 'Turn cancelled',
      hint: 'You stopped this turn before it finished. Nothing was recorded.',
      retryable: true,
    };
  }
  return COPY.unknown;
}
