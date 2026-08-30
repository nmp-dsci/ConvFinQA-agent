import { beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * The store's *actions*, as opposed to `applyEvent`, which `store.test.ts`
 * covers.
 *
 * These are the reducers the e2e suite exercises through a browser and a live
 * model — reset dropping the server session, the unread badge appearing only on
 * a conversation nobody is looking at, a double-send being refused. Each of
 * those costs a real turn to check end to end, and three of them are the only
 * assertion in an entire spec. Pinning them here means a regression shows up in
 * 300 ms of vitest rather than in a paid Playwright run, and it means they are
 * still checked in CI, where there is no API key and the streaming specs
 * cannot run at all.
 */

const api = vi.hoisted(() => ({
  listReports: vi.fn<() => Promise<string[]>>(),
  getQuestions: vi.fn(),
  createSession: vi.fn<() => Promise<string>>(),
  getSession: vi.fn(),
  deleteSession: vi.fn<() => Promise<void>>(),
  streamAsk: vi.fn(),
}));

const libApi = vi.hoisted(() => ({
  listDemoReports: vi.fn(),
}));

vi.mock('./api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./api')>();
  return { ...actual, ...api };
});

vi.mock('./lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./lib/api')>();
  return { ...actual, ...libApi };
});

const { useStore } = await import('./store');
const { SessionGoneError } = await import('./api');
import type { SSEEvent } from './types';
import type { StreamAskArgs } from './api';

const RID = 'Single_VLO/2011/page_126.pdf-1';
const OTHER = 'Single_AES/2003/page_168.pdf-1';

/** A `streamAsk` that replays a fixed script of frames and resolves. */
function scripted(events: SSEEvent[]): (args: StreamAskArgs) => Promise<void> {
  return async ({ onEvent }: StreamAskArgs) => {
    for (const event of events) onEvent(event);
  };
}

const ANSWERED: SSEEvent[] = [
  { event: 'stage_start', stage: 'triage' },
  { event: 'answer', answer: '227.0' },
  { event: 'done', turn_index: 0, trace_id: 't1' },
];

function reset(): void {
  useStore.setState({
    reports: [],
    reportsLoading: false,
    reportsError: null,
    activeReportId: null,
    conversations: {},
    pickerOpen: false,
    examples: [],
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  reset();
  api.createSession.mockResolvedValue('sid-1');
  api.getSession.mockResolvedValue({ n_turns: 0 });
  api.deleteSession.mockResolvedValue(undefined);
  api.streamAsk.mockImplementation(scripted(ANSWERED));
});

describe('selectReport', () => {
  it('creates a conversation for a filing that has none, and activates it', async () => {
    await useStore.getState().selectReport(RID);
    const state = useStore.getState();
    expect(state.activeReportId).toBe(RID);
    expect(state.conversations[RID]).toMatchObject({
      reportId: RID,
      sessionId: null,
      messages: [],
      unreadCount: 0,
      isStreaming: false,
    });
  });

  it('clears the unread badge when reopening a conversation', async () => {
    // The second half of the unread contract, and the last assertion in
    // `unread.spec.ts`: clicking the row must zero the badge.
    await useStore.getState().selectReport(RID);
    useStore.setState((s) => ({
      conversations: { ...s.conversations, [RID]: { ...s.conversations[RID], unreadCount: 3 } },
      activeReportId: null,
    }));

    await useStore.getState().selectReport(RID);
    expect(useStore.getState().conversations[RID].unreadCount).toBe(0);
  });

  it('closes the report picker', async () => {
    useStore.getState().openPicker();
    expect(useStore.getState().pickerOpen).toBe(true);
    await useStore.getState().selectReport(RID);
    expect(useStore.getState().pickerOpen).toBe(false);
  });
});

describe('ask', () => {
  it('appends the question and folds the stream into one assistant turn', async () => {
    await useStore.getState().selectReport(RID);
    await useStore.getState().ask(RID, 'what was the value?', '227.0');

    const messages = useStore.getState().conversations[RID].messages;
    expect(messages.map((m) => m.role)).toEqual(['user', 'assistant']);
    expect(messages[1]).toMatchObject({
      text: '227.0',
      status: 'done',
      goldAnswer: '227.0',
      traceId: 't1',
    });
    expect(useStore.getState().conversations[RID].isStreaming).toBe(false);
  });

  it('refuses a second send while the first is still streaming', async () => {
    // Without this guard the composer's Enter key can start two streams into
    // one session and interleave two answers onto one turn.
    await useStore.getState().selectReport(RID);
    useStore.setState((s) => ({
      conversations: { ...s.conversations, [RID]: { ...s.conversations[RID], isStreaming: true } },
    }));

    await useStore.getState().ask(RID, 'second question');
    expect(useStore.getState().conversations[RID].messages).toHaveLength(0);
    expect(api.streamAsk).not.toHaveBeenCalled();
  });

  it('raises an unread count only on a conversation nobody is looking at', async () => {
    await useStore.getState().selectReport(RID);
    await useStore.getState().selectReport(OTHER); // OTHER is now active

    await useStore.getState().ask(RID, 'answered in the background');
    expect(useStore.getState().conversations[RID].unreadCount).toBe(1);

    await useStore.getState().ask(OTHER, 'answered in the foreground');
    expect(useStore.getState().conversations[OTHER].unreadCount).toBe(0);
  });

  it('does not count a failed turn as an unread answer', async () => {
    await useStore.getState().selectReport(RID);
    await useStore.getState().selectReport(OTHER);
    api.streamAsk.mockImplementation(
      scripted([{ event: 'error', error: 'no recording', code: 'no_recording' }])
    );

    await useStore.getState().ask(RID, 'unanswerable');
    const conversation = useStore.getState().conversations[RID];
    expect(conversation.messages[1].status).toBe('error');
    expect(conversation.unreadCount).toBe(0);
  });

  it('ends a transport failure as a failed turn, not a stuck stream', async () => {
    await useStore.getState().selectReport(RID);
    api.streamAsk.mockRejectedValue(new Error('connection reset'));

    await useStore.getState().ask(RID, 'what was the value?');
    const conversation = useStore.getState().conversations[RID];
    expect(conversation.messages[1].status).toBe('error');
    expect(conversation.messages[1].errorText).toContain('connection reset');
    // The one that matters: a thrown stream must still clear the flag, or the
    // composer stays disabled for the rest of the session.
    expect(conversation.isStreaming).toBe(false);
  });

  it('recreates the session and retries once when the server has forgotten it', async () => {
    await useStore.getState().selectReport(RID);
    api.streamAsk
      .mockImplementationOnce(async () => {
        throw new SessionGoneError();
      })
      .mockImplementation(scripted(ANSWERED));

    await useStore.getState().ask(RID, 'what was the value?');

    const messages = useStore.getState().conversations[RID].messages;
    expect(messages.some((m) => m.role === 'system' && /session expired/i.test(m.text))).toBe(true);
    expect(messages.find((m) => m.role === 'assistant')?.status).toBe('done');
    expect(api.createSession).toHaveBeenCalledTimes(2);
  });
});

describe('hydrateSession', () => {
  it('mints a new session when the old one has been evicted', async () => {
    await useStore.getState().selectReport(RID);
    useStore.setState((s) => ({
      conversations: { ...s.conversations, [RID]: { ...s.conversations[RID], sessionId: 'dead' } },
    }));
    api.getSession.mockResolvedValue(null); // 404
    api.createSession.mockResolvedValue('sid-fresh');

    await expect(useStore.getState().hydrateSession(RID)).resolves.toBe('sid-fresh');
    expect(useStore.getState().conversations[RID].sessionId).toBe('sid-fresh');
  });

  it('reuses a session the server still knows about', async () => {
    await useStore.getState().selectReport(RID);
    useStore.setState((s) => ({
      conversations: { ...s.conversations, [RID]: { ...s.conversations[RID], sessionId: 'alive' } },
    }));
    api.getSession.mockResolvedValue({ n_turns: 2 });

    await expect(useStore.getState().hydrateSession(RID)).resolves.toBe('alive');
    expect(api.createSession).not.toHaveBeenCalled();
  });
});

describe('resetConversation', () => {
  it('deletes the server session and empties the local thread', async () => {
    // `reset.spec.ts` asserts the server side of this: `/sessions/<id>` must
    // 404 afterwards, and the next question must start a session with exactly
    // one turn. Both depend on the session id being dropped here — keeping it
    // would send the next question into the old, still-populated session.
    await useStore.getState().selectReport(RID);
    await useStore.getState().ask(RID, 'first question');
    expect(useStore.getState().conversations[RID].sessionId).toBe('sid-1');

    await useStore.getState().resetConversation(RID);

    expect(api.deleteSession).toHaveBeenCalledWith('sid-1');
    expect(useStore.getState().conversations[RID]).toMatchObject({
      messages: [],
      sessionId: null,
      unreadCount: 0,
    });
  });

  it('is a no-op while a turn is streaming', async () => {
    await useStore.getState().selectReport(RID);
    useStore.setState((s) => ({
      conversations: {
        ...s.conversations,
        [RID]: { ...s.conversations[RID], isStreaming: true, sessionId: 'sid-1' },
      },
    }));

    await useStore.getState().resetConversation(RID);
    expect(api.deleteSession).not.toHaveBeenCalled();
    expect(useStore.getState().conversations[RID].sessionId).toBe('sid-1');
  });
});

describe('runAllGold', () => {
  it('asks the supplied questions in order and does not consult the dataset', async () => {
    // In demo mode the caller passes the *recorded* questions. Falling back to
    // the dataset's there would produce a column of "no recording" refusals.
    await useStore.getState().selectReport(RID);
    await useStore.getState().runAllGold(RID, [
      { question: 'q1', gold: '1' },
      { question: 'q2', gold: '2' },
    ]);

    expect(api.getQuestions).not.toHaveBeenCalled();
    const asked = useStore
      .getState()
      .conversations[RID].messages.filter((m) => m.role === 'user')
      .map((m) => m.text);
    expect(asked).toEqual(['q1', 'q2']);
  });

  it('falls back to the dataset when no list is supplied', async () => {
    api.getQuestions.mockResolvedValue([
      { question: 'gold q', gold_answer: '5', gold_program: 'add(2, 3)' },
    ]);
    await useStore.getState().selectReport(RID);
    await useStore.getState().runAllGold(RID);

    const assistant = useStore.getState().conversations[RID].messages[1];
    expect(assistant.goldAnswer).toBe('5');
    expect(assistant.goldProgram).toBe('add(2, 3)');
  });
});

describe('loadExamples', () => {
  it('offers the pack conversations the sessions pane shows first', async () => {
    libApi.listDemoReports.mockResolvedValue([
      { report_id: 'Double_MAR/2010/page_55.pdf', n_questions: 7 },
    ]);
    await useStore.getState().loadExamples();
    expect(useStore.getState().examples).toEqual([
      { reportId: 'Double_MAR/2010/page_55.pdf', nQuestions: 7 },
    ]);
  });

  it('treats a deployment with no pack as empty rather than broken', async () => {
    libApi.listDemoReports.mockRejectedValue(new Error('404'));
    await useStore.getState().loadExamples();
    expect(useStore.getState().examples).toEqual([]);
  });
});

describe('loadReports', () => {
  it('surfaces a failed read as an error the picker can show', async () => {
    // The store logs this deliberately; swallow it so a passing run is quiet.
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});
    api.listReports.mockRejectedValue(new Error('backend down'));
    await useStore.getState().loadReports();
    expect(logged).toHaveBeenCalled();
    logged.mockRestore();
    expect(useStore.getState().reportsError).toContain('backend down');
    expect(useStore.getState().reportsLoading).toBe(false);
  });
});
