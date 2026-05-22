import { nanoid } from 'nanoid';
import { create } from 'zustand';
import { persist, type PersistStorage, type StorageValue } from 'zustand/middleware';
import * as api from './api';
import { SessionGoneError } from './api';
import type { Conversation, Message, SSEEvent, StageName } from './types';

// AbortControllers are not serializable — keep out of persisted state.
const ABORT_CONTROLLERS = new Map<string, AbortController>();

interface State {
  reports: string[];
  reportsLoading: boolean;
  reportsError: string | null;
  activeReportId: string | null;
  conversations: Record<string, Conversation>;
  pickerOpen: boolean;
}

interface Actions {
  loadReports: () => Promise<void>;
  selectReport: (rid: string) => Promise<void>;
  ask: (rid: string, question: string, gold?: string, goldProgram?: string) => Promise<void>;
  runAllGold: (rid: string) => Promise<void>;
  resetConversation: (rid: string) => Promise<void>;
  markRead: (rid: string) => void;
  hydrateSession: (rid: string) => Promise<string>;
  openPicker: () => void;
  closePicker: () => void;
  abortStream: (rid: string) => void;
}

export type Store = State & Actions;

const STORAGE_KEY = 'convfinqa.v1';

const safeStorage: PersistStorage<unknown> = {
  getItem(name) {
    try {
      const raw = window.localStorage.getItem(name);
      if (!raw) return null;
      return JSON.parse(raw) as StorageValue<unknown>;
    } catch {
      return null;
    }
  },
  setItem(name, value) {
    try {
      window.localStorage.setItem(name, JSON.stringify(value));
    } catch {
      /* quota / private mode — ignore */
    }
  },
  removeItem(name) {
    try {
      window.localStorage.removeItem(name);
    } catch {
      /* ignore */
    }
  },
};

function freshConversation(rid: string): Conversation {
  const now = Date.now();
  return {
    reportId: rid,
    sessionId: null,
    messages: [],
    lastUsedAt: now,
    lastReadAt: now,
    unreadCount: 0,
    isStreaming: false,
  };
}

export const useStore = create<Store>()(
  persist(
    (set, get) => ({
      reports: [],
      reportsLoading: false,
      reportsError: null,
      activeReportId: null,
      conversations: {},
      pickerOpen: false,

      openPicker: () => {
        set({ pickerOpen: true });
        // Refetch on open if we have nothing — most common cause is the
        // backend was down on first mount.
        if (get().reports.length === 0 && !get().reportsLoading) {
          void get().loadReports();
        }
      },
      closePicker: () => set({ pickerOpen: false }),

      loadReports: async () => {
        set({ reportsLoading: true, reportsError: null });
        try {
          const reports = await api.listReports();
          set({ reports, reportsLoading: false, reportsError: null });
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          set({ reportsLoading: false, reportsError: msg });
          // eslint-disable-next-line no-console
          console.error('loadReports failed', e);
        }
      },

      selectReport: async (rid) => {
        const exists = !!get().conversations[rid];
        set((state) => {
          const conversations = { ...state.conversations };
          if (!conversations[rid]) {
            conversations[rid] = freshConversation(rid);
          }
          return { ...state, conversations, activeReportId: rid, pickerOpen: false };
        });
        if (exists) get().markRead(rid);
      },

      markRead: (rid) =>
        set((state) => {
          const conv = state.conversations[rid];
          if (!conv) return state;
          return {
            ...state,
            conversations: {
              ...state.conversations,
              [rid]: { ...conv, unreadCount: 0, lastReadAt: Date.now() },
            },
          };
        }),

      hydrateSession: async (rid) => {
        const conv = get().conversations[rid];
        if (!conv) throw new Error(`No conversation for ${rid}`);
        if (conv.sessionId) {
          const info = await api.getSession(conv.sessionId);
          if (info) return conv.sessionId;
          // 404 → session evicted; fall through to create
        }
        const sessionId = await api.createSession(rid);
        set((state) => ({
          ...state,
          conversations: {
            ...state.conversations,
            [rid]: { ...state.conversations[rid], sessionId },
          },
        }));
        return sessionId;
      },

      ask: async (rid, question, gold, goldProgram) => {
        const conv = get().conversations[rid];
        if (!conv || conv.isStreaming) return;

        const userMsg: Message = {
          id: nanoid(),
          role: 'user',
          text: question,
          status: 'done',
          createdAt: Date.now(),
        };
        const asstMsg: Message = {
          id: nanoid(),
          role: 'assistant',
          text: '',
          goldAnswer: gold,
          goldProgram: goldProgram,
          status: 'streaming',
          stages: {},
          tools: [],
          createdAt: Date.now(),
        };
        appendMessages(set, rid, [userMsg, asstMsg]);
        patchConversation(set, rid, (c) => ({
          ...c,
          isStreaming: true,
          lastUsedAt: Date.now(),
        }));

        const controller = new AbortController();
        ABORT_CONTROLLERS.set(rid, controller);

        const handleEvent = (ev: SSEEvent) => {
          patchAssistant(set, rid, asstMsg.id, (m) => applyEvent(m, ev));
        };

        let sessionId: string;
        try {
          sessionId = await get().hydrateSession(rid);
          await api.streamAsk({
            sessionId,
            question,
            signal: controller.signal,
            onEvent: handleEvent,
          });
        } catch (err) {
          if (err instanceof SessionGoneError) {
            // Recreate session and retry once with a system marker.
            patchConversation(set, rid, (c) => ({ ...c, sessionId: null }));
            try {
              const newSid = await get().hydrateSession(rid);
              appendMessages(set, rid, [
                {
                  id: nanoid(),
                  role: 'system',
                  text: 'Session expired — starting a fresh one.',
                  status: 'done',
                  createdAt: Date.now(),
                },
              ]);
              await api.streamAsk({
                sessionId: newSid,
                question,
                signal: controller.signal,
                onEvent: handleEvent,
              });
            } catch (err2) {
              patchAssistant(set, rid, asstMsg.id, (m) => ({
                ...m,
                status: 'error',
                errorText: String(err2),
              }));
            }
          } else if ((err as { name?: string })?.name === 'AbortError') {
            patchAssistant(set, rid, asstMsg.id, (m) => ({
              ...m,
              status: 'error',
              errorText: 'aborted',
            }));
          } else {
            patchAssistant(set, rid, asstMsg.id, (m) => ({
              ...m,
              status: 'error',
              errorText: String(err),
            }));
          }
        } finally {
          ABORT_CONTROLLERS.delete(rid);
          patchConversation(set, rid, (c) => ({
            ...c,
            isStreaming: false,
            lastUsedAt: Date.now(),
          }));
          // Unread bookkeeping: bump count if user is not looking.
          const finalConv = get().conversations[rid];
          const finalMsg = finalConv?.messages.find((m) => m.id === asstMsg.id);
          if (finalMsg && finalMsg.status === 'done') {
            if (get().activeReportId !== rid) {
              patchConversation(set, rid, (c) => ({
                ...c,
                unreadCount: c.unreadCount + 1,
              }));
            } else {
              get().markRead(rid);
            }
          }
        }
      },

      runAllGold: async (rid) => {
        let questions;
        try {
          questions = await api.getQuestions(rid);
        } catch (e) {
          // eslint-disable-next-line no-console
          console.error('getQuestions failed', e);
          return;
        }
        for (const q of questions) {
          await get().ask(rid, q.question, q.gold_answer, q.gold_program);
        }
      },

      resetConversation: async (rid) => {
        const conv = get().conversations[rid];
        if (!conv || conv.isStreaming) return;
        const oldSid = conv.sessionId;
        if (oldSid) {
          api.deleteSession(oldSid).catch(() => {
            /* already gone — fine */
          });
        }
        patchConversation(set, rid, (c) => ({
          ...c,
          messages: [],
          sessionId: null,
          unreadCount: 0,
          lastUsedAt: Date.now(),
          lastReadAt: Date.now(),
        }));
      },

      abortStream: (rid) => {
        const ac = ABORT_CONTROLLERS.get(rid);
        if (ac) ac.abort();
      },
    }),
    {
      name: STORAGE_KEY,
      storage: safeStorage,
      partialize: (state) => ({
        conversations: stripVolatile(state.conversations),
        activeReportId: state.activeReportId,
      }),
      version: 2,
      migrate: (persisted) => {
        // v1 persisted `reports`; v2 drops it (server state, not user state).
        if (persisted && typeof persisted === 'object' && 'reports' in persisted) {
          const { reports: _drop, ...rest } = persisted as Record<string, unknown>;
          void _drop;
          return rest;
        }
        return persisted;
      },
    }
  )
);

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function patchConversation(
  set: (fn: (state: Store) => Store) => void,
  rid: string,
  patch: (c: Conversation) => Conversation
): void {
  set((state) => {
    const conv = state.conversations[rid];
    if (!conv) return state;
    return {
      ...state,
      conversations: { ...state.conversations, [rid]: patch(conv) },
    };
  });
}

function appendMessages(
  set: (fn: (state: Store) => Store) => void,
  rid: string,
  messages: Message[]
): void {
  set((state) => {
    const conv = state.conversations[rid];
    if (!conv) return state;
    return {
      ...state,
      conversations: {
        ...state.conversations,
        [rid]: { ...conv, messages: [...conv.messages, ...messages] },
      },
    };
  });
}

function patchAssistant(
  set: (fn: (state: Store) => Store) => void,
  rid: string,
  msgId: string,
  patch: (m: Message) => Message
): void {
  set((state) => {
    const conv = state.conversations[rid];
    if (!conv) return state;
    const messages = conv.messages.map((m) => (m.id === msgId ? patch(m) : m));
    return {
      ...state,
      conversations: { ...state.conversations, [rid]: { ...conv, messages } },
    };
  });
}

function applyEvent(message: Message, event: SSEEvent): Message {
  switch (event.event) {
    case 'stage_start': {
      const stages = { ...(message.stages ?? {}) };
      const existing = stages[event.stage as StageName];
      stages[event.stage as StageName] = { ...existing, started: true };
      return { ...message, stages };
    }
    case 'stage_output': {
      const stages = { ...(message.stages ?? {}) };
      const existing = stages[event.stage as StageName] ?? { started: true };
      stages[event.stage as StageName] = { ...existing, started: true, output: event.output };
      return { ...message, stages };
    }
    case 'tool_call': {
      const tools = [
        ...(message.tools ?? []),
        { tool: event.tool, args: api.normalizeArgs(event.args) },
      ];
      return { ...message, tools };
    }
    case 'tool_return': {
      const tools = [...(message.tools ?? [])];
      // Fill the most recent matching tool call without a result.
      for (let i = tools.length - 1; i >= 0; i--) {
        const t = tools[i];
        if (t && t.tool === event.tool && t.result === undefined) {
          tools[i] = { ...t, result: event.result };
          break;
        }
      }
      return { ...message, tools };
    }
    case 'answer':
      return { ...message, text: event.answer };
    case 'done':
      return {
        ...message,
        status: message.status === 'error' ? 'error' : 'done',
      };
    case 'error':
      return { ...message, status: 'error', errorText: event.error };
    default:
      return message;
  }
}

function stripVolatile(
  conversations: Record<string, Conversation>
): Record<string, Conversation> {
  const out: Record<string, Conversation> = {};
  for (const [rid, conv] of Object.entries(conversations)) {
    out[rid] = { ...conv, isStreaming: false };
  }
  return out;
}
