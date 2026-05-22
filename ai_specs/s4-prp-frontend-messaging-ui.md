name: "ConvFinQA Frontend — WhatsApp/Messenger-Style Chat UI for the FastAPI Agent"
description: |

## Purpose
Build a dark-mode, messaging-app-style web frontend over the existing FastAPI streaming
endpoint (`/sessions/{sid}/ask/stream`) that mirrors every capability of `cli.py`:

- pick a `report_id` (free-text search **and** dropdown)
- one chat conversation per `report_id`, persisted across reloads, switchable from a
  WhatsApp-style left sidebar
- ask any question, stream the four-stage pipeline events (triage → preprocess →
  retriever → calculator + tool calls) into the assistant bubble
- "Run all gold questions" button per conversation
- "Reset conversation" button per conversation that clears the message history
  for that report so the next question is answered with **no prior history**
  supplied to the agent (server-side: drop the existing session and create a
  fresh one for the same `report_id`)
- correctness coloring (green ✓ / red ✗) when a gold answer is known
- two conversations can run **truly concurrently** (independent SSE streams against
  independent sessions); validated by an e2e Playwright test that drives gold-question
  runs on two reports in parallel and asserts both finish with correct answers.

## Core Principles
1. **Reuse the existing API.** No new backend logic — just CORS + static-file mounting.
2. **One session per report.** `report_id` is immutable per session (existing API rule);
   frontend persists `{report_id → session_id}` in `localStorage` and recreates on 404.
3. **Streaming first.** All assistant replies flow through `/ask/stream`; the bubble
   fills incrementally — never block-then-render.
4. **Per-conversation state.** Each conversation owns its own `isStreaming`, message
   list, and abort controller, so two reports can stream at the same time without
   blocking each other.
5. **Validation Loops.** TypeScript strict + Vite build + Playwright e2e all runnable
   from one `npm` command.
6. **Global rules:** Follow `CLAUDE.md` — `uv` for Python, Ruff/mypy for backend,
   no committed `.env`, no API keys hardcoded.

---

## Goal

A `frontend/` directory inside this repo containing a single-page React app that, when
served alongside `app.py`, gives the user a Messenger/WhatsApp-style UI for the
ConvFinQA agent. Specifically:

- **Dark theme** (true dark, not just a flag — the app should feel native dark).
- **Top bar** showing the current `report_id` plus a "Change report" affordance that
  opens a combo control with both free-text search and full dropdown.
- **Left sidebar** listing every report the user has opened, ordered by recency, with
  unread/streaming indicators. Click switches the active conversation instantly
  without dropping in-progress streams.
- **Chat panel** with user bubbles right (accent color) and assistant bubbles left
  (subtle gray). Inside the assistant bubble while streaming, render compact
  per-stage status chips (`triage…`, `preprocess…`, `retriever…`, `calculator…`)
  and a collapsible tool-call trace for the calculator stage.
- **Composer** with a text input, Send, and a "Run all gold" button that walks
  every gold question for the active report, one at a time per session (server lock
  enforces ordering inside a session), updating the chat live.
- **Reset conversation** action (icon button in the chat panel header next to the
  report title) that wipes the active conversation's messages and discards its
  server-side session, so the next question goes to the agent with an *empty*
  `ConversationHistory`. The conversation entry stays in the sidebar for the same
  `report_id` — only its history is cleared.
- **Landing / onboarding screen** shown when no `activeReportId` is set:
  - For brand-new users (no conversations in localStorage) → a centered welcome
    panel with a 3-step quick-start guide explaining how to create a conversation
    by picking a report, plus a primary "New conversation" CTA that opens the
    `ReportPicker`. The empty sidebar gets a subtle hint ("Your conversations
    will appear here") so the user knows what the panel is for.
  - For returning users (≥1 conversation but none active) → a shorter "Pick a
    conversation from the sidebar or start a new one" empty state with the same
    "New conversation" CTA.
- **Sidebar conversation list** is the canonical "who have you talked to" view —
  one row per `report_id` ever opened, showing the report id, a one-line preview
  of the last message (user q or assistant a), the relative time of last activity,
  and a streaming spinner if the conversation is currently mid-stream. Sorted
  descending by `lastUsedAt` so the most-recently-used report is always on top.
- **Unread-message indicator (WhatsApp-style)**: when a non-active conversation
  finishes streaming a new assistant answer, mark it unread and render a small
  accent-colored dot (or count badge if multiple unread answers accumulate) on
  its sidebar row. Selecting that row clears the unread flag. The active
  conversation never shows an unread badge — incoming messages on the open chat
  are inherently read.
- **Correctness markers**: when an answer corresponds to a known gold (gold question
  asked from the suggestion list **or** via Run-all), append a green ✓ or red ✗ and
  the gold value next to the predicted answer.
- **Concurrent operation**: opening Report A, starting Run-all, switching to Report B,
  and starting Run-all there must result in both running at the same time. Validated
  by Playwright spawning two browser contexts.

## Why
- `cli.py` is fine for engineers but the user wants a demo-able UI for stakeholders.
- The streaming SSE backend is already built; without a frontend, that work is
  invisible to non-developers.
- Multi-conversation switching is impossible in `cli.py` (single session at a time);
  the frontend unlocks it without backend changes.

## What
### Success Criteria
- [ ] `cd frontend && npm install && npm run dev` opens a working app at
      `http://localhost:5173/` after `uv run python cli.py serve` is running.
- [ ] User can search and select a `report_id`; the choice appears in the side bar.
- [ ] Asking a question streams stage-by-stage updates into the assistant bubble; the
      final answer replaces the streaming chips.
- [ ] Switching reports preserves message history per report (localStorage).
- [ ] Sidebar lists every previously-opened report; click switches active
      conversation and never loses an in-progress stream on the other report.
- [ ] "Run all gold" button walks all gold questions sequentially within the report's
      session and shows ✓/✗ next to each answer.
- [ ] "Reset conversation" clears the chat panel for the active report and the
      next question's stream proves an empty history was supplied (verifiable by
      asking a turn that depends on prior context — e.g. "what is the sum?" — and
      seeing the agent fail / ask for clarification rather than inferring values).
- [ ] First-load (empty localStorage) shows the onboarding landing screen with
      a 3-step quick-start guide and a primary "New conversation" CTA.
- [ ] Sidebar lists every report the user has opened, sorted descending by
      `lastUsedAt`. Each row shows the report id, last-message preview, and
      relative timestamp.
- [ ] When a new assistant answer arrives on a *non-active* conversation, that
      sidebar row gets an unread badge (green dot, count for >1). Clicking the
      row clears the badge.
- [ ] Concurrency: starting Run-all on two reports back-to-back results in both
      streams progressing simultaneously (verified by overlapping `stage_start` events
      in DevTools network tab and asserted by the Playwright test).
- [ ] `npm run typecheck` clean, `npm run build` succeeds, `npm run test:e2e` passes.
- [ ] Backend tests still pass: `uv run pytest --ignore=tests/test_agent.py`.

## All Needed Context

### Documentation & References
```yaml
- url: https://fastapi.tiangolo.com/tutorial/cors/
  why: Adding CORSMiddleware to app.py (vite dev server is on a different port)
  critical: |
    allow_origins must include "http://localhost:5173" (vite default). Use
    allow_credentials=False (we don't send cookies), allow_methods=["*"],
    allow_headers=["*"]. Without this, the browser blocks every fetch.

- url: https://github.com/Azure/fetch-event-source
  why: |
    Browser native EventSource is GET-only and cannot send a JSON body. The
    backend's /sessions/{sid}/ask/stream is a POST with a JSON body. Use
    @microsoft/fetch-event-source (POST + JSON + abort + reconnect).
  critical: |
    - Pass `openWhenHidden: true` so background tabs don't auto-close the stream.
    - The library calls `onmessage` per `data:` line. Server emits one JSON
      object per data line (see app.py:ask_stream). JSON.parse(ev.data) inside
      onmessage.
    - Always pass an AbortController.signal so switching reports / closing the
      app cancels in-flight streams.

- url: https://vitejs.dev/guide/
  why: Vite project bootstrap + dev proxy
  critical: |
    Use `npm create vite@latest frontend -- --template react-ts`. For dev,
    proxy /sessions, /reports, /healthz to http://127.0.0.1:8000 in
    vite.config.ts so the browser sees one origin (avoids CORS in dev *and*
    prod-like setup). Don't proxy if you'd rather use CORS — pick one.

- url: https://tailwindcss.com/docs/dark-mode
  why: Dark-mode setup (we want force-dark, not media-query)
  critical: |
    `darkMode: 'class'` in tailwind.config and put `class="dark"` on <html>.
    No light mode in this project — keep it dark always.

- url: https://docs.pmnd.rs/zustand/getting-started/introduction
  why: State store pattern for per-conversation state + localStorage middleware
  critical: |
    Use the `persist` middleware to mirror state to localStorage. Storage key
    must be versioned (e.g. "convfinqa.v1") so future schema changes can bump.

- url: https://playwright.dev/docs/test-webserver
  why: Bring up backend + frontend before e2e tests
  critical: |
    `webServer` accepts an array. We start uvicorn on 8765 and vite preview on
    4173 *before* tests so concurrent runs hit a deterministic environment.

- url: https://playwright.dev/docs/browser-contexts
  why: Each test browser-context = isolated user (separate localStorage)
  critical: |
    Use two contexts (NOT two pages in one context) so each "user" has its own
    conversation list and the test is a clean concurrency check.

- file: app.py
  why: |
    Existing endpoints, request/response schemas, and the SSE protocol the
    frontend must speak.
  critical: |
    /sessions/{sid}/ask/stream emits SSE frames `data: <json>\n\n` with events
    `stage_start | stage_output | tool_call | tool_return | answer | done | error`.
    A 404 on /ask/stream means the session was evicted (TTL=1800s). Recreate
    via POST /sessions and retry once.

- file: pydantic_agent.py
  why: |
    `stream_turn` defines the exact event shape. Keep TS types aligned.
  critical: |
    `tool_call.args` is a parsed dict OR a string (when JSON parse failed). The
    frontend must accept either. `stage_output.output` shape varies per stage:
    triage = {turn_type, conv_type, reasoning},
    preprocess = {sub_questions, program, reasoning},
    retriever = {answers: [{question, answer}], reasoning},
    calculator = {answer}.

- file: cli.py
  why: |
    Reference for behavior parity. The frontend must support every action cli
    supports: pick report, ask, run-all, change report. The cli's `_print_event`
    is the spec for what events to render and how.
  critical: |
    The CLI uses _loose_numeric_match for ✓/✗. Port that exact algorithm to TS
    so frontend and CLI report identical correctness verdicts.

- file: tests/test_cli.py
  why: |
    Demonstrates the SSE mock format expected by the existing client. Useful
    when writing the e2e test's mock fallbacks (if any).
```

### Current Codebase tree (relevant parts)
```text
ConvFinQA-agent/
├── app.py                 # FastAPI; has /sessions/{sid}/ask and /ask/stream
├── cli.py                 # typer + httpx + SSE client (the spec we mirror)
├── pydantic_agent.py      # stream_turn — emits the SSE events
├── dspy_agent.py          # ConvFinQA pipeline (data + agent)
├── pyproject.toml
├── runs/                  # GEPA run artifacts
├── tests/
│   ├── test_api.py
│   ├── test_cli.py        # SSE mocking pattern
│   └── test_pydantic_agent.py
└── PRPs/
    ├── templates/prp_base.md
    └── prp-frontend-messaging-ui.md   # this file
```

### Desired Codebase tree (additions)
```text
ConvFinQA-agent/
├── app.py                       # MODIFIED: + CORSMiddleware
├── frontend/                    # NEW: React + Vite + TS + Tailwind
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts           # dev proxy → http://127.0.0.1:8000
│   ├── tailwind.config.ts       # dark mode 'class'
│   ├── postcss.config.js
│   ├── playwright.config.ts     # webServer = [uvicorn, vite preview]
│   ├── index.html               # <html class="dark">
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx              # layout: <Sidebar/> <ChatPanel/>; <TopBar/> overlay
│   │   ├── styles.css           # @tailwind base/components/utilities
│   │   ├── api.ts               # HTTP helpers + streamAsk() using fetch-event-source
│   │   ├── store.ts             # zustand store + persist middleware
│   │   ├── types.ts             # SSE event types, Conversation, Message
│   │   ├── numericMatch.ts      # port of cli._loose_numeric_match
│   │   └── components/
│   │       ├── TopBar.tsx       # current report + "Change report" button
│   │       ├── ReportPicker.tsx # combobox: text search + scrollable list
│   │       ├── Sidebar.tsx      # conversations list (recency-sorted) + unread badges
│   │       ├── LandingScreen.tsx# new-user / returning-user empty state
│   │       ├── ChatPanel.tsx    # header (report id + reset) + message list + composer
│   │       ├── MessageBubble.tsx# user/assistant/event rendering
│   │       └── Composer.tsx     # textarea + Send + Run-all-gold
│   └── tests/e2e/
│       ├── concurrent.spec.ts   # 2 contexts, 2 reports, run-all in parallel
│       ├── reset.spec.ts        # reset clears server-side history
│       ├── landing.spec.ts      # new-user vs returning-user variants
│       └── unread.spec.ts       # WhatsApp-style unread badge mechanics
└── tests/
    └── test_app_cors.py         # NEW: tiny test asserting CORS headers
```

### Known Gotchas of our codebase & Library Quirks
```text
# CRITICAL: app.py session model is one-report-per-session and immutable.
# To change reports the frontend MUST POST /sessions for each new report.

# CRITICAL: Sessions have TTL=1800s by default (app.SessionStore).
# After eviction, /ask/stream returns 404. Frontend must:
#   1) Catch 404,
#   2) POST /sessions to get a new session_id for the same report_id,
#   3) Replay nothing (server can't restore history; just warn the user
#      with a system message and continue).

# CRITICAL: server emits SSE as `data: <json>\n\n`. Each `data:` line is one
# JSON object. fetch-event-source's `onmessage` parses framing for you and
# gives you `ev.data` as the post-`data: ` string. JSON.parse it once.

# CRITICAL: tool_call.args is dict | str. The cli normalizes via JSON.parse if
# it's a string (see pydantic_agent._coerce_args). Frontend should do the same
# so display is consistent.

# CRITICAL: Browsers cap concurrent HTTP/1 connections per origin at ~6.
# Two concurrent streams is well under that, but plan for ~3-4 simultaneous
# streams before warning the user.

# CRITICAL: A single SESSION cannot have two concurrent asks (server lock in
# app.py:ask_stream). Frontend must disable Send + Run-all on the conversation
# while a stream is active, but MUST allow other conversations to keep running.

# CRITICAL: report_id contains '/' (e.g. "Single_VLO/2011/page_126.pdf-1").
# Use encodeURIComponent on path segments. The backend route uses {rid:path}
# so the encoded value still resolves correctly.

# CRITICAL: numeric_match in cli.py treats "15398.0" == "15398" as match
# because float() round-trip. Re-implement that exact algorithm in TS so the
# UI ✓/✗ never disagrees with the cli verdict.

# CRITICAL: app.py imports are heavy (dspy, pandas). Don't add Python import
# work to the request path. CORSMiddleware is fine — it's middleware, not new
# imports of heavy libs.

# CRITICAL: vite dev server proxy + CORS are both valid solutions. Pick the
# proxy approach for dev (simpler), but still add CORSMiddleware so the prod
# build (served from a different origin) works.

# CRITICAL: Playwright's webServer waits for a 200 on the configured port. Use
# /healthz for the backend (already returns {"ok": true}).
```

## Implementation Blueprint

### Data models and structure
```ts
// types.ts
export type StageName = 'triage' | 'preprocess' | 'retriever' | 'calculator';

export type SSEEvent =
  | { event: 'stage_start'; stage: StageName }
  | { event: 'stage_output'; stage: StageName; output: Record<string, unknown> }
  | { event: 'tool_call'; stage: StageName; tool: string; args: unknown }
  | { event: 'tool_return'; stage: StageName; tool: string; result: string }
  | { event: 'answer'; answer: string }
  | { event: 'done'; turn_index: number }
  | { event: 'error'; error: string };

export interface ToolTrace {
  tool: string;
  args: unknown;
  result?: string;
}

export interface Message {
  id: string;                       // uuid v4
  role: 'user' | 'assistant' | 'system';
  text: string;                     // for user: question; for assistant: final answer or '' while streaming
  goldAnswer?: string;              // optional; when set we render ✓/✗
  status: 'pending' | 'streaming' | 'done' | 'error';
  // assistant-only: live event accumulator
  stages?: Partial<Record<StageName, { started: boolean; output?: unknown }>>;
  tools?: ToolTrace[];
  errorText?: string;
}

export interface Conversation {
  reportId: string;
  sessionId: string | null;         // null if not yet created or 404'd
  messages: Message[];
  lastUsedAt: number;               // epoch ms — bumped on every send/answer
  lastReadAt: number;               // epoch ms — bumped when this conversation is the active one and an answer arrives, or when user opens it
  unreadCount: number;              // number of done assistant messages received while inactive; reset to 0 on activate
  isStreaming: boolean;             // gate concurrent asks within one conversation
}

export interface Store {
  reports: string[];                // cached from GET /reports?limit=500
  activeReportId: string | null;
  conversations: Record<string, Conversation>;
  // actions
  loadReports: () => Promise<void>;
  selectReport: (rid: string) => Promise<void>;     // also marks as read
  ask: (rid: string, question: string, gold?: string) => Promise<void>;
  runAllGold: (rid: string) => Promise<void>;
  resetConversation: (rid: string) => Promise<void>;
  markRead: (rid: string) => void;                  // clears unreadCount, bumps lastReadAt
  hydrateSession: (rid: string) => Promise<string>; // returns session_id
}
```

### List of tasks (in order)

```yaml
Task 1 — Backend CORS:
MODIFY app.py:
  - FIND: "from fastapi import FastAPI, HTTPException, Query, Response"
  - ADD: from fastapi.middleware.cors import CORSMiddleware
  - INSIDE create_app, after `app = FastAPI(lifespan=lifespan)`:
      app.add_middleware(
          CORSMiddleware,
          allow_origins=os.environ.get("FRONTEND_ORIGINS",
              "http://localhost:5173,http://localhost:4173").split(","),
          allow_credentials=False,
          allow_methods=["*"],
          allow_headers=["*"],
      )
  - PRESERVE existing logfire instrumentation order (logfire.instrument_fastapi
    must run AFTER add_middleware? No — order doesn't matter for our use,
    keep logfire calls where they are).

CREATE tests/test_app_cors.py:
  - Use fastapi.testclient.TestClient
  - Send an OPTIONS preflight with Origin: http://localhost:5173
  - Assert response headers contain "access-control-allow-origin"

Task 2 — Frontend bootstrap:
RUN (manual, document in README):
  cd frontend
  npm create vite@latest . -- --template react-ts  # answer "y" to scaffold here
  npm install
  npm install -D tailwindcss@latest postcss autoprefixer
  npm install -D @types/node @playwright/test
  npm install zustand @microsoft/fetch-event-source nanoid
  npx tailwindcss init -p
  npx playwright install chromium

EDIT package.json scripts:
  "dev": "vite",
  "build": "tsc -b && vite build",
  "preview": "vite preview --port 4173",
  "typecheck": "tsc -b --noEmit",
  "lint": "tsc -b --noEmit",
  "test:e2e": "playwright test"

EDIT tsconfig.json:
  - "strict": true, "noUnusedLocals": true, "noUnusedParameters": true.

EDIT tailwind.config.ts:
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: { extend: { colors: {
    bg: '#0b141a', panel: '#111b21', panel2: '#202c33',
    accent: '#005c4b', accent2: '#00a884',
    bubbleUser: '#005c4b', bubbleAssistant: '#202c33',
    textMain: '#e9edef', textMuted: '#8696a0', danger: '#f15c6d',
  }}}

EDIT index.html:
  <html lang="en" class="dark"> ... <body class="bg-bg text-textMain">

EDIT src/styles.css:
  @tailwind base; @tailwind components; @tailwind utilities;

EDIT src/main.tsx:
  import './styles.css'

Task 3 — Vite dev proxy + types alignment:
EDIT vite.config.ts:
  server: {
    proxy: {
      '/healthz': 'http://127.0.0.1:8000',
      '/reports': 'http://127.0.0.1:8000',
      '/sessions': 'http://127.0.0.1:8000',
    },
  },

CREATE src/types.ts:
  - As designed in "Data models" above.

Task 4 — numericMatch.ts (port of cli._loose_numeric_match):
CREATE src/numericMatch.ts:
  export function looseNumericMatch(pred: string, gold: string): boolean {
    const clean = (s: string) =>
      s.trim().replaceAll('$', '').replaceAll(',', '').replaceAll('%', '').trim();
    const p = parseFloat(clean(pred));
    const g = parseFloat(clean(gold));
    if (Number.isFinite(p) && Number.isFinite(g)) return Math.abs(p - g) < 0.01;
    return clean(pred).toLowerCase() === clean(gold).toLowerCase();
  }

CREATE src/numericMatch.test.ts (vitest optional, OR fold into Playwright):
  - "15398.0" matches "15398" → true
  - "12.4%" matches "12.4" → true
  - "abc" matches "abc" → true
  - "5.4" matches "5.41" → false (>=0.01)

Task 5 — api.ts (HTTP + SSE helpers):
CREATE src/api.ts:
  - listReports(q?, limit=500): GET /reports?q=&limit=
  - getQuestions(reportId): GET /reports/{encoded}/questions
  - createSession(reportId): POST /sessions {report_id}
  - getSession(sessionId): GET /sessions/{sid} (used to verify session is still alive)
  - deleteSession(sessionId): DELETE /sessions/{sid} (used by resetConversation;
    swallow 404 — server may have already evicted the session via TTL)
  - streamAsk({sessionId, question, signal, onEvent, onDone, onError}):
      uses fetchEventSource('/sessions/{sid}/ask/stream', {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({question}),
        signal,
        openWhenHidden: true,
        onmessage(ev) { onEvent(JSON.parse(ev.data)); },
        onerror(err) { onError(err); throw err; }, // throw stops auto-retry
      })
  - normalizeArgs(args): if string try JSON.parse, else return as-is.

GOTCHAS:
  - Always wrap report_id with encodeURIComponent in URLs.
  - On 404 from /ask/stream: caller decides whether to recreate session.

Task 6 — store.ts (zustand + persist):
CREATE src/store.ts:
  - useStore = create(persist((set, get) => ({...}), {name: 'convfinqa.v1'}))
  - actions:
      loadReports():
        const reports = await api.listReports();
        set({reports});
      selectReport(rid):
        set({activeReportId: rid});
        if (!get().conversations[rid]) {
          set(state => ({conversations: {...state.conversations, [rid]: {
            reportId: rid, sessionId: null, messages: [],
            lastUsedAt: Date.now(), lastReadAt: Date.now(),
            unreadCount: 0, isStreaming: false,
          }}}));
        } else {
          get().markRead(rid);  // opening a convo == reading it
        }
      markRead(rid):
        patchConversation(rid, c => {
          c.unreadCount = 0;
          c.lastReadAt = Date.now();
        });
      hydrateSession(rid): if no sessionId, POST /sessions; if 404 on later
        request, the caller will null and re-call this.
      ask(rid, question, gold?):
        - guard: if conversations[rid].isStreaming → throw
        - append a user Message
        - append an assistant Message (status: 'streaming')
        - hydrateSession then streamAsk; on each event, mutate the assistant
          Message (immer-style updates with set):
            stage_start → message.stages[stage] = {started: true}
            stage_output → message.stages[stage].output = output
            tool_call → push to message.tools
            tool_return → fill the matching tool entry's result
            answer → message.text = answer
            done → message.status = 'done'; isStreaming=false;
                   if get().activeReportId !== rid:
                     patchConversation(rid, c => { c.unreadCount += 1; });
                   else:
                     get().markRead(rid);  // user is looking at it
            error → message.status = 'error'; message.errorText = error
        - on 404 from POST: clear sessionId, recreate, retry ONCE with a fresh
          system message ("Session expired — starting fresh").
      runAllGold(rid):
        const qs = await api.getQuestions(rid);
        for (const q of qs) {
          await get().ask(rid, q.question, q.gold_answer);
        }
        // sequential within one conversation (server lock anyway).
      resetConversation(rid):
        // Clears the chat and forces a fresh server-side session so the
        // next ask() goes to the agent with an empty ConversationHistory.
        const conv = get().conversations[rid];
        if (!conv || conv.isStreaming) return;  // never reset mid-stream
        const oldSid = conv.sessionId;
        // Best-effort server cleanup; don't block UI on it.
        if (oldSid) {
          api.deleteSession(oldSid).catch(() => {/* already evicted? fine */});
        }
        patchConversation(rid, c => {
          c.messages = [];
          c.sessionId = null;          // ensureSession() will POST a fresh one
          c.lastUsedAt = Date.now();
        });
        // Optional: append a system "History reset" line so the user has a
        // visible confirmation. Omit if you'd rather keep the panel pristine.

CRITICAL: Don't await runAllGold from the UI handler if you want true
"fire-and-forget" so user can switch conversations. Wrap with `void
get().runAllGold(rid)`.

CRITICAL: resetConversation MUST refuse while isStreaming is true — calling
api.deleteSession on a session that's mid-stream would race the SSE handler
and produce inconsistent UI state. Disable the reset button accordingly.

Task 7 — Components:
CREATE src/components/TopBar.tsx:
  - Shows activeReportId or "Pick a report".
  - Button "Change report" toggles ReportPicker.

CREATE src/components/ReportPicker.tsx:
  - Modal overlay with:
    - <input type="search" /> (free-text search; client-side filter
      reports.filter(r => r.toLowerCase().includes(q.toLowerCase())))
    - Below: scrollable list of all matching reports (full dropdown
      behavior). Click → store.selectReport(rid) and close.
  - Use a simple plain-React combobox; don't pull in headlessui to keep
    the dep surface small.

CREATE src/components/Sidebar.tsx:
  - Header: title "Conversations" + small "+ New" button that opens
    ReportPicker (so users can start a new conversation without going
    through the chat panel's empty state).
  - Lists conversations sorted by lastUsedAt DESC. Use
    Object.values(conversations).sort((a,b) => b.lastUsedAt - a.lastUsedAt).
  - Each row carries data-testid="sidebar-row" and data-rid={reportId} for
    deterministic e2e selection.
  - Each row:
    - report_id (truncated, monospace)
    - last-message preview: messages.at(-1)?.text || "(no messages yet)"
      truncated to ~60 chars; if isStreaming render "typing…" instead.
    - relative time (e.g. "2m ago", "Just now") from lastUsedAt — keep
      a small dependency-free formatter inline.
    - unread indicator on the right:
        if conv.unreadCount > 0:
          if unreadCount === 1 → green dot (8px, accent2)
          else → small pill with the count (e.g. "3")
      data-testid="unread-badge" so e2e can assert.
    - row background highlights when activeReportId === reportId.
    - if isStreaming, a small spinner SVG before the report id.
  - Click row → store.selectReport(rid). DO NOT cancel any in-flight
    stream on the previously active conversation.
  - Empty-state (zero conversations): a faded hint
    "Your conversations will appear here. Click + New to start." This
    hint is what new users see in the sidebar while the LandingScreen
    occupies the chat panel.

CREATE src/components/LandingScreen.tsx:
  - Centered card on the chat-panel background. Two variants driven by
    Object.keys(conversations).length:
      a) New user (zero conversations):
         - Big welcome heading: "ConvFinQA Agent"
         - Subhead: "Multi-turn financial QA over SEC filings, streamed
           live."
         - 3-step quick-start, numbered:
           1. Pick a report — every conversation is anchored to one report
              (a single page from a 10-K / 10-Q).
           2. Ask a question — type freely or click one of the gold
              questions for that report. Answers stream stage-by-stage.
           3. Run all gold — kick off the full evaluation set for the
              report and watch ✓/✗ accumulate.
         - Primary CTA button "+ New conversation" — opens ReportPicker.
         - Optional small footnote: "Your conversation list will live in
           the sidebar on the left."
      b) Returning user (≥1 conversation, none active):
         - Compact "Pick a conversation" heading
         - Single line "Choose one from the sidebar, or start a new one."
         - Same primary CTA "+ New conversation".
  - Use data-testid="landing-screen" and data-variant="new"|"returning"
    so e2e tests can assert which variant rendered.

CREATE src/components/ChatPanel.tsx:
  - If !activeReportId → render <LandingScreen />. (Don't render Composer
    or any chat-area chrome.)
  - For activeReportId: render a small panel header with the report id on
    the left and a "Reset history" icon button on the right (trash / refresh
    icon — plain SVG is fine, no icon lib). Tooltip: "Clear conversation
    history (next question goes to agent with no prior context)".
  - The reset button is disabled while conversation.isStreaming is true and
    when there are no messages to clear.
  - Click → confirm via window.confirm("Reset this conversation? The agent
    will answer the next question with no prior history.") → call
    store.resetConversation(rid).
  - Below the header: render <MessageBubble /> for each message, then
    <Composer />.
  - Auto-scroll to bottom on new message or new event.
  - Mark the reset button with data-testid="reset-conversation" so the e2e
    test can drive it deterministically.

CREATE src/components/MessageBubble.tsx:
  - role=user → right-aligned green bubble (bubbleUser), text only.
  - role=assistant:
    - while status==='streaming':
      - small chips for each stage (gray pill if started, accent if has
        output). Tool traces (calculator) shown as collapsible <details>.
    - when status==='done':
      - render answer text big.
      - if goldAnswer set → render ✓ green / ✗ red and the gold value.
  - role=system → italic centered muted line.

CREATE src/components/Composer.tsx:
  - <textarea> bound to local state (Enter→send, Shift+Enter→newline).
  - Send button: disabled if conversation.isStreaming OR no activeReportId.
  - "Run all gold" button: same disable rule. Calls store.runAllGold.
  - Above the input: question suggestion chips for the report
    (top 5 gold questions); clicking a chip pre-fills the textarea AND
    passes its gold answer to ask() so ✓/✗ shows.

EDIT src/App.tsx:
  - On mount: store.loadReports().
  - Layout: grid; left col 320px = Sidebar; right col = TopBar over ChatPanel.
  - The TopBar always renders. If !activeReportId, TopBar shows a placeholder
    label ("No conversation selected") and a "+ New conversation" button.
  - If !activeReportId, ChatPanel internally delegates to <LandingScreen />,
    which picks the new-user vs returning-user variant from
    Object.keys(conversations).length.
  - The "+ New conversation" CTA from any of {LandingScreen, Sidebar header,
    TopBar} opens the same ReportPicker (lift `pickerOpen` state to App.tsx
    or to the store — pick whichever is simpler; a single boolean in the
    store keeps it observable from any component).

Task 8 — Playwright config + concurrent test:
CREATE frontend/playwright.config.ts:
  webServer: [
    {
      command: 'cd .. && uv run python cli.py serve --port 8765',
      url: 'http://127.0.0.1:8765/healthz',
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: 'npm run build && npm run preview -- --port 4173',
      url: 'http://127.0.0.1:4173/',
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
  use: { baseURL: 'http://127.0.0.1:4173' }
  Note: preview must hit port 8765 not 8000 — set VITE_API_BASE in env or use
  a runtime base URL pulled from window.location query (?api=...). Simplest:
  in vite.config make the proxy target read from process.env.API_BASE so
  preview proxy hits 8765 in tests, 8000 normally.
  ALTERNATIVELY: in Playwright tests, don't use preview proxy — just set the
  api base URL via a localStorage seed before page.goto:
      await page.addInitScript(() =>
        window.localStorage.setItem('convfinqa.apiBase', 'http://127.0.0.1:8765'));
  and have api.ts read that key on startup. Pick this approach — simpler.

CREATE frontend/tests/e2e/concurrent.spec.ts:
  test('runs gold questions for two reports concurrently', async ({browser}) => {
    const ctxA = await browser.newContext();
    const ctxB = await browser.newContext();
    const a = await ctxA.newPage();
    const b = await ctxB.newPage();
    for (const p of [a, b]) {
      await p.addInitScript(() =>
        window.localStorage.setItem('convfinqa.apiBase', 'http://127.0.0.1:8765'));
    }

    // Pick deterministic small reports — find ones with ≤3 gold questions to
    // keep total runtime sane. Query via a fixture call to /reports/{rid}/questions.
    const REPORT_A = 'Single_VLO/2011/page_126.pdf-1';
    const REPORT_B = 'Single_AES/2003/page_168.pdf-1';

    await a.goto('/');
    await b.goto('/');

    // Pick reports
    for (const [page, rid] of [[a, REPORT_A], [b, REPORT_B]] as const) {
      await page.getByRole('button', {name: /change report/i}).click();
      await page.getByPlaceholder(/search reports/i).fill(rid);
      await page.getByRole('button', {name: rid}).click();
    }

    // Kick off Run-all on both AT THE SAME TIME
    await Promise.all([
      a.getByRole('button', {name: /run all gold/i}).click(),
      b.getByRole('button', {name: /run all gold/i}).click(),
    ]);

    // Both should show 2+ messages within 5s — proving streams are interleaving
    await expect.poll(async () =>
      (await a.locator('[data-role="assistant-message"]').count()) >= 1
      && (await b.locator('[data-role="assistant-message"]').count()) >= 1,
      {timeout: 10_000}).toBeTruthy();

    // Wait for both to fully complete
    for (const page of [a, b]) {
      await expect.poll(async () =>
        await page.locator('[data-streaming="false"][data-final="true"]').count() >= 2,
        {timeout: 180_000}).toBeTruthy();
    }

    // Assert most/all answers correct (goldClass='match' set on bubbles
    // when looseNumericMatch fired). ≥80% threshold to tolerate occasional
    // model variance.
    for (const [page, label] of [[a, 'A'], [b, 'B']] as const) {
      const total = await page.locator('[data-gold]').count();
      const matches = await page.locator('[data-gold="match"]').count();
      expect(total, `${label} should have at least one gold-marked answer`)
        .toBeGreaterThan(0);
      expect(matches / total, `${label} accuracy`).toBeGreaterThanOrEqual(0.8);
    }
  });

CREATE additional test in concurrent.spec.ts (or sibling file
frontend/tests/e2e/reset.spec.ts) — proves resetConversation clears history:

  test('reset conversation drops server-side history', async ({page, request}) => {
    await page.addInitScript(() =>
      window.localStorage.setItem('convfinqa.apiBase', 'http://127.0.0.1:8765'));
    const RID = 'Single_VLO/2011/page_126.pdf-1';
    await page.goto('/');

    // Pick the report and ask a turn that establishes history.
    await page.getByRole('button', {name: /change report/i}).click();
    await page.getByPlaceholder(/search reports/i).fill(RID);
    await page.getByRole('button', {name: RID}).click();

    // Ask question 0 (a "Number" turn — directly retrievable).
    await page.getByRole('textbox').fill('what were the value of futures 2013 long?');
    await page.getByRole('button', {name: /^send$/i}).click();
    await expect(page.locator('[data-final="true"]').first()).toBeVisible({timeout: 60_000});

    // Read the session_id from the store via window for a backend assertion.
    const sidBefore = await page.evaluate(() => {
      const raw = window.localStorage.getItem('convfinqa.v1');
      const s = raw ? JSON.parse(raw).state : null;
      return s?.conversations?.[arguments[0]]?.sessionId ?? null;
    }, RID);
    expect(sidBefore).not.toBeNull();

    // Confirm server has a non-empty history for this session.
    const before = await request.get(`http://127.0.0.1:8765/sessions/${sidBefore}`);
    expect((await before.json()).n_turns).toBeGreaterThan(0);

    // Trigger reset; auto-accept the confirm dialog.
    page.once('dialog', d => d.accept());
    await page.getByTestId('reset-conversation').click();

    // Chat panel must be empty.
    await expect(page.locator('[data-role="assistant-message"]')).toHaveCount(0);

    // The old session must be gone server-side (404), and a fresh one is
    // created lazily on the next ask. Verify both.
    const afterDelete = await request.get(`http://127.0.0.1:8765/sessions/${sidBefore}`);
    expect(afterDelete.status()).toBe(404);

    // Ask a context-dependent question; with empty history the agent must
    // not be able to answer "what is the sum?" correctly. We don't assert
    // a specific failure mode (model-dependent) — we only assert that the
    // server-side n_turns went 0 → 1 for the *new* session.
    await page.getByRole('textbox').fill('what is the sum?');
    await page.getByRole('button', {name: /^send$/i}).click();
    await expect(page.locator('[data-final="true"]').last()).toBeVisible({timeout: 60_000});

    const sidAfter = await page.evaluate(() => {
      const raw = window.localStorage.getItem('convfinqa.v1');
      const s = raw ? JSON.parse(raw).state : null;
      return s?.conversations?.[arguments[0]]?.sessionId ?? null;
    }, RID);
    expect(sidAfter).not.toBe(sidBefore);
    const after = await request.get(`http://127.0.0.1:8765/sessions/${sidAfter}`);
    const body = await after.json();
    expect(body.n_turns).toBe(1);
    // The single turn the new session saw is the post-reset question.
    expect(body.history[0].question).toBe('what is the sum?');
  });

CREATE frontend/tests/e2e/landing.spec.ts — proves landing screen variants:

  test('new user lands on the welcome quick-start', async ({page}) => {
    await page.addInitScript(() => {
      window.localStorage.clear();
      window.localStorage.setItem('convfinqa.apiBase', 'http://127.0.0.1:8765');
    });
    await page.goto('/');

    const landing = page.getByTestId('landing-screen');
    await expect(landing).toBeVisible();
    await expect(landing).toHaveAttribute('data-variant', 'new');
    await expect(landing).toContainText(/pick a report/i);
    await expect(landing).toContainText(/ask a question/i);
    await expect(landing).toContainText(/run all gold/i);

    // Sidebar empty-state hint visible when no conversations exist.
    await expect(page.getByText(/your conversations will appear here/i)).toBeVisible();

    // Primary CTA opens the picker.
    await page.getByRole('button', {name: /new conversation/i}).first().click();
    await expect(page.getByPlaceholder(/search reports/i)).toBeVisible();
  });

  test('returning user with no active selection sees compact variant', async ({page}) => {
    // Seed a conversation in localStorage.
    await page.addInitScript(() => {
      window.localStorage.setItem('convfinqa.apiBase', 'http://127.0.0.1:8765');
      window.localStorage.setItem('convfinqa.v1', JSON.stringify({
        state: {
          activeReportId: null,
          conversations: {
            'Single_VLO/2011/page_126.pdf-1': {
              reportId: 'Single_VLO/2011/page_126.pdf-1',
              sessionId: null, messages: [],
              lastUsedAt: Date.now(), lastReadAt: Date.now(),
              unreadCount: 0, isStreaming: false,
            },
          },
          reports: [],
        },
        version: 0,
      }));
    });
    await page.goto('/');

    const landing = page.getByTestId('landing-screen');
    await expect(landing).toBeVisible();
    await expect(landing).toHaveAttribute('data-variant', 'returning');
    // The seeded conversation must appear in the sidebar.
    await expect(page.getByText('Single_VLO/2011/page_126.pdf-1')).toBeVisible();
  });

CREATE frontend/tests/e2e/unread.spec.ts — proves unread badge mechanics:

  test('unread badge appears on inactive conversation, clears on click', async ({page}) => {
    await page.addInitScript(() =>
      window.localStorage.setItem('convfinqa.apiBase', 'http://127.0.0.1:8765'));
    const RID_A = 'Single_VLO/2011/page_126.pdf-1';
    const RID_B = 'Single_AES/2003/page_168.pdf-1';
    await page.goto('/');

    // Open A, ask a quick number question, wait for first answer.
    await page.getByRole('button', {name: /new conversation/i}).first().click();
    await page.getByPlaceholder(/search reports/i).fill(RID_A);
    await page.getByRole('button', {name: RID_A}).click();
    await page.getByRole('textbox').fill('what were the value of futures 2013 long?');
    await page.getByRole('button', {name: /^send$/i}).click();
    await expect(page.locator('[data-final="true"]').first()).toBeVisible({timeout: 60_000});

    // Open B and start a question. While B is streaming, switch back to A.
    await page.getByRole('button', {name: /new conversation/i}).first().click();
    await page.getByPlaceholder(/search reports/i).fill(RID_B);
    await page.getByRole('button', {name: RID_B}).click();
    await page.getByRole('textbox').fill('what was the value of physical contracts 2013 long?');
    await page.getByRole('button', {name: /^send$/i}).click();

    // Switch focus to A while B is mid-stream.
    await page.getByText(RID_A).click();

    // Wait for B's stream to finish in the background — its sidebar row
    // should now show the unread badge.
    const bRow = page.getByText(RID_B).locator('xpath=ancestor::*[@data-testid="sidebar-row"][1]');
    const badge = bRow.getByTestId('unread-badge');
    await expect(badge).toBeVisible({timeout: 90_000});

    // Click B → badge clears.
    await bRow.click();
    await expect(badge).toHaveCount(0);
  });
```

### Per-task pseudocode (only the load-bearing bits)

```ts
// src/api.ts — streamAsk skeleton
export async function streamAsk(args: {
  apiBase: string;
  sessionId: string;
  question: string;
  signal: AbortSignal;
  onEvent: (e: SSEEvent) => void;
}): Promise<void> {
  const {apiBase, sessionId, question, signal, onEvent} = args;
  await fetchEventSource(`${apiBase}/sessions/${sessionId}/ask/stream`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({question}),
    signal,
    openWhenHidden: true,
    async onopen(res) {
      // GOTCHA: if 404, throw so caller recreates session
      if (res.status === 404) throw new SessionGoneError();
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    },
    onmessage(ev) {
      if (!ev.data) return;
      onEvent(JSON.parse(ev.data) as SSEEvent);
    },
    onerror(err) {
      throw err; // stops library's auto-retry; caller handles
    },
  });
}
```

```ts
// src/store.ts — ask() reducer (the heart)
ask: async (rid, question, gold) => {
  const conv = get().conversations[rid];
  if (!conv || conv.isStreaming) return;

  const userMsg: Message = {id: nanoid(), role: 'user', text: question, status: 'done'};
  const asstMsg: Message = {
    id: nanoid(), role: 'assistant', text: '',
    goldAnswer: gold, status: 'streaming', stages: {}, tools: [],
  };
  patchConversation(rid, c => {
    c.isStreaming = true;
    c.lastUsedAt = Date.now();
    c.messages.push(userMsg, asstMsg);
  });

  let sessionId = await ensureSession(rid);
  const ac = new AbortController();
  try {
    await streamAsk({
      apiBase: getApiBase(), sessionId, question, signal: ac.signal,
      onEvent: ev => patchAssistant(rid, asstMsg.id, ev),
    });
  } catch (e) {
    if (e instanceof SessionGoneError) {
      // Recreate once
      await invalidateSession(rid);
      sessionId = await ensureSession(rid);
      patchConversation(rid, c => c.messages.push({
        id: nanoid(), role: 'system',
        text: 'Session expired — starting a fresh one.', status: 'done',
      }));
      try {
        await streamAsk({apiBase: getApiBase(), sessionId, question,
          signal: ac.signal, onEvent: ev => patchAssistant(rid, asstMsg.id, ev)});
      } catch (e2) {
        patchAssistant(rid, asstMsg.id, {event:'error', error:String(e2)});
      }
    } else {
      patchAssistant(rid, asstMsg.id, {event:'error', error:String(e)});
    }
  } finally {
    patchConversation(rid, c => { c.isStreaming = false; });
  }
}
```

### Integration Points
```yaml
ROUTES:
  - app.py: add CORSMiddleware (Task 1)

CONFIG:
  - frontend/.env.example: VITE_API_BASE=http://127.0.0.1:8000  (read by api.ts)
  - In Playwright tests, override via localStorage seed so tests target 8765.

DEV WORKFLOW:
  - Terminal 1: uv run python cli.py serve --reload
  - Terminal 2: cd frontend && npm run dev
  - Browser: http://localhost:5173

PROD-LIKE SERVE (optional, not required by this PRP):
  - cd frontend && npm run build
  - Mount frontend/dist on FastAPI via app.mount("/", StaticFiles(...,
    html=True), name="ui") — but only do this if user asks; out of scope here.
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Backend
uv run ruff check . --fix
uv run mypy app.py pydantic_agent.py cli.py

# Frontend
cd frontend && npm run typecheck
cd frontend && npm run build
```

### Level 2: Unit / Integration tests
```bash
# Backend (must still pass after CORS change)
uv run pytest --ignore=tests/test_agent.py -v

# (test_app_cors.py validates the OPTIONS preflight succeeds)
```

### Level 3: Frontend e2e (the load-bearing test)
```bash
cd frontend
# First run only:
npx playwright install chromium

# This brings up uvicorn + vite preview, runs the concurrent test:
npm run test:e2e
```

### Level 4: Manual smoke
```bash
# Terminal 1
uv run python cli.py serve --reload
# Terminal 2
cd frontend && npm run dev
# Browser: http://localhost:5173
#  1) Click "Change report", search "VLO", pick Single_VLO/2011/page_126.pdf-1
#  2) Click a suggested gold question — watch stage chips populate, then answer
#     appear with green ✓.
#  3) Click "Run all gold" — sequence completes with ≥4/5 ✓.
#  4) Click "Change report", pick another report — verify left sidebar now lists
#     two conversations and can be switched between.
#  5) Start Run-all on report A, switch to report B, start Run-all there. Both
#     progress visible in sidebar (streaming spinner on each).
#  6) Hard reload — both conversations restored from localStorage.
#  7) Click "Reset history" on report A. Chat clears. Ask a context-dependent
#     question like "what is the sum?". The agent should NOT be able to
#     resolve it (no history → no prior values to reference). Confirm via
#     `curl http://localhost:8000/sessions/<new sid>` that n_turns == 1 and
#     history[0].question matches what you just asked.
#  8) Reset button must be disabled while a stream is running on that
#     conversation (verify visually).
#  9) Clear localStorage and reload — the new-user landing screen with the
#     3-step quick-start must render. Click "+ New conversation" → picker
#     opens. After selecting a report, the landing screen disappears and
#     the chat panel takes over.
# 10) Unread badge: on report A, ask a question, wait for answer. While
#     report A is open, the active row never shows a badge. Switch to a
#     fresh report B, ask a question. While B is streaming, click on A in
#     the sidebar. When B's answer eventually lands, B's sidebar row must
#     show a green dot. Click B → dot clears.
# 11) Sidebar order: take actions on multiple reports and verify the most
#     recently used one is always at the top of the sidebar.
```

## Final validation Checklist
- [ ] `uv run pytest --ignore=tests/test_agent.py` passes (incl. test_app_cors.py)
- [ ] `uv run ruff check .` clean
- [ ] `uv run mypy app.py pydantic_agent.py` clean
- [ ] `cd frontend && npm run typecheck` clean
- [ ] `cd frontend && npm run build` succeeds
- [ ] `cd frontend && npm run test:e2e` passes (concurrent + reset + landing + unread specs)
- [ ] Manual smoke (steps 1-11 above) all work
- [ ] Fresh-localStorage load shows the new-user landing variant; reload
      after one conversation exists shows the returning-user variant.
- [ ] Unread badge appears on inactive rows and clears on click.
- [ ] Sidebar order is `lastUsedAt DESC` at all times.
- [ ] Two simultaneous Run-all sessions visibly progress in parallel (check
      DevTools Network — two concurrent EventSource POSTs)
- [ ] Sidebar persists after page reload
- [ ] Switching report mid-stream does NOT cancel the originating stream
- [ ] No console errors in browser

---

## Anti-Patterns to Avoid
- ❌ Don't use native EventSource — it's GET-only and won't accept the JSON body.
- ❌ Don't serialize all asks through one global "isStreaming" flag — that
  breaks the concurrency requirement. Per-conversation flag only.
- ❌ Don't try to use one session for multiple reports — sessions are
  immutable (server enforces it), and you'll just fight the model.
- ❌ Don't reimplement `numeric_match` differently in TS — port the cli's
  exact algorithm so verdicts match.
- ❌ Don't await `runAllGold` in a click handler if you want background
  progress while the user navigates — `void` it and rely on store state.
- ❌ Don't send `report_id` in the AskRequest body — the server has
  `extra="forbid"`; that field will 422.
- ❌ Don't poll the server for events — use the SSE stream you already have.
- ❌ Don't add light mode "for completeness" — out of scope and adds churn.
- ❌ Don't bundle a UI library (MUI, Chakra) — Tailwind + plain HTML is
  enough for this scope and keeps install fast.
- ❌ Don't implement "reset" by clearing only frontend messages — the
  server-side ConversationHistory must also be dropped, otherwise the next
  ask still gets the old context. Use DELETE /sessions/{sid} and clear
  sessionId so the next ask creates a fresh session.
- ❌ Don't allow reset while a stream is in flight — disable the button on
  isStreaming. Racing a DELETE with an open SSE stream produces undefined
  behavior on both sides.
- ❌ Don't increment unreadCount on the active conversation. The user is
  literally watching it stream — flagging it as "unread" is wrong.
- ❌ Don't sort the sidebar by anything other than `lastUsedAt DESC` (no
  alphabetical, no by-unread-first). Recency-only matches the WhatsApp
  pattern users expect and keeps active streams at the top where the
  user's eye already is.
- ❌ Don't gate the LandingScreen on `reports.length === 0` (the catalog
  list); gate it on `Object.keys(conversations).length`. The two are
  unrelated: catalog has ~80 reports always, conversations is "what *I*
  have asked about".

---

## Confidence Score
**8 / 10**

Rationale:
- High confidence: backend is already complete and exercised; SSE protocol is
  well-defined; React/Vite/Tailwind is well-trodden; Playwright is
  well-suited to the concurrency test.
- Risks (the −2):
  1. The Vite scaffold step is interactive (`npm create vite@latest .`) — the
     implementing agent must answer prompts. Documented but worth flagging.
  2. The Playwright concurrent test can be flaky if the model is slow or
     temperature isn't deterministic; threshold=80% is set deliberately, but
     a real run might trip on bad luck. Mitigation: cache hits via
     `.dspy_cache/` make repeated runs nearly deterministic, and the test
     uses report_ids that the existing repo has shown to score 5/5 and 4/4.
  3. Strict TS + zustand persist + immer-style updates have boilerplate the
     implementer has to get right on the first try.

If the implementing agent gets stuck, the highest-leverage actions are:
- run `npm run dev` and tail the browser console;
- run `uv run python cli.py serve --reload` so backend changes propagate;
- and if SSE silently hangs, check that fetch-event-source's `openWhenHidden`
  is set and that `onerror` re-throws (default would auto-retry indefinitely).
