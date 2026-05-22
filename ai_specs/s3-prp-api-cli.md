name: "ConvFinQA Serving Layer — FastAPI API + Interactive CLI"
description: |

## Purpose
Build a thin production serving layer over the existing optimized ConvFinQA runner:

- `app.py`: FastAPI multi-turn API
- `cli.py`: typer + questionary client over HTTP
- `pydantic_agent.py`: expose a shared per-turn primitive
- `pyproject.toml`: add missing runtime deps `fastapi`, `uvicorn`, `httpx`

Execution background lives at `.claude/commands/execute-prp.md`.

## Load-Bearing Requirements

### Offline vs production
- Offline already exists: `ConversationRunner.run_conversation(report_id, questions)` in `pydantic_agent.py`
- Production needs one-turn-at-a-time execution with server-side history
- Extract a public `async def run_turn(question, report_id, conversation) -> str`
- Both offline and API paths must call `run_turn`

### Session model
- A session is created with exactly one `report_id`
- Session `report_id` is immutable
- To switch reports, the client must create a new session
- `/sessions/{sid}/ask` must reject unknown fields like `report_id` via `extra="forbid"`

### History model
- Use existing `ConversationHistory`
- Keep prompt history in `ConversationHistory.as_text()` format
- Do not use Pydantic AI `message_history`

### Correctness standard
- The key validation is not a demo conversation
- API/CLI parity must be checked on the same held-out `conv_examples_test` dataset used by:
  - `dspy_agent.py`
  - `pydantic_agent.py`
- Comparisons must assert identical row coverage:
  - `(report_id, turn_index, question, gold_answer)`
- Validation must compare API outputs against both DSPy and Pydantic runner outputs

## Deliverables

### 1. `pyproject.toml`
Add only:
- `fastapi`
- `uvicorn`
- `httpx`

Do not add unrelated dependencies.

### 2. `pydantic_agent.py`
Refactor to expose:

```python
async def run_turn(question: str, report_id: str, conversation: ConversationHistory) -> str
```

Requirements:
- shared primitive for offline + API
- mutates `conversation` in place
- looks up document from `_DOCS`
- preserves current `run_conversation` behavior

### 3. `app.py`
Implement FastAPI endpoints:

```text
GET    /healthz
GET    /reports?q=&limit=
GET    /reports/{rid}
GET    /reports/{rid}/questions
POST   /sessions
GET    /sessions/{sid}
POST   /sessions/{sid}/ask
DELETE /sessions/{sid}
```

Behavior:
- reports list must be intersection of `_DOCS.keys()` and `qa_data.report_id`
- session store is in-memory MVP
- require `uvicorn --workers 1`
- use per-session `asyncio.Lock` around full `/ask` execution
- TTL eviction for idle sessions, default 30 min, configurable for tests
- TTL cleanup must remove both session and lock
- instrument FastAPI with Logfire when token is present

### 4. `cli.py`
Commands:

```text
cli
cli ask [--report ID --question "..."]
cli reports [--q SUBSTR]
cli serve [--host 0.0.0.0 --port 8000]
```

Behavior:
- default `cli` enters interactive loop
- flow: pick report -> show gold questions -> ask preset or free-form -> continue / change report / quit
- “Change report” must create a new session
- all CLI question answering goes through HTTP, never in-process runner calls
- use one reusable `httpx.Client`
- use long timeout, e.g. `120.0`
- handle `questionary` returning `None` as abort

### 5. Validation helpers
Add a small API evaluation helper, e.g. `api_eval.py` or equivalent.

It should:
- drive the live API across every conversation in `conv_examples_test`
- write an API predictions CSV with shape:
  - `report_id, turn_index, question, gold_answer, pred_answer, correct`
- compare against:
  - DSPy `predictions.csv`
  - Pydantic `pydantic_predictions.csv`
- fail loudly on row-coverage drift

## Tests

### `tests/test_api.py`
Use:
- `fastapi.testclient.TestClient`
- `Agent.override(model=TestModel())` for all four sub-agents

Cover:
- health check
- report listing / details / questions
- session create/get/delete
- session report binding immutability
- reject extra fields on `/ask`
- new session starts empty
- same report, two sessions, isolated histories
- TTL eviction
- `run_turn` is the path exercised by `/ask`

### `tests/test_cli.py`
Use:
- `typer.testing.CliRunner`
- `httpx.MockTransport`

Cover:
- `reports`
- one-shot ask
- interactive flow
- “Change report” creates new session
- CLI uses HTTP responses, not in-process shortcuts

## Success Criteria
- [ ] `uv run uvicorn app:app --workers 1 --port 8000` starts cleanly
- [ ] `uv run pytest tests/test_api.py -v` passes
- [ ] `uv run pytest tests/test_cli.py -v` passes
- [ ] `uv run pytest tests/test_pydantic_agent.py -v` still passes
- [ ] `ruff check app.py cli.py tests/test_api.py tests/test_cli.py pydantic_agent.py --fix` is clean
- [ ] `ConversationRunner.run_conversation(...)` and `/ask` both call `pydantic_agent.run_turn(...)`
- [ ] API validation runs on the same `conv_examples_test` dataset as `dspy_agent.py` and `pydantic_agent.py`
- [ ] API predictions CSV matches the existing runner artifact schema
- [ ] API-vs-DSPy and API-vs-Pydantic comparisons enforce identical `(report_id, turn_index, question, gold_answer)` coverage
- [ ] CLI validation goes through the live API

## Relevant Repo Context

### `pydantic_agent.py`
Reuse:
- `ConversationRunner`
- `ConversationHistory`
- `_DOCS`
- module-level agents
- `evaluate(...)`
- `compare_runs(...)`

Mirror the existing parity standard rather than inventing a looser one.

### `dspy_agent.py`
Reuse:
- `qa_data`
- `_DOCS`
- `conv_examples_test`
- `numeric_match`
- existing predictions artifact shape

### `pyproject.toml`
Current repo state already includes:
- `typer`
- `questionary`
- `logfire`
- `pydantic`

It does not include:
- `fastapi`
- `uvicorn`
- top-level `httpx`

## Implementation Notes
- `ConversationRunner` is for offline batch eval; API should not instantiate it per request
- hold the session lock for the entire `run_turn(...)` call
- cancel lifespan background tasks cleanly on shutdown
- keep the app as a thin transport layer; do not duplicate agent orchestration
- list only reports that have both a document and at least one gold question
- the in-memory store is single-instance only; document that `--workers 1` is required

## Suggested Validation Flow

1. Run existing runner evaluations if needed:
   - `RUN_GEPA= uv run python dspy_agent.py`
   - `uv run python pydantic_agent.py`
2. Start API:
   - `uv run uvicorn app:app --workers 1 --port 8000`
3. Run API dataset evaluation helper against `conv_examples_test`
4. Compare API predictions to DSPy and Pydantic predictions
5. Run unit tests and Ruff

## References
- `pydantic_agent.py`
- `dspy_agent.py`
- `tests/test_pydantic_agent.py`
- `PRPs/prp-pydantic-agent.md`
- `pyproject.toml`
- FastAPI lifespan/testing docs
- Pydantic AI testing docs
- Logfire FastAPI integration docs
- Questionary autocomplete docs
- Typer callback docs
- HTTPX client docs
