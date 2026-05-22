name: "ConvFinQA Project Streamlining — Adopt `src/` Layout and Decompose Monoliths"
description: |

## Purpose

Restructure the ConvFinQA-agent repository from its current "9 .py files at root, including two ~1500-line monoliths" layout into a professional `src/convfinqa/` package with single-responsibility modules under 300 lines each. The end state matches how a top-tier AI/ML team would lay out the same problem: every concept has one home, imports form a DAG (no cycles), backends are swappable, and scripts are dumb entry points that delegate into the package.

This is **not** a feature change. Behaviour, accuracy numbers, CLI flags, env vars, and HTTP endpoints all stay identical. The user-visible deltas are:

- Imports inside the codebase: `from config import settings` → `from convfinqa.config import settings`
- Run commands: `uv run python pydantic_agent.py` → `uv run convfinqa-eval` (or the equivalent `python -m scripts.evaluate`)
- The two ~1500-line agent files cease to exist as monoliths

A coding agent executing this PRP MUST validate after every phase. The two monolith decompositions (Phase 3) are the riskiest — they will be split across many small commits and each split MUST keep the test suite + an end-to-end smoke run of `pydantic_agent` green before moving on.

---

## Current State (as of 2026-05-18)

Phases 1, 2, 3, and most of Phase 4 have landed. Root `.py` files have all been deleted; every Python module now lives under `src/convfinqa/` (the package), `scripts/` (entry points), or `tests/`. All 56 tests pass; `ruff check src/ scripts/ tests/` is clean.

The `prompt_optim/` package and `src/convfinqa/optimization/harness.py` referenced earlier in this PRP have since been deleted — they were a half-finished experiment that never shipped (v2 came from GEPA). See `PRPs/s5-prompt-optimisation.md` for historical context, but treat it as obsolete.

**Still outstanding** (success criterion `No file in src/convfinqa/ exceeds 400 lines`):
- `src/convfinqa/optimization/gepa.py` — 494 LOC
- `src/convfinqa/serving/app.py` — 489 LOC
- `src/convfinqa/backends/dspy.py` — 458 LOC

These three remain modestly oversize and would need further sub-module decomposition (e.g. split `gepa.py` into `gepa/training.py` + `gepa/scoring.py`; split `serving/app.py` per Task 4.1 into `sessions.py` + `routes_eval.py` + a slimmer `app.py`).

**Doc consolidation (Task 4.7) not done**: AGENTS.md is still the architecture doc; PRPs/ has not been renamed to docs/decisions/; README.md still references some legacy commands.

**Historical (now stale)**:

**2026-05-16 quick-win pass completed** (Phase 4 prelude, subset of items below):
- `evaluation/` and `runs/` added to `.gitignore`.
- `cli.py` collapsed: root is now a 5-line re-export of `convfinqa.serving.cli.cli_app`; `tests/test_cli.py` repointed; 4 tests still pass.
- `prompt_optim_v2.py` collapsed: root is now a 5-line re-export of `convfinqa.optimization.harness.main`; no test importers existed.
- `mcp/` folder deleted (was unreferenced).
- Full pytest still green (85/85).
- `agent.py` kept as-is — see correction in item 4 below.

What is done:

- `src/convfinqa/` package exists with the target directory structure (`data/`, `pipeline/`, `backends/`, `evaluation/`, `optimization/`, `prompts/`, `serving/`).
- `config.py`, `evaluator/`, `prompts/`, `data.py`, `data_scope.py` have all moved into the package (Phase 1 ✅).
- `data/schemas.py`, `pipeline/stages.py`, `pipeline/tools.py`, `pipeline/wire_format.py` exist as real modules (Phase 2 ✅).
- `scripts/{evaluate,evaluate_api,optimize,serve}.py` exist as thin entry points; `[project.scripts]` is wired up.

What is **not** done (still left for Phase 3/4):

- `dspy_agent.py` (1290 LOC) and `pydantic_agent.py` (1512 LOC) are still monoliths. The package modules that should own their contents (`backends/dspy.py`, `backends/pydantic.py`, `pipeline/runner.py`, `pipeline/prompts_loader.py`, `evaluation/runner.py`, `evaluation/reporting.py`, `evaluation/joining.py`, `optimization/gepa.py`) are 5–15-line shims that do `from <root_monolith> import ...`. The direction must be inverted.
- `app.py` (493 LOC), `cli.py` (333 LOC), `api_eval.py` (407 LOC), `prompt_optim_v2.py` (279 LOC) are still root implementations. `src/convfinqa/serving/app.py` and `src/convfinqa/evaluation/api_runner.py` are also shims pointing back at root.
- Tests still import the root files directly: `tests/test_agent.py` does `import agent`; `test_cli.py` does `import cli`; `test_api.py` and `test_app_cors.py` do `import app`; `test_pydantic_agent.py` does `import pydantic_agent as pa`; `test_api_eval.py` likely uses `api_eval` directly. These need to repoint at `convfinqa.*` before the root files can be deleted.

### Additional cleanup items surfaced during review

These are not blocking the migration but are quick wins that should land alongside Phase 4 (or earlier):

1. **True duplicates** — `cli.py` and `src/convfinqa/serving/cli.py` are two complete copies of the same 333-line file (diff is ~3 hunks of `from api_eval import ...` → `from convfinqa.data.loader import ...`). Same situation for `prompt_optim_v2.py` and `src/convfinqa/optimization/harness.py`. Phase 4 must finish the move (delete the root file, leave a 2-line re-export if any test still imports it) rather than treating them as still-in-progress shims.
2. **`evaluation/` is checked into git** — AGENTS.md and this PRP both claim it is gitignored, but `.gitignore` does not list it. Phase 4 should add `evaluation/` to `.gitignore` and run `git rm --cached evaluation/*.csv evaluation/*.html`.
3. **`mcp/` folder deleted (2026-05-16)** — was unreferenced. The previous target tree listed it as `← unchanged`; that line has been removed.
4. **`agent.py`** (139 LOC) — **correction (2026-05-16):** despite its "compatibility layer" docstring, this file is a full standalone implementation (defines `ConvFinQAOrchestrator`, `AgentResponse`, `ConversationHistory`, `numeric_match`, `sample_records`, `serialize_document`, `run_record`) used only by `tests/test_agent.py`. None of these symbols are exported from `convfinqa.backends.dspy`. Deleting it requires either (a) accepting that `tests/test_agent.py` covers a simple test-only orchestrator that has no production counterpart and should move under `tests/_fixtures/`, or (b) building real equivalents in `convfinqa.backends.dspy` and porting `test_agent.py` to those. **Leave `agent.py` alone for now** — it is not a duplicate or a shim.

---

Read this whole document. Pay extra attention to:

- The current import graph, which has at least three load-order dependencies (`config` → `dspy_agent` → `pydantic_agent` → `api_eval`)
- The `dspy.LM(deepseek/...)` constructor at module-import time in `dspy_agent.py` — `DEEPSEEK_API_KEY` MUST be in `os.environ` before that import fires
- The `PROMPTS_OVERLAY_PATH` external contract — the prompt-optimisation harness writes a JSON file and expects `pydantic_agent.run_turn` to consume it; don't break that
- The frontend ↔ backend Vite proxy contract — every `/healthz`, `/reports`, `/sessions`, `/eval` prefix in `frontend/vite.config.ts` MUST still match a backend route post-refactor

---

## Goal

Land a four-phase refactor that ends with this tree:

```
ConvFinQA-agent/
├── src/convfinqa/                ← the package (everything inside is importable)
│   ├── __init__.py
│   ├── config.py                 ← from root config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             ← from root data.py (cleaned of globals())
│   │   ├── schemas.py            ← ConvExample / QAPair / ConversationHistory
│   │   └── scope.py              ← from root data_scope.py
│   ├── pipeline/                 ← framework-agnostic core
│   │   ├── __init__.py
│   │   ├── stages.py             ← TriageOut / PreprocessOut / RetrievedValues / CalcOut
│   │   ├── runner.py             ← run_turn / stream_turn orchestration
│   │   ├── tools.py              ← calculator tools (add / subtract / ...)
│   │   └── wire_format.py        ← _render_chat_inputs (DSPy ChatAdapter shim)
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── dspy.py               ← DSPy bindings only
│   │   └── pydantic.py           ← pydantic-ai bindings only
│   ├── prompts/                  ← unchanged interface, just relocated
│   │   ├── __init__.py
│   │   ├── v1.py
│   │   └── v2.py
│   ├── evaluation/               ← was top-level evaluator/ + eval bits of the monoliths
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── cache.py
│   │   ├── runner.py             ← _evaluate_version, ConversationRunner
│   │   ├── api_runner.py         ← evaluate_api logic
│   │   ├── reporting.py          ← write_predictions_html, print_accuracy_table
│   │   └── joining.py            ← analyze_predictions, _write_joined_predictions
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── gepa.py               ← GEPA training half of dspy_agent.py
│   │   └── harness.py            ← prompt_optim_v2.py
│   └── serving/
│       ├── __init__.py
│       ├── app.py                ← FastAPI app construction
│       ├── sessions.py           ← in-memory session store + lifespan
│       ├── routes_eval.py        ← /eval/runs endpoints
│       └── cli.py                ← Typer CLI
├── scripts/                      ← thin entry points only
│   ├── evaluate.py
│   ├── evaluate_api.py
│   ├── optimize.py
│   └── serve.py
├── tests/                        ← unchanged file names; imports updated
├── docs/                         ← NEW: consolidate dataset.md, AGENTS.md content
├── data/                         ← unchanged (dataset JSON, not code)
├── evaluation/                   ← unchanged (gitignored runtime outputs)
├── runs/                         ← unchanged (gitignored GEPA artifacts)
├── frontend/                     ← unchanged
├── pyproject.toml                ← scripts entry points added
├── README.md
├── CLAUDE.md
└── AGENTS.md
```

## Why

- The two ~1500-line files (`pydantic_agent.py` 1504 LOC, `dspy_agent.py` 1290 LOC) are unreviewable. Every reviewer who touches them has to scroll past 1000+ lines of unrelated code to find the diff. Today's session caught two latent bugs (`pydantic_predictions_joined.csv` never being written; `_REQUIRED_PRED_COLUMNS` schema drift) that smaller modules would have surfaced earlier.
- Imports inside the monoliths form an undocumented dependency order (config → dspy_agent → pydantic_agent → api_eval) that requires an `# noqa: E402` and a long comment to explain. A proper package with `__init__.py` files makes that order explicit and enforced.
- The pyproject.toml already declares `packages = ["src"]` and a `main = "src.main:app"` entry point. The intent has been there from day one; this PRP just executes it.
- Testing today is awkward — to import `_evaluate_version` you import `pydantic_agent`, which constructs four global `Agent` objects, an `OpenAIProvider`, runs `logfire.configure`, etc. After the refactor, `from convfinqa.evaluation.runner import evaluate_version` is one cheap import.
- Backends are swappable. Today: DSPy + Pydantic AI side-by-side. Tomorrow: `backends/openai_responses.py` plugs in without touching anything in `pipeline/`, `evaluation/`, or `serving/`.

## What

A four-phase migration, each phase landable as its own PR. Phases are ordered by risk (lowest first) so the agent can stop after any phase and still have a working repo.

| Phase | Scope | Files touched | Risk | Time |
|---|---|---|---|---|
| 1 | Package scaffolding; move small files (`config.py`, `evaluator/`, `prompts/`, `data.py`, `data_scope.py`) | ~15 | Low | 1–2 hr |
| 2 | Extract shared schemas + primitives from monoliths | ~10 | Medium | 2–3 hr |
| 3 | Split the two monoliths into per-backend modules | ~20 | High | 3–5 hr |
| 4 | Move `app.py` / `cli.py` / `prompt_optim_v2.py`; thin scripts; docs cleanup | ~12 | Medium | 2 hr |

### Success Criteria

- [ ] `src/convfinqa/` exists with the structure above; every former root `.py` file (except scripts) has a new home
- [ ] `uv run pytest tests/ -v` passes after every phase
- [ ] `uv run python pydantic_agent.py` (legacy path) OR `uv run convfinqa-eval` (new path) reproduces the same v1/v2 accuracy table as before the refactor (v1: 73.0%, v2: 77.1% with cache hit on 200 conversations)
- [ ] `uv run python -m uvicorn convfinqa.serving.app:app --workers 1 --port 8765` starts cleanly; `/healthz` returns 200; `/eval/runs` returns `["v1", "v2"]`
- [ ] `uv run python api_eval.py` (or `uv run convfinqa-eval-api`) drives the running server and writes `evaluation/api_predictions_v2.csv` correctly
- [ ] `uv run python prompt_optim_v2.py` (or its successor `uv run convfinqa-optimize`) loads `pydantic_agent.run_turn` (or its successor) and runs at least one harness phase without import errors
- [ ] `mypy src/convfinqa` exits 0 (current `mypy` config is non-strict; keep that)
- [ ] `ruff check src/ scripts/` exits 0
- [ ] No file in `src/convfinqa/` exceeds 400 lines
- [ ] `frontend/` is untouched; the Vite proxy still resolves backend routes correctly (smoke: open the UI, pick a report, send a question, watch the streamed answer)

## All Needed Context

### Current root inventory (lines)

```
pydantic_agent.py     1504   ← Monolith 1
dspy_agent.py         1290   ← Monolith 2
app.py                 491
api_eval.py            410
cli.py                 333
prompt_optim_v2.py     279
data_scope.py          106
config.py               89   ← already clean (Settings)
data.py                 41   ← uses globals()
                      ----
                      4543
```

### Current top-level folders

```
config.py + evaluator/         ← code (small, already clean)
prompts/                       ← versioned system prompts (v1.py, v2.py)
tests/                         ← pytest suite
PRPs/                          ← planning docs (this file lands here)
data/                          ← dataset JSON
evaluation/                    ← gitignored runtime outputs (predictions, joined, HTML)
runs/<GEPA_NAME>/              ← gitignored GEPA optimization artefacts
frontend/                      ← React + Vite UI
.dspy_cache/                   ← gitignored DSPy LM cache
```

### Documentation & references

```yaml
- file: AGENTS.md
  why: |
    Authoritative architecture / pipeline doc. The Vite proxy invariant, the
    `--workers 1` constraint, and the per-stage capture schema MUST survive
    the refactor unchanged.

- file: CLAUDE.md
  why: |
    Quick-start commands and key invariants. After Phase 4, the commands here
    must be updated to the new entry-point paths (scripts/* or installed CLI).

- file: PRPs/s5-prompt-optimisation.md
  why: |
    Most recent PRP — mirrors the style this PRP follows. Note its phase
    structure, success criteria, validation gates per phase. Same shape here.

- file: pyproject.toml
  why: |
    Already declares `packages = ["src"]` and a `main = "src.main:app"` script.
    Phase 1 wires up `[project.scripts]` for the new CLI entry points.

- file: config.py
  why: |
    Already a clean pydantic-settings Settings class. This is the canonical
    pattern for env-var handling — do NOT reintroduce os.environ.get() calls
    during the refactor. Every consumer reads `from convfinqa.config import settings`.

- file: evaluator/__init__.py
  why: |
    Already a clean package with `numeric_match`, `load_cached_conversations`,
    `flush_csv_atomic` exported. Phase 1 just moves it under `src/convfinqa/evaluation/`
    and updates two callers' imports.

- file: prompts/__init__.py
  why: |
    `load(version)` + `latest()` + `latest_all()` are the public API used by
    `pydantic_agent._resolve_prompts()`. Moving prompts/ under src/convfinqa/
    must preserve this interface byte-for-byte.

- doc: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
  why: |
    PyPA's authoritative explanation of why src/ layout is preferred. Key
    point: without src/, the project root is implicitly on PYTHONPATH and
    tests can pass against the working copy instead of the installed package
    — which masks broken imports.
```

### Known gotchas

```python
# CRITICAL: dspy_agent.py constructs dspy.LM(deepseek/...) at *import time*.
# That constructor reads DEEPSEEK_API_KEY directly from os.environ. The
# current code relies on `from config import settings` running first (which
# calls load_dotenv) before any `import dspy_agent` happens.
#
# After the refactor: any module that imports `convfinqa.backends.dspy`
# transitively triggers the same constructor, so `convfinqa.config` MUST
# be the first import in `src/convfinqa/__init__.py` (or `convfinqa.backends.dspy`
# imports `from convfinqa.config import settings` itself before importing dspy).

# CRITICAL: `app.py` (FastAPI) imports `run_turn` and `stream_turn` from
# pydantic_agent. The frontend depends on the streaming endpoint shape. After
# the refactor, those functions live in `convfinqa.pipeline.runner` (the
# framework-agnostic orchestration) — `app.py` just imports from the new
# location. Do not change the function signatures or yielded event shape.

# CRITICAL: `prompt_optim_v2.py` imports `pydantic_agent` and rebuilds the
# four agents with overlaid prompts via `PROMPTS_OVERLAY_PATH`. The Settings
# field `prompts_overlay_path` is the contract. Don't rename it.

# CRITICAL: data.py mutates globals():
#     globals()[f'{key}_df'] = pd.concat([...])
# This is hacky but TESTABLE because the function returns the merged DataFrame.
# In Phase 2 it gets replaced by an explicit dict return; the public function
# `training_data() -> pd.DataFrame` stays the same.

# CRITICAL: Tests currently import directly from root modules:
#     from pydantic_agent import run_turn
#     from api_eval import load_conv_examples_test
# After each phase, the tests' imports MUST be updated in lockstep. Do not
# leave any test importing a name that just moved.

# CRITICAL: The frontend `/eval/runs` endpoint (and friends) returns version
# strings like "v1", "v2" — `app.py` already has this in `_MODEL_CSV_PATTERN`.
# Don't change the URL prefix `/eval/runs` even though it now means "versions";
# the frontend depends on the literal path.

# CRITICAL: GEPA training in dspy_agent.py uses dspy's GEPA optimizer which
# expects a `Module` with specific predictor names (triage.predict, etc.).
# That entire training path stays in `convfinqa.optimization.gepa` and keeps
# the same Module structure — don't refactor the DSPy module class.
```

### Import migration map

This is the source of truth for every import that needs updating across the codebase. Phase numbers indicate when each move happens.

| Phase | Was | Becomes |
|---|---|---|
| 1 | `from config import settings` | `from convfinqa.config import settings` |
| 1 | `from evaluator import numeric_match, load_cached_conversations, flush_csv_atomic` | `from convfinqa.evaluation import numeric_match, load_cached_conversations, flush_csv_atomic` |
| 1 | `import prompts as _pkg` | `import convfinqa.prompts as _pkg` |
| 1 | `from data import training_data` | `from convfinqa.data.loader import training_data` |
| 2 | `from api_eval import ConvExample, load_conv_examples_test` | `from convfinqa.data.schemas import ConvExample` + `from convfinqa.data.loader import load_conv_examples_test` |
| 2 | `from dspy_agent import QAPair, ConversationHistory` | `from convfinqa.data.schemas import QAPair, ConversationHistory` |
| 2 | `from dspy_agent import _DOCS, qa_data, CALCULATOR_TOOLS` | `from convfinqa.data.loader import _DOCS, qa_data` + `from convfinqa.pipeline.tools import CALCULATOR_TOOLS` |
| 2 | inline `_render_chat_inputs` in `pydantic_agent.py` | `from convfinqa.pipeline.wire_format import render_chat_inputs` |
| 3 | `from pydantic_agent import run_turn, stream_turn` | `from convfinqa.pipeline.runner import run_turn, stream_turn` |
| 3 | `from pydantic_agent import _evaluate_version, print_accuracy_table` | `from convfinqa.evaluation.runner import evaluate_version` + `from convfinqa.evaluation.reporting import print_accuracy_table` |
| 3 | `from pydantic_agent import write_predictions_html, _write_joined_predictions` | `from convfinqa.evaluation.reporting import write_predictions_html` + `from convfinqa.evaluation.joining import write_joined_predictions` |
| 3 | `from dspy_agent import analyze_predictions` | `from convfinqa.evaluation.joining import analyze_predictions` |
| 4 | `from api_eval import evaluate_api, compare_api_outputs` | `from convfinqa.evaluation.api_runner import evaluate_api, compare_api_outputs` |
| 4 | `app.py` route handlers | `from convfinqa.serving.app import app` |
| 4 | `cli.py` Typer app | `from convfinqa.serving.cli import app` |

## Implementation Blueprint

### Phase 1 — Package scaffolding and small file moves

Goal: stand up `src/convfinqa/`, move the four already-clean small things (`config.py`, `evaluator/`, `prompts/`, `data.py`/`data_scope.py`), update imports, register CLI entry points in `pyproject.toml`. No business logic touched.

```yaml
Task 1.1 — Scaffold the package
CREATE src/convfinqa/__init__.py:
  - Body: just a module docstring + `from convfinqa.config import settings` so
    that any sub-module can rely on dotenv being loaded.
CREATE src/convfinqa/{data,pipeline,backends,evaluation,optimization,serving}/__init__.py:
  - Empty __init__.py for now. Phase 2 + 3 will populate.

Task 1.2 — Move config.py
MOVE config.py → src/convfinqa/config.py:
  - No content changes.
UPDATE callers (api_eval.py, pydantic_agent.py, dspy_agent.py, app.py):
  - FIND: `from config import settings`
  - REPLACE: `from convfinqa.config import settings`

Task 1.3 — Move evaluator/ package
MOVE evaluator/__init__.py → src/convfinqa/evaluation/__init__.py:
  - REPLACE the import lines inside __init__.py:
      `from evaluator.cache import ...` → `from convfinqa.evaluation.cache import ...`
      `from evaluator.metrics import ...` → `from convfinqa.evaluation.metrics import ...`
MOVE evaluator/cache.py → src/convfinqa/evaluation/cache.py
MOVE evaluator/metrics.py → src/convfinqa/evaluation/metrics.py
DELETE evaluator/ folder.
UPDATE callers (api_eval.py, pydantic_agent.py):
  - FIND: `from evaluator import ...`
  - REPLACE: `from convfinqa.evaluation import ...`

Task 1.4 — Move prompts/ package
MOVE prompts/ → src/convfinqa/prompts/:
  - No content changes inside the version files; the __init__.py uses
    `importlib.import_module(f"prompts.{version}")` — that string MUST be
    updated to `f"convfinqa.prompts.{version}"`.
UPDATE pydantic_agent.py:
  - FIND: `import prompts as _prompts_pkg`
  - REPLACE: `import convfinqa.prompts as _prompts_pkg`

Task 1.5 — Move data.py and data_scope.py
MOVE data.py → src/convfinqa/data/loader.py:
  - No behaviour changes yet. The `globals()` mutation stays in Phase 1
    (it gets cleaned in Phase 2). Keep the public function
    `training_data() -> pd.DataFrame` byte-identical.
MOVE data_scope.py → src/convfinqa/data/scope.py:
  - No content changes.
UPDATE callers (api_eval.py, dspy_agent.py):
  - FIND: `from data import training_data`
  - REPLACE: `from convfinqa.data.loader import training_data`

Task 1.6 — Register CLI entry points
MODIFY pyproject.toml:
  - Under `[project.scripts]`, replace `main = "src.main:app"` with:
      convfinqa-eval = "scripts.evaluate:main"
      convfinqa-eval-api = "scripts.evaluate_api:main"
      convfinqa-optimize = "scripts.optimize:main"
      convfinqa-serve = "scripts.serve:main"
  - Phase 4 will create the actual scripts/ files. In Phase 1, just register.

Task 1.7 — Verify imports + run tests
RUN: `uv sync` (re-resolves the package layout)
RUN: `uv run pytest tests/ -v`
RUN: `uv run python pydantic_agent.py` (legacy path still works — root files
      still exist; they just delegate to the new locations)
ASSERT: end-to-end accuracy table matches the pre-refactor numbers exactly.
```

#### Phase 1 validation

```bash
# Syntax + import check across all touched files
uv run python -c "import ast; [ast.parse(open(f).read()) for f in ['src/convfinqa/__init__.py', 'src/convfinqa/config.py', 'src/convfinqa/evaluation/__init__.py', 'src/convfinqa/evaluation/cache.py', 'src/convfinqa/evaluation/metrics.py', 'src/convfinqa/prompts/__init__.py', 'src/convfinqa/data/loader.py']]"

# Verify the package resolves end-to-end
uv run python -c "
from convfinqa.config import settings
from convfinqa.evaluation import numeric_match, load_cached_conversations
from convfinqa.prompts import latest, load
from convfinqa.data.loader import training_data
assert numeric_match('60', '59.7') is True
assert latest() == 'v2'
print('Phase 1 imports OK')
"

# Full test suite
uv run pytest tests/ -v

# End-to-end smoke (hits cache, no API calls if v1/v2 CSVs present)
REUSE_CACHE=1 uv run python pydantic_agent.py | tail -20

# Linting
uv run ruff check src/ tests/
```

### Phase 2 — Extract shared schemas and primitives

Goal: pull `ConvExample`, `QAPair`, `ConversationHistory`, the four pipeline-stage Pydantic models, `_render_chat_inputs`, and calculator tools out of the monoliths into small focused modules. After this phase, the monoliths are noticeably smaller but still functionally identical.

```yaml
Task 2.1 — Extract data schemas
CREATE src/convfinqa/data/schemas.py:
  - MOVE `ConvExample` from api_eval.py (BaseModel with report_id, questions,
    gold_answers, gold_programs, gold_turn_types, gold_conv_types).
  - MOVE `QAPair` and `ConversationHistory` from dspy_agent.py.
  - Keep all fields and methods byte-identical. Add module docstring.
UPDATE api_eval.py and dspy_agent.py:
  - Replace the now-deleted class definitions with re-export shims:
      `from convfinqa.data.schemas import ConvExample`  # re-export
      `from convfinqa.data.schemas import QAPair, ConversationHistory`
  - Old code that imports from api_eval/dspy_agent keeps working.

Task 2.2 — Extract pipeline stage models
CREATE src/convfinqa/pipeline/stages.py:
  - MOVE `TriageOut`, `PreprocessOut`, `RetrievedValues`, `CalcOut` from
    pydantic_agent.py.
  - Re-export Literal types `TurnType`, `ConvType` from this module.
UPDATE pydantic_agent.py:
  - Add re-export shims so existing imports keep working.

Task 2.3 — Extract calculator tools
CREATE src/convfinqa/pipeline/tools.py:
  - MOVE the calculator tool functions from dspy_agent.py
    (add, subtract, multiply, divide, exp, greater).
  - MOVE the `CALCULATOR_TOOLS` list.
UPDATE dspy_agent.py and pydantic_agent.py:
  - Re-export shim:
      `from convfinqa.pipeline.tools import CALCULATOR_TOOLS  # re-export`

Task 2.4 — Extract DSPy ChatAdapter wire format
CREATE src/convfinqa/pipeline/wire_format.py:
  - MOVE `_render_chat_inputs` from pydantic_agent.py (rename to public
    `render_chat_inputs`).
  - Keep the format byte-for-byte identical — the GEPA-optimised prompts were
    tuned against this exact rendering.
UPDATE pydantic_agent.py:
  - Replace inline definition with `from convfinqa.pipeline.wire_format import render_chat_inputs as _render_chat_inputs`.

Task 2.5 — Clean up data.py globals()
MODIFY src/convfinqa/data/loader.py:
  - The function currently does:
      `globals()[f'{key}_df'] = pd.concat([...])` inside a for-loop.
  - Replace with an explicit dict accumulator:
      ```python
      dfs: dict[str, pd.DataFrame] = {}
      features: dict[str, pd.DataFrame] = {}
      for key in data.keys():
          dfs[key] = pd.concat([...])
          features[key] = pd.DataFrame(...)
      ```
  - The public function `training_data() -> pd.DataFrame` keeps the same signature.

Task 2.6 — Move qa_data / _DOCS to the data module
MOVE the module-level `qa_data` and `_DOCS` constants from dspy_agent.py to
  src/convfinqa/data/loader.py:
  - These are loaded once at import time and shared across the codebase.
  - Add re-export shims in dspy_agent.py so old imports keep working.

Task 2.7 — Verify imports + run tests
RUN: pytest, then end-to-end smoke (same as Phase 1).
```

#### Phase 2 validation

```bash
# Each new module is importable on its own
uv run python -c "
from convfinqa.data.schemas import ConvExample, QAPair, ConversationHistory
from convfinqa.pipeline.stages import TriageOut, PreprocessOut, RetrievedValues, CalcOut
from convfinqa.pipeline.tools import CALCULATOR_TOOLS, add, subtract
from convfinqa.pipeline.wire_format import render_chat_inputs
from convfinqa.data.loader import training_data, qa_data, _DOCS
print('Phase 2 imports OK')
print(f'  CALCULATOR_TOOLS has {len(CALCULATOR_TOOLS)} tools')
print(f'  qa_data shape: {qa_data.shape}')
"

# Old imports STILL work (re-export shims)
uv run python -c "
from api_eval import ConvExample, load_conv_examples_test, numeric_match
from dspy_agent import _DOCS, qa_data, CALCULATOR_TOOLS, ConversationHistory, QAPair
print('Phase 2 re-export shims OK')
"

uv run pytest tests/ -v
REUSE_CACHE=1 uv run python pydantic_agent.py | tail -20
uv run ruff check src/ tests/
```

### Phase 3 — Decompose the two agent monoliths

Goal: split `pydantic_agent.py` (1504 LOC) and `dspy_agent.py` (1290 LOC) into per-backend modules + per-concern eval modules. This is the highest-risk phase because the agent constructors have load-order dependencies — execute the sub-steps in order and verify each one.

```yaml
Task 3.1 — Extract Pydantic AI backend
CREATE src/convfinqa/backends/pydantic.py:
  - MOVE the LM_MINI provider + 4 module-level Agent instances (triage_agent,
    preprocess_agent, retriever_agent, calculator_agent) from pydantic_agent.py.
  - MOVE `_make_agents(prompts)` factory.
  - MOVE `_make_task_fn(agents)` factory (used by pydantic-evals).
  - Imports of `from convfinqa.config import settings` MUST come first.

Task 3.2 — Extract pipeline runner
CREATE src/convfinqa/pipeline/runner.py:
  - MOVE `run_turn` and `stream_turn` from pydantic_agent.py.
  - These take `agents: dict[str, Agent]` so they're framework-agnostic — they
    don't import pydantic-ai directly. They only need the four agent objects.
  - MOVE `_calc_trajectory`, `_coerce_args`, `_tool_events_from_messages` helpers.
  - MOVE `ConversationRunner` class.
UPDATE app.py:
  - FIND: `from pydantic_agent import run_turn, stream_turn`
  - REPLACE: `from convfinqa.pipeline.runner import run_turn, stream_turn`

Task 3.3 — Extract prompt resolution
CREATE src/convfinqa/pipeline/prompts_loader.py:
  - MOVE `_deep_merge`, `_load_optimized_prompts`, `_resolve_prompts` from
    pydantic_agent.py.
  - MOVE the `PROMPTS_PATH` / `RUN_DIR` constants.
  - Returns a `dict[str, str]` keyed by short agent name.

Task 3.4 — Extract evaluation runner
CREATE src/convfinqa/evaluation/runner.py:
  - MOVE `_build_dataset`, `_conv_task`, `_evaluate_version`, `_write_predictions_csv`,
    `_write_joined_predictions`, `_REQUIRED_PRED_COLUMNS`, `PREDICTIONS_COLUMNS`,
    `ConvInput`, `ConvOutput`, `TurnAccuracy`, `get_predictions_path` from
    pydantic_agent.py.
  - Function renames: leading-underscore privates that are now exported:
      `_evaluate_version` → `evaluate_version`
  - Update `_evaluate_version`'s pandas + prompts import to use absolute paths.

Task 3.5 — Extract reporting
CREATE src/convfinqa/evaluation/reporting.py:
  - MOVE `write_predictions_html` (and its inline CSS/JS).
  - MOVE `print_accuracy_table`.

Task 3.6 — Extract DSPy backend
CREATE src/convfinqa/backends/dspy.py:
  - MOVE the four DSPy signatures (`TriageSignature`, `PreprocessSignature`,
    `RetrieverSignature`, `CalculationSignature`) and their predictors.
  - MOVE the LM constructors (`lm_mini`, `lm_max`).
  - MOVE the `ConvFinQAPipeline` dspy.Module (or whatever the program is named).
  - MOVE `dspy.configure(...)` call.

Task 3.7 — Extract GEPA optimization
CREATE src/convfinqa/optimization/gepa.py:
  - MOVE the GEPA training block from dspy_agent.py (everything under
    `if settings.run_gepa:` in the __main__).
  - MOVE the mode-selection, resume, baseline-vs-optimized eval, and predictions
    write logic.

Task 3.8 — Extract analyze_predictions
CREATE src/convfinqa/evaluation/joining.py:
  - MOVE `analyze_predictions` from dspy_agent.py.
  - MOVE the helper that builds `*_joined.csv`.

Task 3.9 — Reduce root pydantic_agent.py to a shim
After 3.1-3.8, pydantic_agent.py should be down to <100 lines. Make it a
back-compat shim that just re-exports the public names:
  - `from convfinqa.pipeline.runner import run_turn, stream_turn, ConversationRunner`
  - `from convfinqa.backends.pydantic import triage_agent, preprocess_agent, retriever_agent, calculator_agent`
  - `from convfinqa.evaluation.runner import evaluate_version as _evaluate_version, ConvInput, ConvOutput`
  - `from convfinqa.evaluation.reporting import write_predictions_html, print_accuracy_table`
  - Keep the `__main__` block but rewrite it to a 10-line driver that calls
    `convfinqa.evaluation.runner.run_all_versions(...)`.

Task 3.10 — Reduce root dspy_agent.py to a shim
Similar to 3.9. Keep the `__main__` block but delegate to
`convfinqa.optimization.gepa.main()`.

Task 3.11 — Verify
RUN pytest, end-to-end smoke, type check.
```

#### Phase 3 validation

```bash
# Each new module is importable in isolation
uv run python -c "
from convfinqa.backends.pydantic import triage_agent, preprocess_agent, retriever_agent, calculator_agent
from convfinqa.pipeline.runner import run_turn, stream_turn, ConversationRunner
from convfinqa.pipeline.prompts_loader import PROMPTS_PATH
from convfinqa.evaluation.runner import evaluate_version, ConvInput, ConvOutput
from convfinqa.evaluation.reporting import write_predictions_html, print_accuracy_table
from convfinqa.evaluation.joining import analyze_predictions
print('Phase 3 imports OK')
"

# Line counts: every new module < 400 lines
find src/convfinqa -name '*.py' -exec wc -l {} + | sort -n | tail

# Legacy paths still work via shims
uv run python -c "
from pydantic_agent import run_turn, stream_turn, write_predictions_html
from dspy_agent import analyze_predictions, qa_data
print('Phase 3 shims OK')
"

# Full e2e — this is the critical test
uv run pytest tests/ -v
REUSE_CACHE=1 uv run python pydantic_agent.py | tail -20
# ↑ MUST produce the same v1/v2 accuracy table as before the refactor:
#   v1 overall 73.0%, v2 overall 77.1%, 200 cached conversations
```

### Phase 4 — Serving, scripts, docs, final cleanup

Goal: move the FastAPI app, CLI, and harness into the package; create thin `scripts/` entrypoints; consolidate the docs.

```yaml
Task 4.1 — Move FastAPI app
CREATE src/convfinqa/serving/sessions.py:
  - MOVE the `Session` dataclass, `SessionStore`, lifespan management, TTL
    eviction from app.py.
CREATE src/convfinqa/serving/routes_eval.py:
  - MOVE the `/eval/runs` and `/eval/runs/<v>/...` endpoints.
  - MOVE `_load_preds`, `_slice_accuracy`, `_slices_by`, `_MODEL_CSV_PATTERN`,
    `EVAL_DIR`.
CREATE src/convfinqa/serving/app.py:
  - The `create_app()` factory + CORS + lifespan wiring.
  - Imports session store from `convfinqa.serving.sessions`.
  - Imports eval routes from `convfinqa.serving.routes_eval`.
  - Imports `run_turn`, `stream_turn` from `convfinqa.pipeline.runner`.

Task 4.2 — Collapse duplicate CLI
NOTE (2026-05-16): `cli.py` and `src/convfinqa/serving/cli.py` already exist as
two near-identical copies (diff is import-path-only). The action here is to
collapse, not to move:
  - KEEP src/convfinqa/serving/cli.py as canonical. Confirm it has the
    `from convfinqa.data.loader import load_conv_examples_test` import (not
    `from api_eval import ...`).
  - REPLACE root cli.py with a 2-line re-export:
      `from convfinqa.serving.cli import cli_app, app  # noqa: F401`
  - UPDATE tests/test_cli.py: `import cli` → `from convfinqa.serving import cli`.
  - Once test_cli.py passes against the package path, delete root cli.py.

Task 4.3 — Move api_eval
CREATE src/convfinqa/evaluation/api_runner.py:
  - MOVE `evaluate_api`, `compare_model_accuracies`, `compare_api_outputs`,
    `_evaluate_conversation`, `_evaluate_api_async`, `_load_cached_rows`,
    `_API_CSV_COLUMNS` from api_eval.py.

Task 4.4 — Collapse duplicate prompt-optim harness
NOTE (2026-05-16): `prompt_optim_v2.py` and `src/convfinqa/optimization/harness.py`
already exist as two near-identical copies (diff is import-path-only). Same
collapse pattern as Task 4.2:
  - KEEP src/convfinqa/optimization/harness.py as canonical. Confirm its imports
    point at `convfinqa.data.loader` and `convfinqa.pipeline.runner`, not the
    root monoliths.
  - REPLACE root prompt_optim_v2.py with a 2-line re-export of `main` if any
    caller still uses the legacy path, otherwise delete it.
  - UPDATE tests/test_prompt_improve_v2.py to import from the package path.

Task 4.5 — Create thin scripts
CREATE scripts/evaluate.py:
  - 10-line file: parse args, call `convfinqa.evaluation.runner.main()`.
CREATE scripts/evaluate_api.py:
  - Calls `convfinqa.evaluation.api_runner.main()`.
CREATE scripts/serve.py:
  - Calls `uvicorn.run("convfinqa.serving.app:create_app", factory=True, ...)`.
CREATE scripts/optimize.py:
  - Calls `convfinqa.optimization.gepa.main()` or
    `convfinqa.optimization.harness.main()` based on a flag.

Task 4.6 — Delete root .py files
After 4.1-4.5, these files should be empty shims or fully redundant:
  - app.py (replaced by serving.app)
  - cli.py (replaced by serving.cli)
  - api_eval.py (replaced by evaluation.api_runner)
  - prompt_optim_v2.py (replaced by optimization.harness)
  - pydantic_agent.py (shim from Phase 3 — can be deleted now, or kept as a
    one-line `from convfinqa.pipeline.runner import *` if anyone outside the
    repo depends on the legacy import path)
  - dspy_agent.py (same — can be deleted)
  - data.py, data_scope.py, evaluator/ (already moved in Phase 1)
  - config.py (already moved in Phase 1)
DELETE the root files OR replace each with a one-line deprecation shim that
points to the new path.

Task 4.6b — Gitignore + dead-code cleanup
MODIFY .gitignore:
  - Add `evaluation/` (currently AGENTS.md claims it is ignored but it is not).
RUN: `git rm --cached evaluation/*.csv evaluation/*.html`
DECIDE on mcp/:
  - Confirmed unreferenced anywhere in src/, scripts/, tests/, root.
  - Either delete the folder or add one line to AGENTS.md describing its
    purpose and lifecycle so the next reviewer doesn't flag it again.
DELETE root agent.py:
  - After tests/test_agent.py is repointed at `convfinqa.backends.dspy`, the
    root shim has no remaining importers.

Task 4.7 — Consolidate docs
CREATE docs/architecture.md:
  - Move the architectural content from AGENTS.md here.
  - AGENTS.md becomes a one-paragraph pointer to docs/architecture.md (matches
    the CLAUDE.md convention at /Users/nathanphillips/git/CLAUDE.md).
CREATE docs/dataset.md:
  - Move root dataset.md here.
RENAME PRPs/ → docs/decisions/:
  - Treat each PRP as an ADR (Architecture Decision Record).
UPDATE README.md:
  - Replace every `python pydantic_agent.py` / `python api_eval.py` etc.
    command with the new entry-point form (`uv run convfinqa-eval`,
    `uv run convfinqa-eval-api`, etc.).
  - Update the Repo Layout table.

Task 4.8 — Final validation
RUN: pytest, end-to-end smoke, frontend smoke.
```

#### Phase 4 validation

```bash
# Installed CLI entry points work
uv run convfinqa-eval --help 2>&1 | head -5
uv run convfinqa-serve --help 2>&1 | head -5

# Server starts cleanly under new module path
uv run python -m uvicorn convfinqa.serving.app:create_app --factory --workers 1 --port 8765 &
SERVER_PID=$!
sleep 3
curl -s http://127.0.0.1:8765/healthz | jq .
curl -s http://127.0.0.1:8765/eval/runs | jq .  # ["v1", "v2"]
kill $SERVER_PID

# Frontend smoke (manual)
cd frontend && npm run dev
# Open http://localhost:5173, pick a report, send a question, verify streaming + per-stage chips render

# Final line counts — every src/convfinqa/*.py file under 400 LOC
find src/convfinqa -name '*.py' -exec wc -l {} + | sort -rn | head -20
# Root has at most: pyproject.toml, README.md, CLAUDE.md, AGENTS.md, uv.lock,
# .gitignore (no .py files)
ls *.py 2>&1 | wc -l   # expected: 0

# Full suite
uv run pytest tests/ -v
uv run ruff check src/ scripts/ tests/
uv run mypy src/convfinqa
```

## Validation Loop

Run after every phase, not just at the end. The first failure tells you what to revert.

### Level 1: Imports + syntax

```bash
uv run python -c "
import convfinqa, convfinqa.config, convfinqa.evaluation, convfinqa.prompts, convfinqa.data.loader
print('package imports OK')
"
uv run ruff check src/ scripts/ tests/
```

### Level 2: Unit tests

```bash
uv run pytest tests/ -v
```

Existing tests:
- `tests/test_agent.py`
- `tests/test_api.py`
- `tests/test_api_eval.py`
- `tests/test_app_cors.py`
- `tests/test_cli.py`
- `tests/test_prompt_improve_v2.py`
- `tests/test_pydantic_agent.py`

Each phase will require updating these tests' imports in lockstep with the moves. The set of tests that must pass is the same; only the import paths change.

### Level 3: End-to-end smoke (the critical gate)

```bash
# Reproduces the canonical v1/v2 accuracy table from cache, no API calls
REUSE_CACHE=1 uv run python pydantic_agent.py
```

Expected last 25 lines (must match before/after each phase exactly):

```
[v1] cache hit: 200/200 conversations (770 questions) — skipping
[v1] combined accuracy: 73.0%  (562/770 questions)
[v2] cache hit: 200/200 conversations (770 questions) — skipping
[v2] combined accuracy: 77.1%  (594/770 questions)

----------------------------------------------------------
Cut                      Count            v1            v2
----------------------------------------------------------
Overall                    770        73.0%         77.1%
turn_type=Number           284        85.2%         87.7%
turn_type=Program          486        65.8%         71.0%
conv_type=Type I           640        75.2%         78.8%
conv_type=Type II          130        62.3%         69.2%
question=0                 200        81.0%         82.0%
question=1                 199        75.4%         79.4%
question=2                 160        70.0%         75.6%
...
```

Any divergence here means a refactor introduced a behaviour change. STOP and bisect.

### Level 4: Server + frontend smoke (Phase 4 only)

```bash
# Backend
uv run python -m uvicorn convfinqa.serving.app:create_app --factory --workers 1 --port 8765 &

# Frontend
cd frontend && npm run dev
# Open browser, pick a report, ask one question, verify streaming works.
```

## Final Validation Checklist

- [ ] `find src/convfinqa -name '*.py' | wc -l` ≥ 20 (the package has real internal structure)
- [ ] `find src/convfinqa -name '*.py' -exec wc -l {} +` shows no file > 400 lines
- [ ] `ls *.py 2>/dev/null | wc -l` == 0 (no Python at root)
- [ ] `uv run pytest tests/ -v` passes
- [ ] `uv run ruff check src/ scripts/ tests/` clean
- [ ] `uv run mypy src/convfinqa` clean (under current non-strict config)
- [ ] End-to-end smoke reproduces v1: 73.0%, v2: 77.1%
- [ ] Server starts and `/eval/runs` returns `["v1", "v2"]`
- [ ] Frontend can still chat with a report (manual smoke)
- [ ] README.md reflects the new entry-point commands
- [ ] AGENTS.md / CLAUDE.md updated to point at the new layout
- [ ] No `os.environ.get(...)` calls anywhere in `src/convfinqa/` (all env reads through `convfinqa.config.settings`)
- [ ] No `from <old_root_module> import ...` lines in `src/convfinqa/` (the package never references back to root shims)

## Anti-Patterns to Avoid

- ❌ **Big-bang refactor.** Don't move everything in one commit. Land Phase 1, verify, commit. Then Phase 2. The riskiest moves come last on purpose.
- ❌ **Renaming public functions during the move.** `_evaluate_version` becomes `evaluate_version` is fine (drop the leading underscore on now-exported names). But `run_turn` → `execute_pipeline_turn` is scope creep; leave names alone.
- ❌ **Deleting shims too early.** Phase 3 root files become re-export shims; only Phase 4 deletes them. Tests, the prompt-optim harness, and any external callers all need a beat to migrate their imports first.
- ❌ **Changing the wire format.** `render_chat_inputs` (formerly `_render_chat_inputs`) MUST produce byte-identical output before and after the move. The GEPA-optimised prompts were tuned against the current rendering.
- ❌ **Re-introducing `os.environ.get(...)`.** Everything goes through `settings`. If a new env var is needed, add a field to `convfinqa.config.Settings` and use it. No exceptions.
- ❌ **Touching frontend code.** This refactor is purely Python-side. The frontend reads from `/api/...` proxied to the backend; as long as backend routes don't change, the frontend doesn't care that the implementation moved.
- ❌ **Refactoring the DSPy `Module` class.** GEPA training expects specific predictor names (`triage.predict`, `preprocess.predict`, etc.). Move the file, don't redesign the class.
- ❌ **Skipping the end-to-end smoke after a phase.** The v1: 73.0% / v2: 77.1% baseline is the canary. If those numbers move by even 0.1%, something broke and you won't notice from unit tests alone.

---

## Confidence

PRP confidence score for one-pass execution per phase by a coding agent: **8/10**.

What raises confidence:
- The end-state structure is concrete (full tree provided)
- Import migrations are pre-mapped in a single table
- A canonical accuracy table acts as a smoke-test oracle
- Per-phase validation gates catch regressions early
- The codebase is already 60% there (`config.py`, `evaluator/`, `prompts/` are clean and small)

What keeps it under 10:
- Phase 3 touches the most lines and has load-order dependencies; subtle import-time side effects (dspy.LM constructor, logfire.configure) can surface only at runtime
- GEPA optimisation paths are not covered by the test suite — the agent has to trust the `dspy_agent.py` `__main__` block won't be exercised by the smoke test
- The frontend smoke is manual and skippable, so a Vite-proxy regression could slip if Phase 4 forgets to verify the live server
- Phase 3 landed inverted (package shims import from root rather than the reverse); finishing the migration means actually moving code, not just renaming imports — the load-order gotcha around `dspy.LM(deepseek/...)` is still latent

Mitigation: run the end-to-end smoke after EVERY task (not just every phase), and commit between tasks so any regression bisects to a 50-line diff.
