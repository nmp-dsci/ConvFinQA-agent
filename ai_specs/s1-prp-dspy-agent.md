name: "DSPy Multi-Agent for ConvFinQA — single-file v1 baseline"
description: |

  ## Purpose
  Build a single-file DSPy multi-agent system (`agent.py`) that answers
  conversational questions about a financial document and evaluates itself
  against 2 randomly selected dev records (all turns). This is the **v1
  baseline** — a working end-to-end implementation we can later split into
  modules, optimize with GEPA, and deploy via FastAPI.

  ## Core Principles
  1. **Context is King** — include all docs, examples, and caveats inline
  2. **Validation Loops** — every task ends with an executable check
  3. **Information Dense** — use real keywords/patterns from the codebase
  4. **Progressive Success** — get a 1-record run working first, then evaluate
  5. **Global rules** — follow CLAUDE.md (`uv` for deps, `ruff` for lint, mypy strict, snake_case)

---

## Goal

Create **one file**: `agent.py` (at the repo root) that stands up the DSPy
agent end-to-end and self-evaluates. When run as `uv run python agent.py`, it:

1. Configures DSPy with Claude (Anthropic via LiteLLM).
2. Loads `data/convfinqa_dataset.json` and randomly picks 2 records from the `dev` split (use a fixed seed for reproducibility).
3. For each record, walks every dialogue turn through the 4-agent pipeline (Triage → Preprocessing → Retriever → Calculation), maintaining `ConversationHistory` across turns.
4. Compares each predicted answer to `dialogue.executed_answers` using a numeric-equivalence metric (1e-3 tolerance) and prints per-record + overall Execution Accuracy.

This PRP scope is the **v1 baseline only** — no CLI, no GEPA, no FastAPI, no Logfire. Those are follow-on PRPs.

## Why

- **Foundation for GEPA**: GEPA optimizes over `dspy.Module`s with `Signature`s; we need a clean DSPy implementation before optimization is even possible.
- **Replaces ad-hoc pydantic-ai prototype**: `provider.py` (~54KB) was a v7 pydantic-ai implementation. DSPy gives us declarative signatures + an evaluation harness for free.
- **Multi-turn correctness**: Per CLAUDE.md, >60% of ConvFinQA questions depend on prior turns. A clean conversation-history primitive needs to live at the orchestrator level.
- **Single file = fast iteration**: easier to share, reason about, and refactor later. Modular split (models.py / tools.py / etc.) is a follow-on once the design is stable.

## What

A single executable script at `agent.py` containing:
- 6 calculator functions (plain Python, used by `dspy.ReAct`)
- 4 `dspy.Signature` classes (Triage / Preprocessing / Retriever / Calculation)
- 4 `dspy.Module` agent classes wrapping those signatures
- 1 `ConvFinQAOrchestrator` module wiring them together
- Pydantic models: `QAPair`, `ConversationHistory`, `AgentResponse`
- Dataset loader + document serializer
- `evaluate_random_records(n=2, seed=42)` function and `if __name__ == "__main__"` runner

### Success Criteria

- [ ] `uv run python agent.py` runs end-to-end with no traceback, prints per-record and overall Execution Accuracy
- [ ] The 2 records are reproducible with a fixed seed (re-runs pick the same 2 ids)
- [ ] Multi-turn coreference works: a turn referencing "that value" produces an answer consistent with the prior turn's gold answer (teacher-forced history)
- [ ] All 4 agents are `dspy.Module` subclasses with explicit `dspy.Signature`s
- [ ] `ruff check agent.py --fix && ruff format agent.py` — zero errors
- [ ] `uv run mypy agent.py` — zero errors (strict per pyproject.toml)
- [ ] `uv run pytest tests/test_agent.py -v` — all tests pass
- [ ] Overall Execution Accuracy ≥ 50% on the 2 sampled records' turns (baseline target — GEPA will lift this later)

## All Needed Context

### Documentation & References

```yaml
# MUST READ — DSPy core concepts
- url: https://dspy.ai/learn/programming/signatures/
  why: Class-based typed signatures (the foundation of every agent)
  critical: |
    Use class-based Signatures (not string shortcuts). Output fields use
    dspy.OutputField. Field `desc=` strings are included in the prompt — they matter.

- url: https://dspy.ai/learn/programming/modules/
  why: dspy.Predict vs ChainOfThought vs ReAct — choose per agent
  critical: |
    Triage = dspy.Predict (cheap classification)
    Preprocessing = dspy.ChainOfThought (decomposition needs reasoning)
    Retriever = dspy.ChainOfThought (scan doc + history)
    Calculation = dspy.ReAct with calculator tools

- url: https://dspy.ai/api/modules/ReAct/
  why: Tool-calling agent — used for the Calculation agent
  critical: |
    Tools are plain Python functions. Type hints + docstrings drive the prompt.
    Use bare types (float, str, int) for tool args, NOT Pydantic models.

- url: https://dspy.ai/learn/evaluation/data/
  why: dspy.Example construction
  critical: |
    .with_inputs("field_name", ...) marks which fields are inputs vs labels.

- url: https://dspy.ai/learn/evaluation/metrics/
  why: How to write a custom metric
  critical: |
    Signature is metric(example, prediction, trace=None) -> bool|float.

- url: https://dspy.ai/api/evaluation/Evaluate/
  why: dspy.Evaluate harness
  critical: |
    num_threads=1 to avoid Anthropic rate limits in v1.

- url: https://dspy.ai/learn/programming/language_models/
  why: Configure DSPy with Anthropic Claude
  critical: |
    dspy.LM("anthropic/claude-sonnet-4-6", api_key=os.environ["ANTHROPIC_API_KEY"])
    then dspy.configure(lm=lm). Process-global — call once.

# MUST READ — codebase context
- file: claude_agent_spec.md
  why: Authoritative spec (sections 4–6, 12). 59KB — skim, don't memorize.

- file: provider.py
  why: |
    Existing pydantic-ai implementation (~54KB). Mirror agent boundaries and
    prompt phrasing. Do NOT copy framework code (different framework).

- file: mcp/server_calculator.py
  why: |
    Existing calculator tool definitions (add, subtract, multiply, divide, exp, greater).
    Re-implement these six as plain Python functions in agent.py for dspy.ReAct
    (do NOT call MCP from the agent — ReAct works best with native callables).

- file: data/convfinqa_dataset.json
  why: |
    21MB. Top-level keys: {"train": [...], "dev": [...]}. Each record has id,
    doc, dialogue, features. Load lazily; pick 2 random ids from dev with a
    fixed seed.

- file: PRPs/pydantic-agent.md
  why: |
    Reference for PRP structure (a 7/10 confidence PRP). Use its task
    granularity and validation patterns as a template — but the implementation
    is pydantic-ai, not DSPy, so don't copy code.

- file: pyproject.toml
  why: |
    ruff/mypy config (strict mode, T201 forbids `print` in source — but agent.py
    runs as a script, so guard prints behind `if __name__ == "__main__"` or use
    logging). Add `dspy>=2.5` here.
```

### Current Codebase tree (relevant subset)

```bash
.
├── CLAUDE.md                       # project instructions
├── claude_agent_spec.md            # full system spec (59KB)
├── pyproject.toml                  # uv-managed; add dspy here
├── provider.py                     # legacy pydantic-ai impl (reference only)
├── mcp/
│   └── server_calculator.py        # calculator tool definitions to mirror
├── data/
│   └── convfinqa_dataset.json      # 21MB, {"train": [...], "dev": [...]}
└── PRPs/
    ├── prp-dspy-agent.md           # this file
    └── pydantic-agent.md           # reference PRP structure
```

### Desired Codebase tree

```bash
.
├── agent.py                        # NEW — single-file DSPy multi-agent + eval runner
└── tests/
    └── test_agent.py               # NEW — pipeline tests using dspy.utils.DummyLM
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: dspy.configure(lm=...) is process-global. Call ONCE in agent.py
#   (inside configure_dspy() helper or at top of __main__). Re-configuring
#   mid-process can cause surprising state.

# CRITICAL: dspy.Signature class-based form — output fields must use
#   dspy.OutputField (not Pydantic Field). For Literal/Enum outputs, declare
#   the type as Literal["a","b"] and DSPy will constrain the LM output.

# CRITICAL: dspy.ReAct tools are plain Python callables with type hints +
#   docstrings. The docstring becomes the tool description in the prompt.
#   Type hints determine the JSON schema. Do NOT use Pydantic models as tool
#   args; use bare types (float, str, int).

# CRITICAL: When a dspy.Module calls another dspy.Module inside .forward(),
#   the inner module shares the global LM. Subclassing Module is fine; do NOT
#   instantiate a fresh LM per call.

# GOTCHA: dspy.Predict returns a `dspy.Prediction` object, not the raw output.
#   Access fields via attribute: `pred.turn_type`, NOT `pred["turn_type"]`.

# GOTCHA: Anthropic via DSPy uses LiteLLM under the hood. Model string format:
#   "anthropic/claude-sonnet-4-6" — note the slash, not the colon (PydanticAI uses ":")

# GOTCHA: pyproject.toml's T201 forbids `print()`. agent.py uses prints for
#   the eval summary — wrap them in a `report()` helper that calls `logging.info`
#   OR add a per-file `# ruff: noqa: T201` only at the top of agent.py if the
#   eval is the entrypoint. Tests should never print — use assertions.

# GOTCHA: The dataset's `dialogue.executed_answers` is the gold for evaluation.
#   It's `list[float | str]`. Compare numerically with a tolerance (1e-3) for
#   floats; string-equal for non-numeric.

# GOTCHA: For random selection, use `random.Random(seed=42).sample(records, 2)`
#   — do NOT call random.sample() globally, since DSPy/LiteLLM may also use the
#   global RNG and that would make the selection non-reproducible.

# GOTCHA: Don't print from inside a dspy.Module.forward() — it interferes with
#   dspy.Evaluate's progress bar. Return diagnostics on the Prediction /
#   AgentResponse instead.
```

## Implementation Blueprint

### Single-file structure (`agent.py`)

```
# === Imports ===
import json, os, random, logging
from functools import cache
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
import dspy

# === 1. Pydantic models (≈ 30 lines) ===
#   - QAPair, ConversationHistory, AgentResponse
#   - ConvFinQARecord / Document / Dialogue / Features (port from CLAUDE.md spec)

# === 2. Calculator tools (≈ 30 lines) ===
#   - add, subtract, multiply, divide, exp, greater  (mirror mcp/server_calculator.py)
#   - CALCULATOR_TOOLS = [...]

# === 3. DSPy Signatures (≈ 30 lines) ===
#   - TriageSignature, PreprocessingSignature, RetrieverSignature, CalculationSignature

# === 4. DSPy Modules (≈ 40 lines) ===
#   - TriageAgent, PreprocessingAgent, RetrieverAgent, CalculationAgent

# === 5. Orchestrator (≈ 35 lines) ===
#   - ConvFinQAOrchestrator with branching on turn_type

# === 6. Dataset loading (≈ 25 lines) ===
#   - _load_raw (cached), get_record, serialize_document
#   - sample_records(n, seed) -> list[ConvFinQARecord]

# === 7. Evaluation (≈ 40 lines) ===
#   - configure_dspy()
#   - numeric_match(pred_answer, gold_answer) -> bool
#   - run_record(orchestrator, record) -> list[(pred, gold, correct)]  (uses teacher-forced history)
#   - evaluate_random_records(n=2, seed=42) -> overall_acc

# === 8. __main__ ===
#   - configure_dspy()
#   - evaluate_random_records()
```

### Pseudocode highlights (the parts most likely to trip Claude up)

```python
# --- Models -----------------------------------------------------------
class QAPair(BaseModel):
    question: str
    answer: str

class ConversationHistory(BaseModel):
    pairs: list[QAPair] = Field(default_factory=list)
    def append(self, q: str, a: str) -> None: self.pairs.append(QAPair(question=q, answer=a))
    def as_text(self) -> str:
        if not self.pairs: return "(no prior turns)"
        return "\n".join(f"Q{i+1}: {p.question}\nA{i+1}: {p.answer}" for i, p in enumerate(self.pairs))

TurnType = Literal["number", "program"]
ConvType = Literal["Type I", "Type II"]

class AgentResponse(BaseModel):
    answer: str
    turn_type: TurnType
    conv_type: ConvType
    program: str | None = None
    sub_questions: list[QAPair] = Field(default_factory=list)


# --- Calculator tools -------------------------------------------------
def add(a: float, b: float) -> float:
    """Return a + b."""
    return a + b
# ... subtract, multiply, divide, exp, greater
CALCULATOR_TOOLS = [add, subtract, multiply, divide, exp, greater]


# --- Signatures -------------------------------------------------------
class TriageSignature(dspy.Signature):
    """Classify a financial question into turn_type and conversation type."""
    question: str = dspy.InputField()
    history: str = dspy.InputField(desc="Prior Q&A in the session")
    turn_type: Literal["number", "program"] = dspy.OutputField(
        desc="'number' if the answer is a single value lookup; 'program' if it requires arithmetic"
    )
    conv_type: Literal["Type I", "Type II"] = dspy.OutputField(
        desc="'Type II' if the question switches to a different aspect of the report; else 'Type I'"
    )

class PreprocessingSignature(dspy.Signature):
    """Decompose a program-type question into lookup sub-questions and a calculation program."""
    question: str = dspy.InputField()
    history: str = dspy.InputField()
    sub_questions: list[str] = dspy.OutputField(
        desc="Each sub-question must be a self-contained value lookup, not a computation"
    )
    program: str = dspy.OutputField(
        desc="DSL program e.g. 'subtract(A, B)' or 'divide(subtract(A, B), B)' where A,B are placeholders matching sub_questions order"
    )

# RetrieverSignature: question + document + history -> answer:str
# CalculationSignature: program + retrieved -> answer:str


# --- Orchestrator -----------------------------------------------------
class ConvFinQAOrchestrator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.triage = TriageAgent()
        self.preprocess = PreprocessingAgent()
        self.retriever = RetrieverAgent()
        self.calculator = CalculationAgent()

    def forward(self, question: str, document: str, history: ConversationHistory) -> AgentResponse:
        hist_text = history.as_text()
        triage = self.triage(question=question, history=hist_text)
        if triage.turn_type == "number":
            r = self.retriever(question=question, document=document, history=hist_text)
            return AgentResponse(answer=r.answer, turn_type="number", conv_type=triage.conv_type,
                                 sub_questions=[QAPair(question=question, answer=r.answer)])
        pp = self.preprocess(question=question, history=hist_text)
        sub_qa: list[QAPair] = []
        for sq in pp.sub_questions:
            r = self.retriever(question=sq, document=document, history=hist_text)
            sub_qa.append(QAPair(question=sq, answer=r.answer))
        retrieved_text = "\n".join(f"Q: {p.question}\nA: {p.answer}" for p in sub_qa)
        calc = self.calculator(program=pp.program, retrieved=retrieved_text)
        return AgentResponse(answer=calc.answer, turn_type="program", conv_type=triage.conv_type,
                             program=pp.program, sub_questions=sub_qa)


# --- Dataset + sampling ----------------------------------------------
DATASET_PATH = Path("data/convfinqa_dataset.json")

@cache
def _load_raw() -> dict:
    return json.loads(DATASET_PATH.read_text())

def sample_records(n: int = 2, seed: int = 42, split: str = "dev") -> list[ConvFinQARecord]:
    rng = random.Random(seed)  # local RNG — reproducible
    records = _load_raw()[split]
    chosen = rng.sample(records, n)
    return [ConvFinQARecord.model_validate(r) for r in chosen]

def serialize_document(rec: ConvFinQARecord) -> str:
    parts = ["PRE-TEXT:", rec.doc.pre_text, "", "TABLE:"]
    for col, rows in rec.doc.table.items():
        parts.append(f"  {col}:")
        for k, v in rows.items():
            parts.append(f"    {k}: {v}")
    parts += ["", "POST-TEXT:", rec.doc.post_text]
    return "\n".join(parts)


# --- Evaluation -------------------------------------------------------
def configure_dspy() -> None:
    lm = dspy.LM("anthropic/claude-sonnet-4-6", api_key=os.environ["ANTHROPIC_API_KEY"])
    dspy.configure(lm=lm)

def numeric_match(pred: str, gold: float | str) -> bool:
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except (ValueError, TypeError):
        return str(pred).strip() == str(gold).strip()

def run_record(orch: ConvFinQAOrchestrator, rec: ConvFinQARecord) -> list[tuple[str, str, bool]]:
    """Walk all turns; return list of (predicted, gold, correct). Teacher-forced history."""
    doc = serialize_document(rec)
    history = ConversationHistory()
    rows: list[tuple[str, str, bool]] = []
    for q, gold in zip(rec.dialogue.conv_questions, rec.dialogue.executed_answers):
        resp = orch(question=q, document=doc, history=history)
        rows.append((resp.answer, str(gold), numeric_match(resp.answer, gold)))
        history.append(q, str(gold))  # teacher-forced — gold answer in history, not the model's
    return rows

def evaluate_random_records(n: int = 2, seed: int = 42) -> float:
    records = sample_records(n=n, seed=seed)
    orch = ConvFinQAOrchestrator()
    total, correct = 0, 0
    for rec in records:
        rows = run_record(orch, rec)
        rec_correct = sum(1 for _, _, ok in rows if ok)
        # report per-record (use logging or a print guarded by __main__)
        total += len(rows); correct += rec_correct
    return correct / total if total else 0.0


# --- __main__ ---------------------------------------------------------
if __name__ == "__main__":
    configure_dspy()
    acc = evaluate_random_records(n=2, seed=42)
    # print/log overall accuracy
```

### Integration Points

```yaml
DEPENDENCIES:
  - add to: pyproject.toml -> [project] dependencies
  - line: 'dspy>=2.5'
  - run: uv sync

ENV:
  - require: ANTHROPIC_API_KEY in .env (already loaded elsewhere; agent.py will
    read it from os.environ directly via dspy.LM)

NO CHANGES TO:
  - mcp/server_calculator.py (we mirror, not import)
  - provider.py (legacy, leave alone)
  - promptfooconfig.yaml (DSPy isn't promptfoo-evaluated here)
  - pyproject.toml's `main = "src.main:app"` entrypoint (a future CLI PRP fills it in)
```

## List of Tasks (in order)

```yaml
Task 1: Add dspy dependency
  MODIFY pyproject.toml:
    - Add 'dspy>=2.5' to [project].dependencies
  RUN: uv sync
  VALIDATE: uv run python -c "import dspy; print(dspy.__version__)"

Task 2: Create agent.py — single-file end-to-end implementation
  CREATE agent.py at the repo root with the 8 sections from the blueprint:
    1. Imports
    2. Pydantic models (QAPair, ConversationHistory, AgentResponse, plus
       ConvFinQARecord/Document/Dialogue/Features per the dataset schema)
    3. Calculator tools (6 fns + CALCULATOR_TOOLS)
    4. 4 dspy.Signature classes
    5. 4 dspy.Module agent classes (TriageAgent uses Predict, Preprocessing &
       Retriever use ChainOfThought, Calculation uses ReAct with tools)
    6. ConvFinQAOrchestrator with turn_type branching, returns AgentResponse
    7. Dataset loading + sample_records(n=2, seed=42) using local random.Random
    8. Evaluation: configure_dspy(), numeric_match, run_record (teacher-forced),
       evaluate_random_records, __main__
  VALIDATE: ruff check agent.py --fix && ruff format agent.py && uv run mypy agent.py

Task 3: Create tests/test_agent.py
  CREATE tests/test_agent.py:
    - Use dspy.utils.DummyLM to stub LM responses (no network calls)
    - test_calculator_tools_basic: add/subtract/divide/greater behaviours
    - test_conversation_history_as_text_format
    - test_orchestrator_number_path: triage->retriever only, no preprocess/calc
    - test_orchestrator_program_path: triage->preprocess->retriever (xN)->calc
    - test_orchestrator_returns_agent_response_shape
    - test_history_propagated_to_inner_modules: verify hist_text reaches retriever
    - test_numeric_match_tolerance: 0.14136 vs 0.1414 with tolerance 1e-3
    - test_sample_records_is_reproducible: same seed -> same ids twice
  VALIDATE: uv run pytest tests/test_agent.py -v

Task 4: Final sweep
  RUN: ruff check agent.py tests/test_agent.py --fix
  RUN: ruff format agent.py tests/test_agent.py
  RUN: uv run mypy agent.py
  RUN: uv run pytest tests/test_agent.py -v
  RUN: ANTHROPIC_API_KEY=... uv run python agent.py
  VERIFY:
    - Prints per-record + overall accuracy
    - Overall Execution Accuracy >= 50%
    - Re-run picks the same 2 record ids (reproducible)
```

## Validation Loop

### Level 1: Syntax & Style
```bash
ruff check agent.py tests/test_agent.py --fix
ruff format agent.py tests/test_agent.py
uv run mypy agent.py

# Expected: zero errors. Read errors and fix at the source — do not suppress.
```

### Level 2: Unit Tests (offline — DummyLM)
```bash
uv run pytest tests/test_agent.py -v -x

# -x stops at first failure. Iterate until green.
```

### Level 3: Smoke Test (offline imports + tool functions)
```bash
uv run python -c "
from agent import (
    add, subtract, CALCULATOR_TOOLS,
    QAPair, ConversationHistory, AgentResponse,
    TriageSignature, PreprocessingSignature, RetrieverSignature, CalculationSignature,
    TriageAgent, PreprocessingAgent, RetrieverAgent, CalculationAgent,
    ConvFinQAOrchestrator,
    sample_records, serialize_document, numeric_match,
)
assert add(1.0, 2.0) == 3.0
assert numeric_match('0.14136', 0.1414) is True
recs = sample_records(n=2, seed=42)
assert len(recs) == 2
recs_again = sample_records(n=2, seed=42)
assert [r.id for r in recs] == [r.id for r in recs_again], 'sampling not reproducible'
print('smoke ok — sampled ids:', [r.id for r in recs])
"
```

### Level 4: Integration Test (live API)
```bash
ANTHROPIC_API_KEY=... uv run python agent.py

# Expected: prints per-record accuracy lines and overall accuracy >= 50%
# Re-run and confirm the same 2 record ids are sampled
```

## Final Validation Checklist
- [ ] `ruff check agent.py tests/test_agent.py` — clean
- [ ] `ruff format --check agent.py tests/test_agent.py` — clean
- [ ] `uv run mypy agent.py` — clean
- [ ] `uv run pytest tests/test_agent.py -v` — all pass
- [ ] Smoke test (Level 3) prints `smoke ok` and the sampled ids
- [ ] `uv run python agent.py` runs to completion with no traceback
- [ ] Re-running picks the same 2 ids (reproducibility)
- [ ] Overall Execution Accuracy ≥ 50% on the 2 sampled records
- [ ] AgentResponse `.program` is populated for program-type turns
- [ ] Calculator tools are wired into `dspy.ReAct` (verify by tracing one program-type turn)

---

## Anti-Patterns to Avoid

- ❌ Don't call `dspy.configure(lm=...)` more than once per process
- ❌ Don't pass Pydantic models as `dspy.ReAct` tool arguments — use bare types (float, str, int)
- ❌ Don't use the global `random.sample()` — instantiate `random.Random(seed)` locally so DSPy/LiteLLM don't perturb the selection
- ❌ Don't import from `provider.py` — it's pydantic-ai legacy code; mirror prompts only
- ❌ Don't use `print()` inside `dspy.Module.forward()` — it interferes with eval output
- ❌ Don't add MCP/subprocess wiring for the calculator — plain Python functions work better with `dspy.ReAct`
- ❌ Don't add a CLI in this PRP — that's a follow-on PRP
- ❌ Don't add Logfire instrumentation in v1 — that's a follow-on PRP
- ❌ Don't add GEPA optimization in v1 — needs a working baseline first
- ❌ Don't skip teacher-forcing in evaluation — using model-generated history compounds errors and makes the metric noisy
- ❌ Don't split into multiple files in v1 — the goal is a single end-to-end `agent.py`. Modularization is a follow-on.

## Confidence Score: 7/10

**Strengths:**
- Single file means fewer moving parts and import-path bugs
- Reference prompts already in `provider.py`
- Comprehensive `claude_agent_spec.md` already authored
- Calculator tool already specified in `mcp/server_calculator.py`
- Reproducible sampling via local `random.Random(seed)`
- Teacher-forced history isolates retriever/calc errors from coreference errors

**Risks:**
- DSPy's behaviour with class-based `Signature` + `Literal` outputs depends on the LM's instruction-following; may need `desc` tuning
- `dspy.ReAct` with 6 tools may produce verbose traces; `max_iters=8` is a starting cap
- Naive `serialize_document` may exceed context for very large tables; mitigation deferred to follow-on PRP
- 2 random records is a small sample — variance can swing accuracy. Fixed seed makes results reproducible, not statistically meaningful
- DSPy version pinning matters — APIs evolved between 2.4 → 2.5 → 2.6. Pinning `>=2.5` keeps us on stable signatures
