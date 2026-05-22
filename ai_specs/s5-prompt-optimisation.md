name: "ConvFinQA Preprocess-Only Prompt Optimisation Harness"
description: |

## Purpose

Build a controlled, generator/evaluator-driven prompt-improvement loop that targets **only the Preprocess agent** inside the existing four-stage ConvFinQA pipeline. The Triage, Retriever, and Calculator system prompts stay frozen at their `dspy_gepa_real_20260502_005251` (v2) values — only `preprocess.predict.signature.instructions` is rewritten in v3.

The diagnosis question this harness answers, for every wrong **program-type** turn, is:

> Was the answer wrong because Preprocess emitted the **wrong program**? Or was the program right but its **sub-questions caused the wrong numbers** to flow into the calculation? Or is there a different reason that doesn't belong to Preprocess at all?

That last bucket — "not Preprocess" — is the harness's exclusion filter. Triage misclassifications, Retriever value mismatches that were unambiguously specified by Preprocess, Calculator execution errors, and pure formatting mismatches all get diagnosed but are NOT shown to the generator/evaluator agents. Only Preprocess-attributable failures populate the generator/evaluator example pools.

**Pipeline source of truth: `pydantic_agent.py` only.** Do not import from, modify, or run `dspy_agent.py` in this work. Stage capture, holdout re-evaluation, and the v3 comparison all go through `pydantic_agent.run_turn` / `evaluate_cached`. The optimised prompt JSON at `runs/<GEPA_NAME>/dspy_optimized_runner.json` is read-only data — the filename has a `dspy_` prefix purely for historical reasons.

**Run naming convention**:
- `v2` = the existing GEPA-optimised baseline (run dir `runs/gepa_real_20260502_005251/` and its `pydantic_predictions.csv`). All four agent prompts (`triage.predict`, `preprocess.predict`, `retriever.predict`, `calculator.react`) come from this run.
- `v3` = the prompt-optimised candidate produced by this PRP. **Only `preprocess.predict.signature.instructions` differs from v2.** The other three agents are byte-identical to v2 — the overlay JSON shadows the preprocess key only. The holdout re-evaluation writes its predictions under a `_v3` suffix (`pydantic_predictions_v3.csv`, `pydantic_predictions_joined_v3.csv`) so the v2-vs-v3 parity report is unambiguous.

**Scope filter on data shown to the generator/evaluator**: only turns where **gold `turn_type == "Program"`** AND the failure was attributed to Preprocess are eligible. Number-turn failures, cascade failures, and non-Preprocess failure modes never reach the generator or the evaluator.

Read this whole document. Pay extra attention to:
- `PRPs/EXAMPLE-prompt-opt.md` — the structural blueprint we are mirroring (different domain).
- The four-stage pipeline contract in `AGENTS.md` and `pydantic_agent.py`.
- The current optimised prompts in `runs/gepa_real_20260502_005251/dspy_optimized_runner.json`.
- The predictions schema documented in `AGENTS.md > Evaluation System`.

---

## Goal

Implement `prompt_improve_v2.py` (plus minimal supporting changes in `pydantic_agent.py`) that:

1. Captures per-stage outputs for every test conversation (`triage`, `preprocess`, `retriever`, `calculator`).
2. Diagnoses each *wrong* turn into one of the failure modes below; **only the two Preprocess-owned modes feed the harness**:
   - **`preprocess_wrong_program`** — gold `turn_type == "Program"` and the operation set / shape of `pred_program` does not match the gold `turn_program`. Preprocess emitted the wrong calculation.
   - **`preprocess_unclear_sub_questions`** — gold `turn_type == "Program"`, the program shape matches gold, but at least one retrieved value is clearly wrong AND the sub-question that produced it is ambiguous, missing the year/entity, or otherwise too vague for the Retriever. Preprocess wrote the right program but asked for the wrong (or under-specified) numbers.
   - other modes (`triage_turn_type`, `triage_conv_type`, `retriever_wrong_value`, `calculator_execution_error`, `formatting_mismatch`, `cascade`) are diagnosed for the dashboard but **excluded from generator/evaluator pools**.
3. Picks **30 unique evaluation conversations** that contain at least one Preprocess-attributable wrong **program** turn, splits them deterministically into `generator_dev` (15) / `evaluator_dev` (15). The remaining test conversations form `holdout_eval`.
4. Runs a single Pydantic AI generator/evaluator harness **for the Preprocess system prompt only**, using `deepseek-v4-pro` (more capable than the `deepseek-chat` used in the pipeline itself, so the optimiser has more headroom than the model it is optimising prompts for) — up to 3 generator/evaluator rounds.
5. Writes `prompts_candidate_v3.json` containing **exactly one key**: `preprocess.predict.signature.instructions`. The Triage, Retriever, and Calculator prompts are not touched.
6. Re-runs the evaluation through `pydantic_agent.py` with the v3 overlay applied via `PROMPTS_OVERLAY_PATH`, writing the holdout predictions under the `_v3` suffix and emitting a v2-vs-v3 comparison so the impact of the Preprocess prompt change is the only variable.

## Why

- The current GEPA-optimised prompts have been frozen since `gepa_real_20260502_005251` and ship with `optimized_test_score = 65.26%` overall, with the largest gap on multi-step program turns. Per-agent inspection of failures suggests the Preprocess stage is the single biggest source of these losses: it emits programs that miss `multiply(..., 100)` for percentage answers, swaps the direction of `subtract`, or asks the Retriever for "the value" when a year and entity are required.
- GEPA is a 5–9 hr black-box search; we want a fast, focused, **agent-attributed** prompt edit loop that a human can review in minutes, with all other agents held constant so any v2→v3 delta is attributable to the Preprocess prompt alone.
- The example PRP (CUAD legal extraction) demonstrated this works in a single-LLM-call setting. ConvFinQA is harder because a wrong final answer can come from any of 4 agents, so attributing failures to a specific agent — and then only showing the agent's own failures to its prompt-improvement loop — is the contribution.

## What

A new top-level Python script:

```
prompt_improve_v2.py
```

with these behaviours:

- `--collect-traces` (default first run): re-runs `pydantic_agent.run_turn` over the test set with stage capture, writing `runs/<GEPA_NAME>/prompt_optim_v2/stage_traces.jsonl`.
- `--diagnose`: labels each wrong turn with a failure mode (Preprocess-owned modes are the harness target; everything else is diagnosed-but-excluded).
- `--split`: deterministically partitions conversations that have at least one Preprocess-attributable failure on a program turn into `generator_dev` / `evaluator_dev`. The remaining test conversations form `holdout_eval`.
- `--optimise`: runs the generator/evaluator loop for **the Preprocess agent only**, up to 3 rounds. Generator and evaluator only see Preprocess-attributable, `turn_type == "Program"` failures.
- `--score-holdout`: applies the v3 overlay (Preprocess prompt only), re-runs the pipeline on `holdout_eval` through `pydantic_agent.run_turn`, writes `pydantic_predictions_v3.csv`, and prints v2 vs v3 comparisons.

Default invocation (`uv run python prompt_improve_v2.py`) runs all phases in order, idempotently (re-uses cached traces / splits unless `--force` is passed).

### Success Criteria

- [ ] `runs/<GEPA_NAME>/prompt_optim_v2/stage_traces.jsonl` exists with one record per `(report_id, turn_index)` containing the four stage outputs.
- [ ] `runs/<GEPA_NAME>/prompt_optim_v2/diagnoses.csv` labels every wrong turn with one of: `triage_turn_type`, `triage_conv_type`, `preprocess_wrong_program`, `preprocess_unclear_sub_questions`, `retriever_wrong_value`, `calculator_execution_error`, `formatting_mismatch`, `cascade`. The Preprocess-owned rows MUST be a subset of rows where gold `turn_type == "Program"`.
- [ ] Generator/evaluator pools contain **only rows where `failure_mode in {preprocess_wrong_program, preprocess_unclear_sub_questions}` AND gold `turn_type == "Program"`**. This is asserted in `splits.json` and tested.
- [ ] `splits.json` records the 30 conversations used for `generator_dev` + `evaluator_dev` (15 each, deterministic from `random_state=42`) and the rest as `holdout_eval`.
- [ ] `category_runs.jsonl`, `evaluator_reviews.jsonl`, `accepted_patches.jsonl`, `rejected_patches.jsonl`, `prompt_diffs.jsonl` written for `preprocess` only (no triage/retriever/calculator entries).
- [ ] `prompts_candidate_v3.json` is a partial JSON containing **exactly one key path** — `preprocess.predict.signature.instructions`. No other keys present. Test asserts this.
- [ ] Re-evaluation on `holdout_eval` produces `pydantic_predictions_v3.csv` + `pydantic_predictions_joined_v3.csv` and a `v2_v3_comparison.csv` slice table, with v3 overall accuracy not regressing more than 2 pp vs v2 on the holdout.
- [ ] `v2_v3_failure_mode_delta.csv` shows the change specifically for `preprocess_wrong_program` and `preprocess_unclear_sub_questions` rows — the targeted improvement signal.
- [ ] `prompt_review_dashboard.html` renders the single Preprocess loop with v2 prompt, v3 prompt, unified diff, and the generator/evaluator example pools.
- [ ] `uv run pytest tests/test_prompt_improve_v2.py` passes (deterministic mode, no LLM calls).

## All Needed Context

### Documentation & References

```yaml
# MUST READ
- file: PRPs/EXAMPLE-prompt-opt.md
  why: |
    Structural blueprint for the generator/evaluator harness — splits, request
    shapes, dashboard, harness guardrails. Mirror it. Differences for ConvFinQA:
    we optimise PER AGENT (triage / preprocess) not PER CATEGORY, the answer
    format is numeric, and failures must be attributed to a specific agent
    before a prompt change is proposed. Note the `Harness Guardrails` and
    `Generator/Evaluator Harness` sections in particular.

- file: AGENTS.md
  why: |
    Authoritative description of the 4-stage pipeline, the predictions CSV
    schema, the GEPA run-artifact layout, the Vite-proxy invariant. The
    "Four-Stage Pipeline" section explains exactly what each agent owns —
    that ownership IS the failure-mode taxonomy.

- file: pydantic_agent.py
  why: |
    The ONLY pipeline we instrument. Patterns to copy / hooks to use:
      * `_load_optimized_prompts` — JSON load shape (lines 68–96). Patch
        this function (Task 5) to honour `PROMPTS_OVERLAY_PATH`.
      * `run_turn` — the canonical 4-stage call sequence; the new `trace_turn`
        re-implements it inline (lines 229–287).
      * `stream_turn` — already yields per-stage events; mirror its structure
        for stage capture (lines 331–434).
      * `evaluate_cached(examples, ...)` — drives the holdout re-eval; we call
        it with the test examples filtered to the holdout conversation set
        (lines 614–634).
      * `compare_prediction_runs(...)` — reuse for v2-vs-v3 parity reporting
        (lines 637–755). It already supports custom labels via
        `left_label` / `right_label`.
    `pydantic_agent.py` currently imports a few names from `dspy_agent` at the
    top (`_DOCS`, `CALCULATOR_TOOLS`, `ConversationHistory`, `QAPair`,
    `analyze_predictions`, `conv_examples_test`, `numeric_match`, `qa_data`).
    Those imports remain; we treat them as if they were defined inside
    pydantic_agent — the harness code itself never touches dspy_agent
    directly. Even better: this PRP also asks you to replace the
    `conv_examples_test` import with `api_eval.load_conv_examples_test()`
    so the test selection is decoupled from dspy_agent (see Task 5b).
    DO NOT change the wire format ("[[ ## name ## ]]" blocks); the optimised
    prompts were tuned against it.

- file: api_eval.py
  why: |
    Self-contained loader and metric — does NOT import dspy_agent. Use it for:
      * `numeric_match` (lines 32–36) — verbatim. Round-to-int equality with
        a string fallback.
      * `load_conv_examples_test()` (lines 53–66) — independent rebuild of
        the test split (`n=100, random_state=42`, then 60/40 train/test).
        This is the test set we score against; it is what powers
        `api_predictions.csv` and is API-driven, no dspy import.
      * `_join_predictions` / `compare_model_accuracies` (lines 92–180) —
        copy the slice-table pattern for the v2-vs-v3 comparison.

- file: runs/gepa_real_20260502_005251/dspy_optimized_runner.json
  why: |
    The "v1" prompts. Layout:
      raw["triage.predict"]["signature"]["instructions"] -> str
      raw["preprocess.predict"]["signature"]["instructions"] -> str
    Our candidate v2 file MUST shadow these exact keys so the existing loader
    in pydantic_agent.py can pick them up via a `--prompts-overlay` argument.

- file: runs/gepa_real_20260502_005251/dspy_predictions_joined.csv
  why: |
    Has the columns we diagnose against:
      gold_answer, pred_answer, correct, pred_turn_type, pred_conv_type,
      q_order, turn_type (gold, capitalised "Number"/"Program"),
      qa_split (gold bool — True = Type II), conv_type (gold from the join).
    NOTE: gold turn_type is capitalised, predicted is lower-case. Normalise
    before comparison or every triage row will look like a misclassification.

- url: https://ai.pydantic.dev/agents/
  why: Pydantic AI Agent constructor + structured output (output_type=BaseModel).

- url: https://ai.pydantic.dev/multi-agent-applications/
  why: Multi-agent patterns; we use two independent agents (generator, evaluator).

- url: https://ai.pydantic.dev/api/models/openai/
  why: |
    OpenAIChatModel + OpenAIProvider with custom `base_url` — same pattern
    pydantic_agent.py:151 already uses for DeepSeek (api.deepseek.com/v1).
    Use `deepseek-chat` (matches what the prompts were tuned against).

- url: https://www.deepseek.com/api-docs/
  why: DeepSeek API surface — OpenAI-compatible. Set DEEPSEEK_API_KEY.

- doc: https://docs.python.org/3/library/difflib.html
  section: difflib.unified_diff
  critical: |
    Use unified_diff for `prompt_diffs.jsonl` and the dashboard. Splitlines
    keepends=True or you'll get malformed diff blocks.
```

### Current Codebase tree (relevant subset)

```text
.
├── AGENTS.md
├── CLAUDE.md
├── PRPs/
│   ├── EXAMPLE-prompt-opt.md              # blueprint to mirror
│   ├── EXAMPLE_multi_agent_prp.md
│   └── templates/prp_base.md
├── api_eval.py
├── app.py
├── data.py
├── data/convfinqa_dataset.json
├── pydantic_agent.py                       # only pipeline used by this PRP
├── pyproject.toml
├── runs/
│   └── gepa_real_20260502_005251/
│       ├── api_predictions.csv
│       ├── api_predictions_joined.csv
│       ├── dspy_optimized_runner.json     # v1 prompts source
│       ├── dspy_predictions_joined.csv    # gold + pred turn_type/conv_type
│       ├── pydantic_predictions.csv
│       └── pydantic_predictions_joined.csv
└── tests/
```

### Desired Codebase tree (additions)

```text
.
├── prompt_improve_v2.py                          # NEW — entry point
├── prompt_optim/                                 # NEW — package
│   ├── __init__.py
│   ├── tracing.py                                # stage capture wrapper around run_turn
│   ├── diagnose.py                               # failure-mode attribution
│   ├── splits.py                                 # deterministic conversation splitter
│   ├── harness.py                                # Pydantic AI generator/evaluator agents
│   ├── prompts.py                                # generator/evaluator system prompts (constants)
│   ├── models.py                                 # FailureExample, PromptPatch, PromptReview, etc.
│   ├── apply.py                                  # build candidate prompts JSON; load overlay
│   ├── score.py                                  # holdout re-eval + delta tables
│   └── dashboard.py                              # static HTML dashboard renderer
├── runs/<GEPA_NAME>/prompt_optim_v2/             # NEW — artefact dir (per run)
│   ├── stage_traces.jsonl
│   ├── diagnoses.csv
│   ├── splits.json
│   ├── category_runs.jsonl                       # one record per (agent, loop)
│   ├── evaluator_reviews.jsonl
│   ├── accepted_patches.jsonl
│   ├── rejected_patches.jsonl
│   ├── prompt_diffs.jsonl
│   ├── prompts_candidate_v3.json                # overlay onto the v2 base
│   ├── pydantic_predictions_v3.csv              # holdout re-eval, v3 prompts
│   ├── pydantic_predictions_joined_v3.csv
│   ├── v2_v3_comparison.csv
│   ├── v2_v3_failure_mode_delta.csv
│   └── prompt_review_dashboard.html
└── tests/
    └── test_prompt_improve_v2.py                 # NEW — deterministic-mode tests
```

### Known Gotchas of our codebase & Library Quirks

```python
# CRITICAL: the test set MUST come from api_eval.load_conv_examples_test()
# (n=100, random_state=42, 60/40 split -> the held-out 40 conversations).
# This is the same selection used to produce api_predictions.csv and is
# self-contained (no dspy_agent import). Do NOT recreate the dspy_agent
# variant (n=200 + 60 additional) — this PRP intentionally uses the
# pydantic/api test set so the comparison v2 vs v3 stays inside the
# pydantic_agent universe.

# CRITICAL: gold turn_type strings are capitalised "Number"/"Program" (data.py:26),
# predicted are lowercase "number"/"program" (TriageOut Literal). Normalise via
# .str.lower() on both sides before computing diagnosis.

# CRITICAL: gold conv_type comes from `qa_split` column where True == "Type II".
# api_eval.py:73 maps it; mirror that.

# CRITICAL: the 4 agents share one DeepSeek client (LM_MINI in pydantic_agent.py).
# The optimisation harness MUST also use deepseek-chat — the prompts were tuned
# for that model and switching breaks the premise. Reuse pydantic_agent's
# `_deepseek_provider` if you import it; otherwise build a fresh one with the
# same base_url / api_key. Reuse is preferred to avoid two clients in flight.

# CRITICAL: pydantic_agent.py loads .env at import time and (today) imports a
# handful of helpers from dspy_agent — which in turn constructs
# `dspy.LM(deepseek/...)` at module scope. That cost is paid once, transparently,
# the first time we import pydantic_agent. The harness must NEVER import
# dspy_agent itself; importing pydantic_agent is fine. Keep
# `from dotenv import load_dotenv; load_dotenv(Path.home() / ".env")` BEFORE
# any pydantic_agent import. Same pattern as pydantic_agent.py:34-36.

# GOTCHA: pydantic_agent's `run_turn` mutates `conversation` in place (appends
# the answer). For tracing we must NOT double-append, so the trace wrapper
# should call run_turn directly and capture the per-stage outputs by attaching
# Logfire span attributes OR by re-implementing the 4 calls inline in
# `tracing.py` mirroring run_turn. Re-implement inline — it is 60 lines and
# avoids monkey-patching.

# GOTCHA: pydantic-ai's `agent.run(...)` is async. The whole script is async.
# Use `asyncio.Semaphore` to cap concurrency; api_eval.py:225 caps at 8. Mirror
# that for stage-trace collection. The harness loop itself is sequential per
# agent (only 2 agents to optimise) — no concurrency needed.

# GOTCHA: the optimised "instructions" string in dspy_optimized_runner.json
# is just a system-prompt body — it does NOT include any auto-injected
# reasoning field. The Pydantic port adds `reasoning` to TriageOut /
# PreprocessOut via the `output_type=` Pydantic model. When we replace
# `instructions` we are only replacing the system-prompt portion; the
# Pydantic output models and wire format ("[[ ## name ## ]]") are unchanged.

# GOTCHA: dspy_optimized_runner.json's `calculator.react` value is a JSON-
# stringified object with "instruction" inside. Don't try to optimise it
# in this PRP — out of scope. Triage and Preprocess are plain string
# instructions and are the only two we touch.

# GOTCHA: numeric_match (api_eval.py:32-36) does
# round(float(pred)) == round(float(gold)) — so "117.0" vs "117" matches but
# "117%" vs "117" does NOT (float() raises on '%'). For diagnosis,
# `formatting_mismatch` should detect the case where stripping '%' on both
# sides would have made the answer correct. Always import this from api_eval,
# never from dspy_agent.

# GOTCHA: the gold `turn_program` in `qa_data` (which we obtain via
# `data.training_data()`, NOT via dspy_agent) uses DSL syntax with intermediate
# refs like `#0`, e.g. `divide(subtract(A, B), B), multiply(#0, 100)`. The
# Preprocess agent emits a different DSL using A/B/... letters. Comparing
# programs literally is hopeless. Compare:
#   * the OPERATIONS multiset (e.g., {divide, subtract, multiply}),
#   * the OPERAND COUNT (number of distinct sub_questions / table cells used),
#   * presence of `multiply(..., 100)` for percentage answers.
# data.py:28-30 already extracts `turn_program_actions` and
# `turn_program_calcs` (the operation list) — reuse those by calling
# `from data import training_data` directly. data.py is dependency-light and
# does not pull in dspy.

# GOTCHA: the dataset has ~60% turns with cross-turn dependencies. A wrong
# answer at turn k often poisons k+1, k+2, ... Diagnose only the FIRST wrong
# turn per conversation as the root cause; subsequent failures may be cascade
# effects, not new root-cause failures. Tag cascade rows with a
# `cascade_of=<turn_index>` field but do NOT use them for prompt improvement.

# GOTCHA: pydantic-ai Agent expects `instructions` (or `system_prompt`) as
# kwargs. We pass the v1 instructions string from JSON straight in. Do not
# wrap/reformat it.

# GOTCHA: ruff is in `pyproject.toml` with `extend-select = ["T201"]`
# (banning print). Use `# ruff: noqa: T201` at top of prompt_improve_v2.py
# and prompt_optim/score.py — they print progress like the existing scripts.
```

## Implementation Blueprint

### Data models and structure

Create in `prompt_optim/models.py`:

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

AgentName = Literal["preprocess"]   # only one agent is optimised in this PRP
FailureMode = Literal[
    "preprocess_wrong_program",          # HARNESS TARGET
    "preprocess_unclear_sub_questions",  # HARNESS TARGET
    "triage_turn_type",                  # diagnosed-but-excluded
    "triage_conv_type",                  # diagnosed-but-excluded
    "retriever_wrong_value",             # diagnosed-but-excluded
    "calculator_execution_error",        # diagnosed-but-excluded
    "formatting_mismatch",               # diagnosed-but-excluded
    "cascade",                           # diagnosed-but-excluded
    "gold_or_metric_issue",              # diagnosed-but-excluded
]
PREPROCESS_FAILURE_MODES: set[str] = {
    "preprocess_wrong_program",
    "preprocess_unclear_sub_questions",
}
SplitName = Literal["generator_dev", "evaluator_dev", "holdout_eval"]


class StageTrace(BaseModel):
    report_id: str
    turn_index: int
    question: str
    history_text: str
    triage_reasoning: str
    pred_turn_type: Literal["number", "program"]
    pred_conv_type: Literal["Type I", "Type II"]
    pred_sub_questions: list[str] = []      # empty when number turn
    pred_program: str = ""                   # empty when number turn
    retrieved: list[dict[str, str]] = []     # [{question, answer}, ...]
    calc_trajectory: dict | None = None
    pred_answer: str
    gold_answer: str
    gold_turn_type: Literal["number", "program"]
    gold_conv_type: Literal["Type I", "Type II"]
    gold_turn_program: str
    gold_program_ops: list[str]              # extracted via data.py logic
    correct: bool


class FailureExample(BaseModel):
    row_id: str  # "<report_id>__<turn_index>"
    report_id: str
    turn_index: int
    question: str
    history_text: str
    gold_answer: str
    pred_answer: str
    gold_turn_type: str
    pred_turn_type: str
    gold_turn_program: str
    pred_program: str
    pred_sub_questions: list[str]
    retrieved: list[dict[str, str]]
    failure_mode: FailureMode
    cascade_of: int | None = None
    notes: list[str] = []


class PromptPatchRequest(BaseModel):
    agent: AgentName
    current_instructions: str
    failure_mode_summary: dict[str, int]
    generator_examples: list[FailureExample] = Field(max_length=15)
    original_generator_guide: str
    previous_generated_prompt: str | None = None
    evaluator_feedback: str | None = None
    loop_index: int


class PromptPatch(BaseModel):
    agent: AgentName
    failure_analysis: list[str]
    revised_instructions: str
    expected_improvements: list[str]
    regression_risks: list[str]
    changed_rules: list[str]


class PromptReviewRequest(BaseModel):
    agent: AgentName
    current_instructions: str
    generator_instructions: str
    generator_patch: PromptPatch
    evaluator_examples: list[FailureExample] = Field(max_length=15)
    loop_index: int


class PromptReview(BaseModel):
    decision: Literal["accept", "revise", "reject"]
    generalization_score: float = Field(ge=0.0, le=1.0)
    rationale: list[str]
    likely_fixes: list[str]
    likely_regressions: list[str]
    requested_changes: list[str]
```

### List of tasks (in order)

```yaml
Task 1 — Stage-trace capture
CREATE prompt_optim/__init__.py:
  - empty file (package marker)

CREATE prompt_optim/tracing.py:
  - MIRROR pattern from: pydantic_agent.py:run_turn (lines 229-287)
  - Imports allowed: `pydantic_agent` (for `_render_chat_inputs`,
    `triage_agent`, `preprocess_agent`, `retriever_agent`, `calculator_agent`,
    `_DOCS`, `ConversationHistory`), `api_eval` (for `numeric_match`,
    `load_conv_examples_test`), `data` (for `training_data`).
    DO NOT import `dspy_agent` directly anywhere in prompt_optim/*.
  - Provide async def `trace_turn(question, report_id, conversation) -> StageTrace`
    that re-implements the four agent calls inline (NOT via run_turn) and
    returns the structured trace plus mutates `conversation` once. Reusing
    the agent instances from pydantic_agent guarantees wire format parity.
  - Provide async def `collect_traces(examples, qa_data, gepa_name, max_concurrency=8)`
    that walks every conversation, threads ConversationHistory once per
    conversation, writes `runs/<GEPA_NAME>/prompt_optim_v2/stage_traces.jsonl`,
    and returns a list[StageTrace].
  - GOTCHA: For Number turns, pred_program/pred_sub_questions/retrieved must
    still be filled (with retriever output), but pred_program stays empty
    string to match the predictions CSV convention.
  - GOTCHA: capture `triage.reasoning` and the calculator trajectory; stream_turn
    in pydantic_agent already shows how to extract the trajectory via
    `calc_result.all_messages()` — copy that.

Task 2 — Failure-mode diagnosis
CREATE prompt_optim/diagnose.py:
  - Provide def `diagnose(traces: list[StageTrace]) -> list[FailureExample]`
  - Apply the rules in this order, FIRST MATCH WINS:
      1. correct=True   -> skip (do not return).
      2. cascade detection: within a single conversation, if any earlier turn
         was wrong, tag this turn as `cascade` with cascade_of=
         <earliest_wrong_turn_index>. Cascade rows are kept for dashboard
         display but excluded from generator/evaluator pools. Cascade
         detection runs FIRST so we never label a downstream collateral
         failure as a Preprocess root cause.
      3. `formatting_mismatch` if numeric_match fails BUT
         numeric_match(pred.rstrip('%'), gold.rstrip('%')) succeeds, OR if
         stripping commas/$/units would match.
      4. `triage_turn_type` if pred_turn_type.lower() != gold_turn_type.lower().
         (gold is "Program"/"Number" — lower-case before comparing.)
         These rows are EXCLUDED from the harness — Triage is held constant.
      5. `preprocess_wrong_program` (HARNESS TARGET) if gold_turn_type=="program"
         AND the operation multiset of pred_program differs from the gold
         turn_program. Use data.py's existing regex (`turn_program_calcs`) to
         extract operation lists from both. Sub-rules to record in `notes`:
           - missing `multiply` on a percent-asking question (gold contains
             `multiply(...,100)` or the question contains "%"/"percent"/
             "percentage" but pred does not),
           - swapped subtract direction (gold subtract(B,A), pred subtract(A,B)
             on a "change/decline/growth" question),
           - extra/missing op (count of distinct ops differs),
           - reuses literal numbers from history that should have been
             re-fetched.
      6. `preprocess_unclear_sub_questions` (HARNESS TARGET) if
         gold_turn_type=="program" AND the operation multiset of pred_program
         MATCHES the gold turn_program but the final answer is still wrong
         AND at least one retrieved answer is clearly bad (None / empty /
         off by an order of magnitude vs numbers visible in the document for
         the question's year/entity). Sub-rules in `notes`:
           - sub-question missing year/period that the question implies,
           - sub-question missing entity/metric specifier (e.g., "the value"
             instead of "weighted average grant date fair value of restricted
             stocks in 2007"),
           - duplicate sub-question wording across two letters that resolve
             to different operands in the gold program.
      7. `retriever_wrong_value` (excluded from harness) if pred_program ops
         match gold AND a retrieved value is wrong but the sub-question is
         unambiguous (year + entity both present, no obvious phrasing
         issue). This is a Retriever-side fault, NOT Preprocess.
      8. `calculator_execution_error` (excluded) otherwise — program looks
         right, retrieved looks right, final answer wrong. Calculator-owned.
      9. `triage_conv_type` (excluded) is NOT used as a primary label in this
         PRP — `conv_type` only routes the Preprocess prompt, never gates
         correctness on its own. Drop it.
  - Write `diagnoses.csv` with columns:
      report_id, turn_index, q_order, gold_answer, pred_answer,
      pred_turn_type, gold_turn_type, pred_program, gold_turn_program,
      pred_sub_questions, retrieved_summary,
      failure_mode, cascade_of, harness_eligible, notes
    where `harness_eligible = (failure_mode in PREPROCESS_FAILURE_MODES
    and gold_turn_type=="program" and cascade_of is None)`.

Task 3 — Deterministic conversation splitter (Preprocess-only filter)
CREATE prompt_optim/splits.py:
  - Provide def `build_splits(failures: list[FailureExample],
    n_dev_convs=30, seed=42) -> dict[SplitName, list[str]]`.
  - ELIGIBILITY: only failures where
        failure_mode in PREPROCESS_FAILURE_MODES
        AND gold_turn_type == "program"
        AND cascade_of is None
    can ANCHOR a conversation in `generator_dev` / `evaluator_dev`. This is
    the "30 unique evaluation conversations… for turn_type = Program"
    requirement.
  - SELECTION RULE:
      * Take the set of unique report_ids that have at least one harness-
        eligible failure, sort them, then
        `random.Random(seed).sample(sorted_ids, k=min(30, len(sorted_ids)))`.
      * First 15 conversations -> generator_dev, next 15 -> evaluator_dev.
        If fewer than 30 are available, split what exists 50/50 (record the
        actual counts in splits.json).
      * Within a chosen conversation, ALL harness-eligible rows go to that
        conversation's split. Non-eligible rows in those same conversations
        (cascade, triage, retriever, calculator, formatting, number-turn
        failures) are written to splits.json under
        `excluded_in_dev_conversations` for traceability but never reach
        the harness.
  - HOLDOUT: every test conversation NOT chosen for dev (whether it had
    failures or not) becomes `holdout_eval`. The holdout is full
    conversations re-run end-to-end through pydantic_agent — not just
    failure rows — because we score v3 against v2 on the same input set.
  - Save `splits.json` with shape:
      {
        "generator_dev_conversations": [...],     # 15 report_ids
        "evaluator_dev_conversations": [...],     # 15 report_ids
        "holdout_eval_conversations":  [...],     # the rest
        "row_assignments": {row_id: split_name},  # only harness-eligible rows
        "excluded_in_dev_conversations": [row_id, ...],
        "counts": {
          "harness_eligible_total": int,
          "generator_dev_rows": int,
          "evaluator_dev_rows": int,
          "holdout_eval_conversations": int
        }
      }
  - DO NOT split a single conversation across generator_dev and evaluator_dev
    — turn-level dependencies make that meaningless.
  - DO NOT include any row where gold_turn_type != "program" in the dev
    pools, even if it appears in a chosen conversation. Tests assert this.

Task 4 — Generator/evaluator agents (Pydantic AI) — Preprocess only
CREATE prompt_optim/prompts.py:
  - Two top-level constants:
      GENERATOR_SYSTEM_PROMPT_PREPROCESS
      EVALUATOR_SYSTEM_PROMPT
  - GENERATOR_SYSTEM_PROMPT_PREPROCESS must:
      * State the agent's role in the pipeline: Preprocess receives the
        question, conversation history, and conv_type; it emits an ordered
        list of `sub_questions` (verbatim retrieval lookups, no arithmetic)
        and a `program` over A/B/C/... using only `add`, `subtract`,
        `multiply`, `divide`, `exp`, `greater`.
      * State the failure modes the harness has observed:
        `preprocess_wrong_program` (wrong operations, wrong direction,
        missing `multiply(...,100)` for percent answers) and
        `preprocess_unclear_sub_questions` (vague sub-questions caused the
        Retriever to return wrong numbers).
      * Forbid behaviour changes that would break the pipeline contract:
        do NOT bypass the Retriever, do NOT do arithmetic in sub-questions,
        do NOT change the program operator vocabulary, do NOT alter the
        "[[ ## name ## ]]" wire format expectations of downstream agents.
      * Allow the generator to copy verbatim phrasing from prior history
        sub-questions when reusing cached values, since that is how the
        existing optimised prompt already encourages history reuse.
  - EVALUATOR_SYSTEM_PROMPT mirrors the example PRP: skeptical reviewer that
    checks generalisation against unseen evaluator examples, rejects broad
    rewrites and prompt bloat, and is allowed to accept / revise / reject.
  - Reuse the request/response shapes from PRPs/EXAMPLE-prompt-opt.md
    "Generator request" / "Evaluator request" sections, swapping CUAD-
    specific fields for ConvFinQA's (sub_questions, retrieved values, gold
    program ops, conv_type, etc.).

CREATE prompt_optim/harness.py:
  - Build the OpenAIChatModel for `deepseek-v4-pro` via the same provider
    pattern as pydantic_agent.py:151-155. The pipeline itself runs
    `deepseek-chat`; using `deepseek-v4-pro` for the optimiser gives the
    generator/evaluator more reasoning headroom than the model whose
    prompts we are improving.
      _deepseek_provider = OpenAIProvider(
          base_url="https://api.deepseek.com/v1",
          api_key=os.environ["DEEPSEEK_API_KEY"],
      )
      LM_PRO = OpenAIChatModel("deepseek-v4-pro", provider=_deepseek_provider)
  - Two agents: `generator_agent` (output_type=PromptPatch,
    instructions=GENERATOR_SYSTEM_PROMPT_PREPROCESS) and `evaluator_agent`
    (output_type=PromptReview, instructions=EVALUATOR_SYSTEM_PROMPT).
  - Async def `improve_preprocess(current_instructions: str,
      generator_examples: list[FailureExample],
      evaluator_examples: list[FailureExample],
      out_dir: Path) -> PromptPatch | None`:
        Run up to 3 generator/evaluator loops following the example PRP
        guardrails:
          - generator never sees evaluator_examples,
          - evaluator can request at most 2 revisions; loop 3 is accept/reject,
          - on accept -> return patch (one PromptPatch object),
          - on reject -> return None (keep v2 preprocess prompt unchanged).
        After every loop, append to category_runs.jsonl,
        evaluator_reviews.jsonl. On accept, append to accepted_patches.jsonl
        and prompt_diffs.jsonl (unified diff vs current_instructions). On
        terminal reject, append to rejected_patches.jsonl.
  - DO NOT add a triage / retriever / calculator entry point. There is one
    function and it optimises Preprocess.
  - PRE-CHECK: assert that every example in generator_examples and
    evaluator_examples has gold_turn_type == "program" and failure_mode in
    PREPROCESS_FAILURE_MODES. Refuse to run otherwise (raise ValueError).

Task 5 — Apply candidate prompts (v3 overlay — Preprocess only)
CREATE prompt_optim/apply.py:
  - Provide def `write_candidate_v3(patch: PromptPatch | None,
      v2_path: Path, out_path: Path)`:
        Loads `runs/<GEPA_NAME>/dspy_optimized_runner.json` (the v2 baseline)
        only to validate structure. Writes a partial JSON containing
        EXACTLY one key path:
          {
            "preprocess.predict": {"signature": {"instructions": "..."}}
          }
        If `patch is None` (evaluator rejected), write
        `{}` and log "no v3 change — using v2 preprocess prompt verbatim".
        DO NOT include triage.predict, retriever.predict, or
        calculator.react keys under any circumstance — those agents are
        held constant. Tests assert this.
  - Provide def `apply_overlay(base: dict, overlay: dict) -> dict` that
    deep-merges the partial overlay onto the v2 JSON in memory. Triage,
    Retriever, Calculator entries in `base` are left untouched.
  - Provide def `rebuild_pipeline_agents(overlay_path: Path) -> dict[str, Agent]`:
        Loads v2 JSON, deep-merges the overlay, then constructs FOUR fresh
        `pydantic_ai.Agent(...)` objects (triage / preprocess / retriever /
        calculator) using the same model, output types, and the (possibly
        overlaid) instructions. Returns them by name. Even though only
        Preprocess changed, all four agents are reconstructed for symmetry
        with the existing pydantic_agent.run_turn dispatch path. This is
        required because `pydantic_ai.Agent` binds `instructions` at
        construction time; mutating the existing module-level agents from
        pydantic_agent.py is not supported.

MODIFY pydantic_agent.py:
  - Add support for a `PROMPTS_OVERLAY_PATH` env var. If set,
    `_load_optimized_prompts` reads that JSON, deep-merges it onto the
    base run JSON, and extracts instructions from the merged dict.
  - Add an optional `agents: dict[str, Agent] | None = None` kwarg to
    `run_turn` (and `stream_turn`). When provided, those agents are used
    instead of the module-level ones — this is the hook score.py uses.
  - Single localised change near lines 53–99 plus a small kwarg-threading
    change in `run_turn`. Preserve existing behaviour when the env var is
    unset and `agents=None`.

Task 5b — Decouple test-set selection from dspy_agent
MODIFY pydantic_agent.py:
  - Replace the `from dspy_agent import ... conv_examples_test ...` line with
    a local helper that calls `api_eval.load_conv_examples_test()` (or
    inline the same selection code). All other imports from dspy_agent stay
    where they are, but the test split is now sourced from api_eval, which
    is the only test-set source the harness uses.
  - Update the `__main__` block that currently does
    `evaluate_cached(conv_examples_test, ...)` to use the new local helper
    return value. Behaviour after this change: `pydantic_agent.py` can be
    run end-to-end without touching dspy.

Task 6 — Holdout scoring (v2 vs v3)
CREATE prompt_optim/score.py:
  - async def `score_holdout(holdout_conversation_ids, gepa_name)`:
      * `os.environ["PROMPTS_OVERLAY_PATH"] = str(.../prompts_candidate_v3.json)`
      * Build fresh agents via `apply.rebuild_pipeline_agents(overlay_path)`.
      * For each holdout conversation, walk turns by calling
        `pydantic_agent.run_turn(question, report_id, conversation,
        agents=fresh_agents)`. Concurrency cap = 8 (mirror api_eval).
      * Write `runs/<GEPA_NAME>/prompt_optim_v2/pydantic_predictions_v3.csv`
        with the same schema as `pydantic_predictions.csv` (the v2 baseline).
        Then call `api_eval._join_predictions(...)` to produce
        `pydantic_predictions_joined_v3.csv`.
  - def `compare_v2_v3(out_dir, gepa_name)`:
      * v2 frame = `runs/<GEPA_NAME>/pydantic_predictions_joined.csv` filtered
        to the holdout conversation set.
      * v3 frame = `pydantic_predictions_joined_v3.csv`.
      * Reuse `pydantic_agent.compare_prediction_runs(left_csv=v2, right_csv=v3,
        left_label="v2", right_label="v3", output_name="v2_v3_comparison.csv")`.
      * Then attach the v2 failure_mode (from diagnoses.csv) to each row and
        emit a per-failure-mode delta table — "did the prompt edit fix the
        targeted failure modes?".
  - GOTCHA: `compare_prediction_runs` raises if the two CSVs don't cover the
    same rows. Filter the v2 CSV to the holdout conversations BEFORE passing
    it in.

Task 7 — Static dashboard
CREATE prompt_optim/dashboard.py:
  - def `render(out_dir: Path)`: builds prompt_review_dashboard.html from
    the JSONL artefacts only (no agent state). Single file output, inline
    CSS, no external assets.
  - Sections:
      - Summary header: gepa_name, accept/reject decision for Preprocess,
        v2 vs v3 holdout scores, breakdown of harness-eligible failure
        counts by mode.
      - Held-constant panel: lists the three agents whose prompts were
        NOT touched (triage, retriever, calculator) and includes a one-line
        excerpt of each v2 prompt to make it explicit.
      - Preprocess panel:
          * v2 prompt textarea (read-only)
          * proposed v3 prompt textarea (read-only)
          * unified diff (difflib.unified_diff)
          * failure-mode summary (counts per mode in the dev pools)
          * generator examples (collapsible rows): question, history,
            gold_answer, pred_answer, gold_turn_program, pred_program,
            pred_sub_questions, retrieved values, failure_mode, notes
          * evaluator examples (same shape, separate pool)
          * per-loop generator output + evaluator feedback
          * final decision + rationale
      - Holdout panel: v2_v3_comparison.csv and
        v2_v3_failure_mode_delta.csv rendered as tables.
  - The dashboard MUST make it visually obvious that only Preprocess was
    optimised — the other agents render in a "held constant" panel, not
    side-by-side as if they were optimised.

Task 8 — Entry point
CREATE prompt_improve_v2.py:
  - # ruff: noqa: T201 at top
  - load_dotenv(Path.home() / ".env") BEFORE any pydantic_agent import.
  - argparse with subcommands: collect-traces, diagnose, split, optimise,
    score-holdout, dashboard, all.
  - `all` runs them in sequence, using existing artefacts when present.
  - Print progress like api_eval.py (bar + counts).

Task 9 — Tests (deterministic mode)
CREATE tests/test_prompt_improve_v2.py:
  - Import-time test: package imports without LLM key (use monkeypatch on
    DEEPSEEK_API_KEY).
  - Diagnosis tests: hand-craft StageTrace fixtures covering every failure
    mode and assert diagnose() labels them correctly.
  - Eligibility test: assert that for every diagnosed FailureExample,
    `harness_eligible` is True iff
    `failure_mode in PREPROCESS_FAILURE_MODES and gold_turn_type=="program"
    and cascade_of is None`.
  - Splits tests:
      * deterministic — run twice with same seed, assert identical
        splits.json,
      * generator/evaluator pools contain ONLY harness-eligible rows,
        ONLY gold_turn_type=="program", and NO cascade rows,
      * no conversation appears in both generator_dev and evaluator_dev,
      * a number-turn failure inside a chosen dev conversation is logged
        in `excluded_in_dev_conversations` and NOT in `row_assignments`.
  - Apply tests:
      * `write_candidate_v3(patch, ...)` produces JSON with EXACTLY the key
        path `preprocess.predict.signature.instructions` and nothing else,
      * `write_candidate_v3(None, ...)` produces `{}`,
      * `apply_overlay({"preprocess.predict": {...}}, base_json)` replaces
        only the preprocess instructions; triage/retriever/calculator
        instructions are byte-identical to the input.
  - Harness tests:
      * `improve_preprocess` with stubbed generator/evaluator agents that
        accept on first loop returns the patch and writes
        accepted_patches.jsonl,
      * three "revise" decisions terminate after loop 3 with None and
        write rejected_patches.jsonl,
      * passing a generator_examples list containing a non-program / non-
        Preprocess example raises ValueError.
  - DO NOT make HTTP calls or invoke the generator/evaluator agents. Mock
    `generator_agent.run` and `evaluator_agent.run` with stubs that return
    a known PromptPatch / PromptReview.
```

### Per-task pseudocode

```python
# Task 1 — tracing.py
async def trace_turn(question: str, report_id: str, conversation) -> StageTrace:
    # PATTERN: copied from pydantic_agent.run_turn (don't import — re-implement)
    document = _DOCS[report_id]
    hist_text = conversation.as_text()

    triage = (await triage_agent.run(_render_chat_inputs({"question": question}))).output

    sub_questions, pred_program, retrieved = [], "", []
    calc_traj = None

    if triage.turn_type == "number":
        retr = (await retriever_agent.run(_render_chat_inputs({
            "turn_type": "number", "questions": [question],
            "document": document, "history": hist_text,
        }))).output
        retrieved = [qa.model_dump() for qa in retr.answers]
        answer = str(retr.answers[0].answer)
    else:
        pp = (await preprocess_agent.run(_render_chat_inputs({
            "question": question, "history": hist_text,
            "conv_type": triage.conv_type,
        }))).output
        sub_questions, pred_program = list(pp.sub_questions), pp.program

        retr = (await retriever_agent.run(_render_chat_inputs({
            "turn_type": "program", "questions": sub_questions,
            "document": document, "history": hist_text,
        }))).output
        retrieved = [qa.model_dump() for qa in retr.answers]

        calc_result = await calculator_agent.run(_render_chat_inputs({
            "question": question,
            "retrieved": retrieved, "program": pred_program,
        }))
        calc_traj = _coerce_trajectory(calc_result.all_messages())  # mirror stream_turn
        answer = str(calc_result.output.answer)

    conversation.append(question=question, answer=answer, report_id=report_id)
    # gold values come from qa_data — caller fills them after; trace_turn returns
    # the prediction-side fields only
    return StageTrace(... pred fields ...)


# Task 2 — diagnose.py
def diagnose(traces: list[StageTrace]) -> list[FailureExample]:
    # CRITICAL: cascade detection comes BEFORE per-turn diagnosis
    by_conv = group_by(traces, "report_id")
    out = []
    for rid, conv_traces in by_conv.items():
        conv_traces.sort(key=lambda t: t.turn_index)
        first_wrong: int | None = None
        for t in conv_traces:
            if t.correct:
                continue
            if first_wrong is None:
                first_wrong = t.turn_index
                mode, notes = _root_cause(t)
                out.append(_to_failure_example(t, mode, notes, cascade_of=None))
            else:
                # cascade: include for display, exclude from prompt-improvement pools
                out.append(_to_failure_example(t, "cascade", [], cascade_of=first_wrong))
    return out


# Task 4 — harness.improve_preprocess
async def improve_preprocess(current_instructions, gen_egs, eval_egs, out_dir):
    # PRE-CHECK: Preprocess-only, program-only.
    for e in (*gen_egs, *eval_egs):
        if e.failure_mode not in PREPROCESS_FAILURE_MODES:
            raise ValueError(f"Non-Preprocess failure_mode in pool: {e.failure_mode}")
        if e.gold_turn_type.lower() != "program":
            raise ValueError(f"Non-program turn in pool: row_id={e.row_id}")

    request = PromptPatchRequest(
        agent="preprocess",
        current_instructions=current_instructions,
        failure_mode_summary=Counter(e.failure_mode for e in gen_egs),
        generator_examples=gen_egs[:15],
        original_generator_guide=GENERATOR_SYSTEM_PROMPT_PREPROCESS,
        loop_index=1,
    )
    previous = None
    for loop in (1, 2, 3):
        request.loop_index = loop
        patch = (await generator_agent.run(request.model_dump_json())).output
        review_req = PromptReviewRequest(
            agent="preprocess",
            current_instructions=current_instructions,
            generator_instructions=GENERATOR_SYSTEM_PROMPT_PREPROCESS,
            generator_patch=patch,
            evaluator_examples=eval_egs[:15],
            loop_index=loop,
        )
        review = (await evaluator_agent.run(review_req.model_dump_json())).output
        _append_jsonl(out_dir / "category_runs.jsonl", {...})
        _append_jsonl(out_dir / "evaluator_reviews.jsonl", review.model_dump())
        if review.decision == "accept":
            _append_jsonl(out_dir / "accepted_patches.jsonl", patch.model_dump())
            _append_jsonl(out_dir / "prompt_diffs.jsonl", {
                "agent": "preprocess",
                "diff": "".join(difflib.unified_diff(
                    current_instructions.splitlines(keepends=True),
                    patch.revised_instructions.splitlines(keepends=True),
                    fromfile="v2", tofile="v3",
                )),
            })
            return patch
        if review.decision == "reject" or loop == 3:
            _append_jsonl(out_dir / "rejected_patches.jsonl", {
                "agent": "preprocess", "loop": loop, "rationale": review.rationale,
            })
            return None
        # decision == "revise"
        previous = patch
        request.previous_generated_prompt = patch.revised_instructions
        request.evaluator_feedback = "\n".join(review.requested_changes)
    return None
```

### Integration Points

```yaml
ENV:
  - DEEPSEEK_API_KEY: required (already used by the pipeline)
  - PROMPTS_OVERLAY_PATH: NEW — read by pydantic_agent._load_optimized_prompts
    when set; points at prompts_candidate_v3.json
  - GEPA_NAME: existing — defaults to "gepa_real_20260502_005251"

ARTEFACTS:
  - All new artefacts under runs/<GEPA_NAME>/prompt_optim_v2/
    (folder name kept for parity with the EXAMPLE PRP; the prompts inside
    are labelled v3 to follow the user's v2/v3 run-naming convention.)
  - DO NOT touch runs/<GEPA_NAME>/dspy_optimized_runner.json — it is the v2
    baseline and read-only.
  - DO NOT touch dspy_agent.py or any dspy_* file under runs/.

DEPENDENCIES (pyproject.toml):
  - pydantic-ai>=0.2 already present
  - logfire already present (no extra config needed; instrumentation optional)
  - No new deps required.
```

## Validation Loop

### Level 1: Syntax & Style

```bash
uv run ruff check prompt_improve_v2.py prompt_optim/ tests/test_prompt_improve_v2.py --fix
uv run mypy prompt_improve_v2.py prompt_optim/
# Expected: no errors. The codebase has `strict = false` so adopt the same posture.
```

### Level 2: Unit Tests

```bash
uv run pytest tests/test_prompt_improve_v2.py -v
# Must pass without DEEPSEEK_API_KEY by mocking the two pydantic-ai agents.
```

Test cases that MUST exist:

```python
def test_diagnose_triage_turn_type_mismatch_excluded_from_harness():
    """pred=number, gold=program -> triage_turn_type, harness_eligible=False."""

def test_diagnose_formatting_mismatch_takes_precedence_over_program():
    """pred='117.0%' gold='117%' -> formatting_mismatch, not preprocess_wrong_program."""

def test_diagnose_cascade_detection():
    """Conv with turn 0 wrong, turn 1 wrong: turn 1 -> cascade with cascade_of=0
    and harness_eligible=False even if it would otherwise look like Preprocess."""

def test_diagnose_preprocess_wrong_program_via_op_multiset():
    """gold ops {divide,subtract,multiply}, pred ops {divide} ->
    preprocess_wrong_program, harness_eligible=True (gold is program)."""

def test_diagnose_preprocess_unclear_sub_questions():
    """ops match gold, retrieved value is None / empty for a sub-question that
    omits the year -> preprocess_unclear_sub_questions, harness_eligible=True."""

def test_diagnose_retriever_wrong_value_excluded():
    """ops match gold, sub-question is unambiguous (year+entity), retrieved
    value is wrong -> retriever_wrong_value, harness_eligible=False."""

def test_diagnose_calculator_execution_error_excluded():
    """ops match gold, retrieved values look fine, final answer wrong ->
    calculator_execution_error, harness_eligible=False."""

def test_split_is_deterministic():
    """build_splits(...seed=42) twice yields identical assignments."""

def test_split_keeps_conversation_intact():
    """No conversation appears in BOTH generator_dev and evaluator_dev."""

def test_split_only_program_preprocess_rows_in_dev_pools():
    """Every row in generator_dev / evaluator_dev has gold_turn_type=='program'
    and failure_mode in PREPROCESS_FAILURE_MODES."""

def test_split_excludes_number_turn_failures_in_chosen_conversations():
    """A number-turn failure in a chosen conversation is in
    excluded_in_dev_conversations, never in row_assignments."""

def test_apply_writes_single_key_json():
    """write_candidate_v3(patch, ...) JSON has exactly one leaf path
    'preprocess.predict.signature.instructions' and no triage/retriever/
    calculator keys."""

def test_apply_writes_empty_json_on_reject():
    """write_candidate_v3(None, ...) writes '{}'."""

def test_apply_overlay_preserves_other_agents():
    """apply_overlay({'preprocess.predict': {...}}, base) leaves triage,
    retriever, calculator instructions byte-identical to base."""

def test_harness_rejects_non_preprocess_examples():
    """improve_preprocess raises ValueError if any example has
    failure_mode='triage_turn_type' or gold_turn_type='number'."""

def test_harness_accepts_on_first_loop(monkeypatch):
    """Stub generator + evaluator agents; loop returns patch immediately."""

def test_harness_rejects_after_three_loops(monkeypatch):
    """Three 'revise' decisions -> final loop returns None and writes rejected_patches.jsonl."""
```

### Level 3: Integration

Smoke (no LLM):

```bash
# Fresh stage trace collection on a 3-conversation slice:
uv run python prompt_improve_v2.py collect-traces --limit 3
test -s runs/gepa_real_20260502_005251/prompt_optim_v2/stage_traces.jsonl

# Diagnosis + splits without any LLM call:
uv run python prompt_improve_v2.py diagnose
uv run python prompt_improve_v2.py split
test -s runs/gepa_real_20260502_005251/prompt_optim_v2/diagnoses.csv
test -s runs/gepa_real_20260502_005251/prompt_optim_v2/splits.json
```

Full optimise loop (DEEPSEEK_API_KEY required):

```bash
uv run python prompt_improve_v2.py optimise
test -s runs/gepa_real_20260502_005251/prompt_optim_v2/prompts_candidate_v3.json
```

Holdout score (v2 baseline vs v3 prompts via pydantic_agent):

```bash
uv run python prompt_improve_v2.py score-holdout
# Verify v2 vs v3 deltas are printed and v2_v3_comparison.csv exists.

# The same v3 prompts can also be re-scored end-to-end through the standard
# pydantic_agent.py entry point (no dspy involved), which is the canonical
# evaluation surface:
PROMPTS_OVERLAY_PATH=runs/gepa_real_20260502_005251/prompt_optim_v2/prompts_candidate_v3.json \
PYDANTIC_EVAL_FORCE=1 \
uv run python pydantic_agent.py
```

End-to-end:

```bash
uv run python prompt_improve_v2.py all
open runs/gepa_real_20260502_005251/prompt_optim_v2/prompt_review_dashboard.html
```

## Final validation Checklist

- [ ] `uv run pytest tests/test_prompt_improve_v2.py -v` passes
- [ ] `uv run ruff check prompt_improve_v2.py prompt_optim/` clean
- [ ] `uv run mypy prompt_improve_v2.py prompt_optim/` clean
- [ ] `stage_traces.jsonl` length == sum of turns across the test conversations
- [ ] `diagnoses.csv` row count == count of turns where `correct=False`
- [ ] `splits.json` covers exactly 30 unique conversations across `generator_dev` + `evaluator_dev`, no overlap, deterministic
- [ ] No conversation appears in two splits
- [ ] `prompts_candidate_v3.json` validates as JSON and contains EXACTLY the `preprocess.predict.signature.instructions` key path (or is `{}` if rejected). NO `triage.predict`, `retriever.predict`, or `calculator.react` entries — those agents are held constant at v2.
- [ ] Generator/evaluator example pools contain only rows where `gold_turn_type=="program"` and `failure_mode in {preprocess_wrong_program, preprocess_unclear_sub_questions}`. Verified by inspecting `splits.json` and `category_runs.jsonl`.
- [ ] `pydantic_agent` accepts `PROMPTS_OVERLAY_PATH=...` and successfully runs one conversation end-to-end with the overlaid Preprocess prompt (manual smoke)
- [ ] `v2_v3_comparison.csv` shows holdout deltas, with overall v3 accuracy not regressing more than 2 pp vs v2
- [ ] `v2_v3_failure_mode_delta.csv` shows the per-failure-mode change between v2 and v3
- [ ] `prompt_review_dashboard.html` renders correctly in a browser
- [ ] No file under `prompt_optim/`, `prompt_improve_v2.py`, or `tests/test_prompt_improve_v2.py` imports `dspy_agent`

---

## Anti-Patterns to Avoid

- ❌ DO NOT optimise the Triage, Retriever, or Calculator prompts in this PRP. v3 differs from v2 in exactly one place: `preprocess.predict.signature.instructions`. If the dashboard shows a diff for any other agent, the implementation is wrong.
- ❌ DO NOT show the generator or evaluator any failure that is not `(gold_turn_type=="program") AND (failure_mode in PREPROCESS_FAILURE_MODES) AND (cascade_of is None)`. Number-turn failures, formatting mismatches, retriever-side bugs, calculator execution errors, and cascade rows must be filtered out before the harness sees them.
- ❌ DO NOT import or modify `dspy_agent.py` from any new file in this PRP. Pipeline state is owned by `pydantic_agent.py` only; the optimised prompt JSON in `runs/<GEPA_NAME>/dspy_optimized_runner.json` is read-only data.
- ❌ DO NOT mix `generator_dev` and `evaluator_dev` examples — collapses the loop into single-set tuning.
- ❌ DO NOT score v2 on the same conversations the harness consumed — only the untouched holdout counts.
- ❌ DO NOT compare gold vs predicted programs by string equality — use operation multisets.
- ❌ DO NOT diagnose cascade rows as Preprocess failures — they are downstream of the first wrong turn.
- ❌ DO NOT change the wire format (`[[ ## name ## ]]`) — the existing prompts are tuned to it.
- ❌ DO NOT change the pipeline model. The pipeline runs on `deepseek-chat`; the optimiser uses `deepseek-v4-pro`. Holdout re-eval MUST run on `deepseek-chat` to make v2 vs v3 comparable.
- ❌ DO NOT mutate `dspy_optimized_runner.json` in place; only write candidate v3 to a fresh file.
- ❌ DO NOT promote `prompts_candidate_v3.json` to live until a human reviews the dashboard.

---

## Confidence Score

**8.5 / 10** for one-pass implementation success.

Why 8.5 and not higher:
- The diagnosis boundary between `preprocess_unclear_sub_questions` (Preprocess fault — sub-question was vague, Retriever did its best with what it was asked) and `retriever_wrong_value` (Retriever fault — Preprocess gave a clean question, Retriever still pulled the wrong cell) is heuristic. The implementer will likely need to iterate on the sub-question-clarity rule after seeing the first batch of diagnoses on real data. This is the single biggest point of judgment in the PRP.
- The `rebuild_pipeline_agents(overlay_path)` mechanism for the holdout re-eval is the most fragile bit — Pydantic AI Agents bind their `instructions` at construction time. We construct fresh agents in `apply.py` against the overlaid prompts and pass them through `run_turn(..., agents=...)`, which is the cleanest path.
- Everything else (Preprocess-only scope, program-only filter, splits, single harness loop, single-key candidate JSON, dashboard "held constant" panel) is mechanically clear and tightly scoped — narrower than the original PRP and therefore lower-risk to implement.
