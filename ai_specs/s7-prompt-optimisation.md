name: "ConvFinQA Prompt-Improvement Harness — Per-Case Diagnose → Route+Fix → Verify Loop for Sub-Agent System Prompts"

## Purpose

Build an evaluation harness that iteratively improves sub-agent `system_prompt`s based on observed errors. It reads committed predictions (`evaluation/pydantic_predictions_v2.csv`) and, for each conversation's first failed question, runs a **per-case three-step flow** with an optional retry loop:

1. **Diagnose** — Router LLM classifies which sub-agent (Triage / Preprocess / Retriever / Calculator) caused the wrong answer. One `LM_MAX` call. Outputs `RouterDiagnosis`; no fix.
2. **Route + Fix** — The router's `failed_agent` selects one of four specialist Fix LLMs (`TriageFix / PreprocessFix / RetrieverFix / CalculatorFix`). The specialist proposes a single `system_prompt` rule for its own sub-agent only. One `LM_MAX` call.
3. **Verify** — Patch that sub-agent's prompt with the proposed rule and re-run the conversation **from turn 0 through the originally-failed turn `k`** (`report_id`, turns 0..k), feeding predicted answers forward as conversation history. Replaying through `k` is required because the conversation is multi-turn — turns 1..k may reference earlier Q&A, so the history must be rebuilt with the patched agents. The patch passes iff:
   - turn `k` now matches its gold answer, AND
   - no previously-correct prior turn (0..k-1) regresses (each was correct in the original run; it must remain correct).
4. **Retry on fail** — If verify fails AND attempts so far `< retry_n`, go back to step 2 (route+fix) with the failed `FixAttempt` appended to `payload.prior_attempts`. The router's diagnosis from step 1 is reused — diagnose is **not** re-called. The specialist sees its prior `patch_applied` + the replay's first failing turn (with stage IOs) and proposes a refined rule. Stop on a passing verify or once `retry_n` total attempts have run.

**Default `retry_n = 1`** — one diagnose → fix → verify pass per case, **no retries**. Override with `--retry-n N` (CLI) or `RETRY_N=N` (env). Hard ceiling 3 (up to 2 retries).

After every case has run, two finalisation steps execute once:

- **Assemble** — regenerate `src/convfinqa/prompts/v3_opt.py` from v2.py + the four `rules_<agent>_v3_opt.jsonl` stores.
- **Regression eval** — re-score `v3_opt` against the full dev/test set and emit deltas vs v2.

The only allowed output is a `system_prompt` change for a single sub-agent. No code, pipeline, model, or tool changes. Every rule in the final store has been empirically verified against the case that motivated it.

## Per-Case Loop

Cases are processed **sequentially**, one fully resolved (or unresolved) before the next starts. Rules added by earlier cases compound into the live prompt seen by later cases via `_assemble_current_prompts()` before each case. No `asyncio.gather` anywhere across cases.

```
For each first-wrong case (one row per report_id, at turn k = min(turn_index where correct == False)):

  ┌────────────────────────────────────────────────────────────────────┐
  │ Step 1 — DIAGNOSE                                                  │
  │   diagnostic_router_agent(RouterPayload) → RouterDiagnosis         │
  │   1 LM_MAX call. Picks failed_agent ∈                              │
  │     {triage, preprocess, retriever, calculator, ambiguous}.        │
  │   ambiguous → mark unresolved, skip to next case (no fix attempt). │
  └────────────────────────┬───────────────────────────────────────────┘
                           │  (diagnosis cached for this case)
                           ▼
  ┌──────────── attempt loop (1..retry_n) ─────────────────────────────┐
  │                                                                    │
  │ Step 2 — ROUTE + FIX                                               │
  │   FIX_AGENTS[failed_agent](FixPayload) → FixProposal               │
  │   1 LM_MAX call into the specialist for that sub-agent only.       │
  │   FixPayload carries:                                              │
  │     • router_diagnosis (from Step 1, reused on every retry)        │
  │     • current_prompt  — sub-agent's live prompt = v2 + already-    │
  │                         passing rules from rules_<agent>.jsonl     │
  │     • prior_rule_attempts — cross-run history for this sub-agent   │
  │     • prior_attempts  — within-case history (empty on attempt 1,   │
  │                         populated on retries)                      │
  │                                                                    │
  │ Step 3 — VERIFY                                                    │
  │   verify_patch(failed_agent, FixProposal.rule, ...) → FixAttempt   │
  │   No LLM of its own; runs the patched production sub-agents to     │
  │   replay turns 0..k. Pass iff turn k matches gold AND turns 0..k-1 │
  │   still match gold (no regression).                                │
  │                                                                    │
  │   append_attempt(...)  — always (pass or fail).                    │
  │                                                                    │
  │   if pass:                                                         │
  │     append_rule(...)  → live baseline for that sub-agent grows               │
  │     stop attempt loop, mark case resolved.                         │
  │   elif attempts < retry_n:                                         │
  │     payload.prior_attempts.append(FixAttempt)                      │
  │     go to Step 2 (router NOT re-called — diagnosis is cached).     │
  │   else:                                                            │
  │     stop attempt loop, mark case unresolved.                       │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
```

After the case loop completes for all cases:

- **Assemble** — `assemble_v3_opt()` reads the four `rules_<agent>_v3_opt.jsonl` stores and writes `src/convfinqa/prompts/v3_opt.py`.
- **Regression** — subprocess-invokes `convfinqa-eval-api --version v3_opt` and writes deltas vs v2.

The four `evaluation/rules_<agent>_v3_opt.jsonl` stores are the **source of truth**. `v3_opt.py` is a regenerable cache assembled from them. Per-agent isolation lets each sub-agent's rules evolve independently and makes rollback trivial (`jq 'select(.rule_id != …)'`).

## Feature Architecture

**Five LLM-backed agents total** (all using `LM_MAX = OpenAIChatModel(settings.lm_max_model)`):

| Agent | Role | Input | Output |
|---|---|---|---|
| `diagnostic_router_agent` | Classify-only — pick which subagent failed and explain why. NEVER proposes a fix. Called once per case (Step 1). | `RouterPayload` (failing case IO, gold/pred, current four v2 prompts, history) | `RouterDiagnosis` (failed_agent, failure_mode, failure_explanation, supporting_evidence, confidence) |
| `triage_fix_agent` | Specialist — write rules for the Triage subagent only. Knows triage's job (turn_type + conv_type classification) deeply. Called once per attempt (Step 2) when `failed_agent == "triage"`. | `FixPayload` (router's `RouterDiagnosis`, Triage's current live prompt, prior `rule_attempts` for triage, within-case `prior_attempts`, failing case IO) | `FixProposal` (rule, fix_type, confidence, rationale) |
| `preprocess_fix_agent` | Specialist — write rules for Preprocess only. Knows the DSL (`add/subtract/multiply/divide/exp/greater`), sub-question specification rules, percentage convention. | same shape, but receives Preprocess's current prompt + prior preprocess attempts | `FixProposal` |
| `retriever_fix_agent` | Specialist — write rules for Retriever only. Knows table/text cell-lookup discipline and direct-answer (number-turn) behaviour. | same shape, but receives Retriever's current prompt + prior retriever attempts | `FixProposal` |
| `calculator_fix_agent` | Specialist — write rules for Calculator only. Knows tool ordering, operand assignment, program override rules. | same shape, but receives Calculator's current prompt + prior calculator attempts | `FixProposal` |

### Data flow per case

```
   failing case from CSV
            │
            ▼
   ┌─────────────────────────────────────┐
   │ Step 1 — DIAGNOSE                   │
   │ diagnostic_router_agent (1 LM call) │
   │ outputs: RouterDiagnosis            │
   └────────────────┬────────────────────┘
                    │ failed_agent ∈
                    │ {triage,preprocess,retriever,calculator,ambiguous}
                    │ (ambiguous → unresolved, no fix attempt)
                    ▼
   ╔═══════════════════════════════════════════════════════════╗
   ║  attempt loop (1..retry_n) — router NOT re-called inside  ║
   ║                                                           ║
   ║   ┌────────────────────────────────────────────────────┐  ║
   ║   │ Step 2 — ROUTE + FIX                               │  ║
   ║   │ FIX_AGENTS[failed_agent] (1 LM call)               │  ║
   ║   │ inputs: RouterDiagnosis, current_prompt for THIS   │  ║
   ║   │   sub-agent, prior cross-run rule_attempts for     │  ║
   ║   │   THIS sub-agent, within-case prior_attempts       │  ║
   ║   │ outputs: FixProposal (rule, fix_type, confidence,  │  ║
   ║   │   rationale)                                       │  ║
   ║   └────────────────┬───────────────────────────────────┘  ║
   ║                    ▼                                      ║
   ║   ┌────────────────────────────────────────────────────┐  ║
   ║   │ Step 3 — VERIFY (no LLM of its own)                │  ║
   ║   │ build_patched_prompt → make_agents → replay 0..k   │  ║
   ║   │ pass iff turn k matches gold AND no regression in  │  ║
   ║   │   turns 0..k-1                                     │  ║
   ║   └────────────────┬───────────────────────────────────┘  ║
   ║                    │                                      ║
   ║   append_attempt(...) always                              ║
   ║                    │                                      ║
   ║   ┌────────────────┴─────────────────┐                    ║
   ║   ▼                                  ▼                    ║
   ║  pass → append_rule(...) → STOP    fail                   ║
   ║                                      │                    ║
   ║                                      ▼                    ║
   ║                              attempts < retry_n?          ║
   ║                              yes → loop back to Step 2    ║
   ║                              no  → STOP (unresolved)      ║
   ╚═══════════════════════════════════════════════════════════╝
```

### Why the split (router + 4 specialists)

- **Specialization**: each fix agent's prompt is tuned to the failure modes that subagent actually exhibits. The TriageFix agent never needs to think about DSL operators; the PreprocessFix agent never needs to think about `turn_type` classification. Smaller, focused prompts → fewer false positives.
- **Cross-attempt awareness scoped per agent**: the prior `rule_attempts` injected into a specialist are *only that subagent's history*. Other subagents' attempts are irrelevant noise.
- **Independent prompt evolution**: each specialist's `FIX_<AGENT>_SYSTEM_PROMPT` can be iterated on without touching the others or the router.
- **Cheap router**: the router only classifies, doesn't propose fixes — short response, cheap. Reused across all retries within a case so we never pay for re-diagnosis.

### What does NOT change

- `verify_patch` is still pure code (no LLM of its own — it does invoke the patched production sub-agents).
- The four sub-agent run-time prompts (Triage/Preprocess/Retriever/Calculator in `prompts/v2.py`) — these are the *targets* of the fixes, not part of the diagnostic stack.
- The append-rule / append-attempt store logic.
- The post-loop assemble + regression steps.

### Cost & concurrency

| Step | LLM | Calls per case | Concurrency |
|---|---|---|---|
| Step 1 — Diagnose | `diagnostic_router_agent` | 1 | Sequential across cases |
| Step 2 — Route + Fix | One of four `FIX_AGENTS[failed_agent]` | 1 per attempt (1..retry_n) | Sequential (only one specialist per case; rules compound case-to-case) |
| Step 3 — Verify | None directly — replays patched production pipeline | `k+1` turn-runs per attempt | Sequential within the replay |
| Assemble | None (pure code) | 0 | n/a |
| Regression | Subprocess invokes production eval | n/a | Inherits production eval concurrency |

With default `retry_n = 1`, total cost per case = 1 router call + 1 specialist call + `k+1` turn-runs. With `retry_n = 3` the upper bound is 1 router call + 3 specialist calls + `3(k+1)` turn-runs.

## Goal

Implement `scripts/diagnose_failures.py` + `src/convfinqa/diagnosis/` with:

- Loads predictions CSV, filters to **first wrong turn per `report_id`** (`min(turn_index)` where `correct == False`) to isolate root causes and skip cascade-poisoned downstream turns.
- All config via `convfinqa.config.settings` — never `load_dotenv`. New fields: `lm_max_model: str = "deepseek-reasoner"`, `rules_dir: Path = "evaluation"`, `retry_n: int = 1` (1 ≤ N ≤ 3, where N is the total attempts cap — N=1 ⇒ no retries).
- CLI modes:
  - `--diagnose-only`: run Step 1 (diagnose) only for every case; no fix, no verify. Used to inspect router output without spending fix-or-replay budget.
  - `--stage {assemble,regression}`: standalone post-loop stages (no harness invocation).
  - default: run the full per-case loop end-to-end → assemble → regression.
  - `--limit N`, `--reset-rules`, `--force`, `--skip-regression`, `--version v2`.
- Each case runs up to `settings.retry_n` total attempts (default 1 = no retries). The router is called once per case; only Step 2 (specialist fix) is repeated on retry. Iterations 2/3 see prior `FixAttempt`s in the payload; with default 1 the retry loop is effectively disabled — every case gets exactly one diagnose + one fix + one verify (unless ambiguous/duplicate-patch terminates earlier).
- Log every Step 2/3 line prefixed `[<agent>]` so the log clearly indicates which sub-agent's rules are being modified.

## Outputs

All artefacts use the `_v3_opt` suffix to sit alongside v1/v2 outputs.

| File | Description |
|---|---|
| `evaluation/diagnostic_results_v3_opt.csv` | One row per `(report_id, turn_index, attempt_id)`. |
| `evaluation/diagnostic_results_v3_opt.html` | HTML clone of `pydantic_predictions_v2.html` — same dark theme, sortable headers, collapsible JSON cells. |
| `evaluation/case_results_v3_opt.jsonl` | Structured `CaseResult` backup; one row per case, captures router diagnosis + every attempt's `FixAttempt`. |
| `evaluation/rules_<agent>_v3_opt.jsonl` (×4) | **Source of truth for `v3_opt.py`** — passes only. One line per verified rule: `rule_id`, `agent`, `rule`, `fix_type`, `confidence`, `verified_on`, `verified_at`, `supersedes`. |
| `evaluation/rule_attempts_<agent>_v3_opt.jsonl` (×4) | **Full attempt history — passes AND failures.** One line per verify call (across all cases and runs). Read by the specialist fix agent on subsequent runs so it doesn't re-propose rules already known to pass or fail. Never read by the assembler (does not affect `v3_opt.py`). |
| `evaluation/unresolved_cases_v3_opt.json` | Cases that exhausted `retry_n` without a passing verify, plus router-`ambiguous` cases. |
| `evaluation/regression_v3_opt.csv` | Post-loop regression: per-case delta v2 → v3_opt. |
| `evaluation/model_accuracy_comparison_v3_opt.csv` | Post-loop regression summary row. |
| `src/convfinqa/prompts/v3_opt.py` | **Regenerable cache** — assembled post-loop. Never hand-edited. Loaded via `convfinqa.prompts.load("v3_opt")`. |

## Diagnose Cache (reuse Step 1 across runs)

The router LLM call is the single most expensive part of Step 1 for high-confidence cases (one full `LM_MAX` reasoning call). Re-running it every time the operator switches modes (`--diagnose-only` → full, or full → full after editing prompts) is wasteful when the case's failing inputs haven't changed. The harness therefore reuses prior router diagnoses when available.

**Cache store**: `evaluation/case_results_v3_opt.jsonl` — the same file the harness already writes incrementally per case (one `CaseResult` per line). Each line carries `router_diagnosis`. No new file is introduced.

**Cache key**: `(report_id, turn_index)`. This is the case identity; the loader filter (first-wrong-per-report_id) already keys by it. The cache is **not** invalidated by changes to `pred_program`, `pred_answer`, or the four v2 prompts — operators wipe with `--no-diagnose-cache` or `--reset-rules` (which clears all stores) when they want fresh diagnoses.

**Read path** (at the start of `run_harness`, before the case loop):
1. If `case_results_v3_opt.jsonl` exists AND `--no-diagnose-cache` was NOT passed: stream-parse each line into a `CaseResult`, build `cache: dict[(report_id, turn_index), RouterDiagnosis]` from any entry whose `router_diagnosis` is non-null.
2. The file is then truncated and rewritten fresh by the run — the cache is held in memory for the duration of the run.

**Per-case behaviour** (in `run_case`):
- Look up `cache[(report_id, turn_index)]`. On hit: skip the `route_case()` LLM call, attach the cached `RouterDiagnosis` to the case, log `[<agent>] diagnosis: cached (mode=… conf=…)`. On miss: run Step 1 as normal.
- The rest of the per-case flow (Step 2 + 3 attempt loop, or `--diagnose-only` placeholder) is unchanged. **Cache hits apply equally to `--diagnose-only` and full-mode runs** — so a `--diagnose-only` pass populates the cache, and a subsequent full run reuses every diagnosis from it.

**Promotion path — `--diagnose-only` → full**:
1. Operator runs `uv run python scripts/diagnose_failures.py --diagnose-only` (no fix/verify, but writes `case_results_v3_opt.jsonl` with router diagnoses).
2. Operator inspects diagnoses (HTML viewer or JSONL grep).
3. Operator runs `uv run python scripts/diagnose_failures.py` (full mode). Every case the router already diagnosed loads from cache — zero router LLM cost, only Step 2+3 cost is paid.

**Verify cache**: not introduced in this iteration. Verify is deterministic given (failed_agent, rule, current_prompts), but `current_prompts` evolves intra-run as earlier cases promote rules, so a naive verify cache would serve stale answers. Stays uncached.

**Cache invalidation**:
- `--no-diagnose-cache` (new flag): the harness ignores any existing cache file. The case_results_v3_opt.jsonl is still overwritten on this run's output.
- `--reset-rules`: clears rules + rule_attempts but does NOT delete `case_results_v3_opt.jsonl`. To wipe diagnoses, combine with `--no-diagnose-cache` (or `rm evaluation/case_results_v3_opt.jsonl` manually).
- Editing the router's system prompt: cache is NOT auto-invalidated. Operator must pass `--no-diagnose-cache` for the first run after a router-prompt change.

**Anti-pattern**: do NOT key the cache on prompt or prediction hashes. The user-chosen contract is "case identity only" — invalidation is operator-driven, not content-driven. A content-keyed cache would over-invalidate (any v2 prompt tweak that doesn't affect the failure pattern would still pay the router cost) and under-document (operators stop noticing when the cache silently misses).

## Statefulness Across Runs

1. **First run / `--reset-rules`**: all JSONL stores empty → assemble writes `v3_opt.py` byte-identical to `v2.py`. `--reset-rules` truncates BOTH `rules_<agent>_v3_opt.jsonl` AND `rule_attempts_<agent>_v3_opt.jsonl` — there is no separate `--keep-attempts` toggle. Single switch, single behaviour: pass `--reset-rules` to wipe everything; omit it to keep both stores intact (the default). Rationale: keeping attempts without rules creates a confusing half-state where the specialist agent sees prior `promoted_rule_id` references that no longer exist in the rules store. The two stores are conceptually coupled; reset them together or not at all.
2. **During the case loop**: every verify call appends one line to `rule_attempts_<failed_agent>_v3_opt.jsonl` with `verify_result ∈ {"passed","failed"}`. On `passed`, the harness ALSO appends one line to `rules_<failed_agent>_v3_opt.jsonl`; the in-memory live prompt for that sub-agent is reassembled so subsequent cases see the patched baseline. On `failed`, only the attempts log is updated.
3. **Specialist awareness**: each specialist Fix call is given the prior attempts for its own sub-agent, so it can avoid re-suggesting any rule already known to pass (no-op duplicate) or already known to fail (waste of a verify cycle). This is a prompt-level soft check; a hard byte-equality guard on prior `patch_applied` strings also terminates the attempt loop if the agent ignores the guidance.
4. **End of case loop**: the assembler reassembles `v3_opt.py` from the union of `rules_<agent>_v3_opt.jsonl` stores only. The attempts log is never consulted by the assembler.
5. **Subsequent runs**: both stores reused as-is; new attempts and new passes accumulate. The attempts log grows monotonically and gets the specialist agents "smarter" across runs.
6. **Targeted rollback**: delete lines from `rules_<agent>_v3_opt.jsonl`, re-run `--stage assemble`. The corresponding attempt line in `rule_attempts_<agent>_v3_opt.jsonl` stays — a previously-passing rule that was rolled back is now visible to the specialist agent as a known pass it can re-introduce, or override via `supersedes`.

## Diagnostic Results Schema

CSV columns (in order):

**Group A — preserved from `pydantic_predictions_v2.csv`**: `report_id, turn_index, question, gold_answer, pred_answer, correct, pred_program, gold_program, pred_turn_type, gold_turn_type, pred_conv_type, gold_conv_type, pred_sub_questions, history_text, triage_io, preprocess_io, retriever_io, calculator_io`

**Group B — diagnostic agent output (varies per `attempt_id`)**: `attempt_id, failed_agent, failure_mode, failure_explanation, supporting_evidence, system_prompt_fix, fix_type, confidence`

**Group C — harness verify result (empty in `--diagnose-only`)**: `harness_correct, harness_first_failing_turn, harness_turn_results, harness_pred_answer, harness_triage_io, harness_preprocess_io, harness_retriever_io, harness_calculator_io`

- `harness_correct`: bool, True iff the originally-failed turn `k` is now correct AND no previously-correct turn in 0..k-1 regressed.
- `harness_first_failing_turn`: int or empty; the first turn index in 0..k whose pred ≠ gold. `== k` means the patch didn't fix it; `< k` means the patch caused a regression on a previously-correct turn.
- `harness_turn_results`: JSON-encoded list of `{turn_index, question, gold_answer, pred_answer, correct}` for every turn 0..k actually executed (replay stops early on the first failing turn).
- `harness_pred_answer`: convenience field — the pred_answer of `harness_first_failing_turn`, or of turn `k` if the replay was fully correct.
- `harness_*_io`: stage IOs from the first failing turn (or turn `k` when replay is fully correct).

Row identity: `(report_id, turn_index, attempt_id)` is unique. In `--diagnose-only`, Group C cells are emitted as empty strings (the HTML renders them as `—`).

## Hard Constraint: Only `system_prompt` Changes

Every fix MUST be a prompt addition to the failed agent's `system_prompt`. Forbidden: code (`def`, `import`, `Agent(`, class defs), pipeline-structure changes, model/sampling swaps, tool changes, other agents' prompts, gold/metric/dataset edits. Enforced by prompt rules AND by a mechanical token check on the output. Fixes are appended under a `## Additional Rules (automated patch)` header.

## The Four Sub-Agents (for attribution)

**1. Triage** (every turn): Classifies `turn_type ∈ {number, program}` and `conv_type ∈ {Type I, Type II}`. Does NOT retrieve, compute, or program. If wrong → downstream pipeline shape is wrong.

**2. Preprocess** (program turns only): Decomposes question into `sub_questions` (atomic value lookups, fully specified with year+entity+metric) and `program` (DSL over A,B,C… mapping to sub_questions positionally). DSL ops: `add, subtract, multiply, divide, exp, greater`. Percentage answers require `multiply(..., 100)` outermost. If wrong → bad program or vague sub-questions.

**3. Retriever** (every turn): For program turns, looks up each `sub_question` and returns raw values only. For number turns (Preprocess skipped), produces the final answer directly. Does NOT compute or invent sub-questions. If wrong → well-specified sub-question returned wrong cell, OR wrong direct lookup.

**4. Calculator** (program turns only): Executes the program via tool calls (`add, subtract, multiply, divide, exp, greater, finish`) over retrieved values (first = A, second = B, …). MAY override the program if it contradicts the question (e.g., add the missing `multiply(...,100)`). If wrong → wrong tool order, swapped operand assignment, spurious/missing `*100`, execution error.

## Investigation Protocol (used inside the diagnostic prompt)

The diagnostic agent stops at the first stage where a clear fault is found. Two complementary walks are embedded in `DIAGNOSTIC_ROUTER_SYSTEM_PROMPT`:

### Forward walk (cheap fault detection, run first)

1. **Read gold signals**: `gold_turn_type` (lowercased), `gold_program`, `gold_answer`. Extract gold op multiset via `r'\b(add|subtract|multiply|divide|exp|greater)\b'`.
2. **Triage**: `pred_turn_type != gold_turn_type` → `failed_agent="triage"`, `failure_mode="wrong_turn_type"`. Stop.
3. **Preprocess** (if `gold_turn_type=="program"`): op multisets differ → `preprocess` with `missing_multiply_100 | wrong_subtract_direction | extra_or_missing_op | wrong_op`. Multisets match but sub-questions vague (missing year/entity/metric) → `vague_sub_questions`.
4. **Retriever**: well-specified sub-question returned wrong/empty value → `wrong_retrieved_value`. Number-turn miss → `wrong_direct_lookup`.
5. **Calculator** (only if upstream looks correct): inspect `calculator_io.trajectory` for `wrong_tool_order | spurious_multiply_100 | missing_multiply_100_in_calc | wrong_operand_assignment | execution_error`.
6. **Ambiguous**: no single clear fault → `failed_agent="ambiguous"`, confidence 0.3–0.5.

### Backward walk (root-cause attribution when forward walk is inconclusive)

For `gold_turn_type=="program"` cases the production data flow is `Triage → Preprocess → Retriever → Calculator → pred_answer`. When two stages both look "off" (e.g., Retriever returned the wrong cell AND Preprocess wrote a vague sub-question), the forward walk's first-fault-wins rule can mis-attribute. Re-walk in reverse to pin the **root cause**:

1. **Start at the output**. `pred_answer` ≠ `gold_answer` is the symptom — established by the loader filter.
2. **Calculator**: did it execute `pred_program` faithfully over `retriever_io.output.values`? If yes, blame is upstream (move on). If no — wrong tool order, wrong operand assignment, spurious/missing `*100` *inside* the trajectory — blame Calculator.
3. **Retriever**: were the returned values correct lookups for the `sub_questions` actually given to it? Two sub-cases:
   - Sub-question was **well-specified** (year + entity + metric all present) but value is wrong → Retriever's fault. It had everything it needed and still missed.
   - Sub-question was **vague** (e.g., missing year) → Retriever was set up to fail. The root cause is upstream — Preprocess should have specified more. Do NOT blame Retriever for guessing wrong when the spec was ambiguous.
4. **Preprocess**: does the program (op multiset + structure) match what gold demands? Are sub-questions well-specified? If either is wrong, Preprocess is the root cause even if downstream errors were the proximate symptom.
5. **Triage**: did it correctly classify `turn_type`? If wrong, the whole downstream chain ran in the wrong mode — Triage is the root cause regardless of what looks broken downstream.

**Rule of thumb for the router**: when forward and backward walks agree, report that agent. When they disagree (forward fingers stage N, backward fingers stage N−1), prefer the backward walk — the proximate symptom is downstream of the root cause. The Preprocess/Retriever boundary (below) is the most common place this disagreement shows up: Retriever returns a wrong value (forward fault), but the sub-question it was given was under-specified (backward root cause = Preprocess).

### Preprocess/Retriever boundary (the hardest judgment)
- Sub-question missing year/entity/metric specifier → Preprocess.
- Sub-question clearly specifies year+entity+metric but value is wrong/empty → Retriever.
- Partially specified → ambiguous.

### Retry mode (iterations 2 and 3) — only active when `retry_n > 1`

With the default `retry_n = 1`, retry mode is dormant — the rest of this section only applies when the operator opts into multi-iteration refinement.

When `prior_attempts` is non-empty, the agent must read each prior `patch_applied` + the replay's **first failing turn in 0..k** (its question, IOs, pred vs gold) and identify why the patched replay still failed. Two failure shapes (where `k` is the originally-failed turn index):

- `first_failing_turn == k`: the patch didn't fix the originally-failed turn. Likely too narrow / vague phrasing / wrong agent attributed. Propose a refined fix (broader trigger, worked example) or re-attribute to a different agent.
- `first_failing_turn < k` (regression): the patch broke a previously-correct earlier turn. The patch trigger is too aggressive — propose a refinement that narrows it (add a guard condition referencing the failing turn's distinguishing feature) so it doesn't fire on those earlier turns.

Options: refine the same fix (broaden/narrow trigger, add example), re-attribute to a different agent, or give up with `failed_agent="ambiguous"`. NEVER repeat a prior `patch_applied` verbatim — the harness's duplicate-fix guard treats repeats as terminal.

## Rules for `system_prompt_fix`

Allowed: a new rule, modification, or example for the failed agent's prompt. Forbidden: anything non-prompt (see Hard Constraint). Must be specific, target only the failed agent, minimise regression risk by adding conditions rather than overriding defaults, quote the failure pattern, and be one fix per diagnosis. Copy-pasteable into `src/convfinqa/prompts/v2.py`.

### Worked example — Preprocess `missing_multiply_100`

Question: "what was the net change over the 2005 value, in percentage?"
`gold_program`: `multiply(divide(subtract(B,A),A),100)`; `pred_program`: `divide(subtract(B,A),A)`; `gold_answer=37.5`, `pred_answer=0.375`.

Diagnosis: `failed_agent="preprocess"`, `failure_mode="missing_multiply_100"`, evidence cites the program diff + the "in percentage" phrase + `pred=gold/100`.
Fix (`add_rule`, confidence 0.95): "If the question contains any of 'in percentage', 'percent change', 'percentage change', 'growth rate', or 'as a percentage', the program MUST include `multiply(..., 100)` as the outermost operation."

On retry (if the patch matched only standalone 'percentage' and the question used "in percentage"): broaden the trigger to phrase variants and add a worked example, same `failed_agent`, `fix_type="modify_rule"`.

## Data Models (in `diagnosis/models.py`)

```python
FailedAgent = Literal["triage", "preprocess", "retriever", "calculator", "ambiguous"]
FixType = Literal["add_rule", "modify_rule", "add_example", "clarify_instruction"]

class StageIO(BaseModel):
    input: dict[str, Any] = {}
    output: dict[str, Any] = {}
    trajectory: list[dict[str, Any]] = []

class TurnResult(BaseModel):
    turn_index: int
    question: str
    gold_answer: str
    pred_answer: str
    correct: bool

class FixAttempt(BaseModel):
    iteration: int                       # 1..settings.retry_n (default 1; max 3)
    failed_agent: str
    patch_applied: str
    full_prompt: str
    # Replay results — turns 0..k where k = originally-failed turn_index:
    turn_results: list[TurnResult]       # one per turn 0..k actually executed
                                         # (stops early on first failing turn)
    correct: bool                        # turn k passed AND no regression on 0..k-1
    first_failing_turn: int | None       # None iff correct == True;
                                         # == k means "patch didn't fix it";
                                         # < k means "patch caused a regression"
    # Stage IOs from the first failing turn (or turn k if everything passed).
    triage_io: StageIO | None
    preprocess_io: StageIO | None
    retriever_io: StageIO | None
    calculator_io: StageIO | None
    # Convenience accessor for back-compat with diagnose-only placeholders:
    @property
    def pred_answer(self) -> str:
        if not self.turn_results: return ""
        idx = self.first_failing_turn if self.first_failing_turn is not None else self.turn_results[-1].turn_index
        return next((t.pred_answer for t in self.turn_results if t.turn_index == idx), "")

class RouterPayload(BaseModel):                     # input to diagnostic_router_agent (Step 1 — Diagnose)
    report_id: str; turn_index: int
    question: str; history_text: str
    gold_answer: str; pred_answer: str
    gold_program: str
    gold_turn_type: str; pred_turn_type: str       # lowercased
    gold_conv_type: str; pred_conv_type: str
    triage_io: StageIO                              # from ORIGINAL failing run
    preprocess_io: StageIO | None
    retriever_io: StageIO
    calculator_io: StageIO | None
    current_triage_prompt: str                      # from prompts/v2.py — context only
    current_preprocess_prompt: str
    current_retriever_prompt: str
    current_calculator_prompt: str
    # NOTE: router does NOT receive prior_rule_attempts — it only classifies.

class FixPayload(BaseModel):                        # input to one of the 4 specialist fix agents (Step 2 — Route+Fix)
    # Identifying fields:
    report_id: str; turn_index: int
    question: str; history_text: str
    gold_answer: str; pred_answer: str
    gold_program: str
    # The router's diagnosis (read-only handoff from Step 1, cached across retries):
    router_diagnosis: RouterDiagnosis
    # ONLY the stage IO for the failed agent + immediate upstream context:
    failed_agent_io: StageIO                        # the IO of the agent we're fixing
    upstream_ios: dict[str, StageIO]                # IOs of agents that ran before it (for context)
    # The current v2 prompt for THIS specialist's agent only:
    current_prompt: str                             # e.g. PREPROCESS_SYSTEM_PROMPT from v2
    # Prior attempts FOR THIS AGENT only (passes + failures), most-recent-N:
    prior_rule_attempts: list[RuleAttempt]          # capped to settings.max_prior_attempts_in_payload
    # Within-case retry history (only populated when retry_n > 1):
    prior_attempts: list[FixAttempt] = []

class RouterDiagnosis(BaseModel):                   # output of diagnostic_router_agent (Step 1 — Diagnose)
    failed_agent: FailedAgent                       # routes to one of 4 specialists; "ambiguous" → unresolved
    failure_mode: str                               # quick-reference tag, e.g. "missing_multiply_100"
    failure_explanation: str                        # 2–4 sentences
    supporting_evidence: list[str]                  # 2–5 quoted IO snippets
    confidence: float = Field(ge=0.0, le=1.0)
    # NOTE: no `system_prompt_fix` — router classifies, specialist proposes.

class FixProposal(BaseModel):                       # output of one of the 4 specialist fix agents (Step 2 — Route+Fix)
    rule: str                                       # the new system_prompt addition (was `system_prompt_fix`)
    fix_type: FixType
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str                                  # 1–3 sentences linking the rule to the router's diagnosis
                                                    # and (if applicable) to the prior_rule_attempts the specialist consulted.

# Convenience alias for backwards-compatible Group B CSV columns:
StageDiagnosis = RouterDiagnosis                    # keep the type name in callsites that just need the routing info

class CaseResult(BaseModel):
    report_id: str; turn_index: int
    question: str; gold_answer: str
    original_pred_answer: str
    gold_turn_type: str; gold_program: str
    attempts: list[FixAttempt]                      # 1..retry_n (full); 1 placeholder (diagnose-only)
    diagnoses: list[StageDiagnosis]                 # same length as attempts
    resolved: bool
    winning_iteration: int | None
    final_patch: str | None

class Rule(BaseModel):                              # one line of rules_<agent>_v3_opt.jsonl
    rule_id: str                                    # "prep-20260519-103300-a1b2c3"
    agent: Literal["triage", "preprocess", "retriever", "calculator"]
    rule: str
    fix_type: FixType
    confidence: float
    verified_on: list[dict[str, Any]]               # [{"report_id":..., "turn_index":...}]
    verified_at: str                                # ISO-8601 UTC
    supersedes: list[str] = []                      # prior rule_ids to filter out

class RuleAttempt(BaseModel):                       # one line of rule_attempts_<agent>_v3_opt.jsonl
    attempt_id: str                                 # "prep-att-20260519-103300-a1b2c3"
    agent: Literal["triage", "preprocess", "retriever", "calculator"]
    rule: str                                       # the proposed patch_applied text
    fix_type: FixType
    confidence: float                               # diagnostic agent's confidence at proposal time
    verify_result: Literal["passed", "failed"]
    attempted_on: dict[str, Any]                    # {"report_id":..., "turn_index": k}
    attempted_at: str                               # ISO-8601 UTC
    # On "failed", capture why so the next diagnose call can avoid the same shape:
    first_failing_turn: int | None                  # only set when verify_result == "failed"
    failure_reason: Literal["did_not_fix", "caused_regression", "duplicate_patch", "ambiguous_followup"] | None
    # If this attempt was later promoted to rules_<agent>_v3_opt.jsonl, the rule_id pointer:
    promoted_rule_id: str | None = None             # set only on verify_result == "passed"
```

## File Layout

Legend: **[NEW]** = added by this feature · **[MODIFIED]** = small additions to an existing file · **[GENERATED]** = written by the harness, never hand-edited · **[READ-ONLY]** = consumed but not changed.

### Code (under `src/convfinqa/`, `scripts/`, `tests/`)

```
src/convfinqa/
  config.py                                  [MODIFIED]   add settings: lm_max_model, rules_dir,
                                                          retry_n, max_prior_attempts_in_payload
  backends/pydantic.py                       [MODIFIED]   add LM_MAX = OpenAIChatModel(settings.lm_max_model)
  evaluation/reporting.py                    [MODIFIED]   extract PREDICTIONS_CSS, PREDICTIONS_JS,
                                                          render_filter_bar to module level (no behaviour change)
  prompts/
    __init__.py                              [READ-ONLY]  already supports load("v3_opt")
    v2.py                                    [READ-ONLY]  base for v3_opt assembly
    v3_opt.py                                [GENERATED]  assembled by diagnosis.assembler after the case loop
  diagnosis/                                 [NEW]        entire package
    __init__.py                              [NEW]
    loader.py                                [NEW]        first-wrong-per-report_id filter + RouterPayload build
    models.py                                [NEW]        RouterPayload, FixPayload, RouterDiagnosis, FixProposal,
                                                          FixAttempt, TurnResult, RuleAttempt, Rule, CaseResult, StageIO
    prompts.py                               [NEW]        DIAGNOSTIC_ROUTER_SYSTEM_PROMPT + 4 specialist prompts
                                                          (FIX_TRIAGE / FIX_PREPROCESS / FIX_RETRIEVER / FIX_CALCULATOR)
    agents.py                                [NEW]        5 LM_MAX agents (1 router + 4 specialists) +
                                                          route_case() + propose_fix() dispatcher
    verify.py                                [NEW]        patch + replay turns 0..k (no LLM)
    rules_store.py                           [NEW]        per-agent JSONL CRUD: rules + rule_attempts
    assembler.py                             [NEW]        assemble_v3_opt() — writes prompts/v3_opt.py
    harness.py                               [NEW]        per-case loop driver: Diagnose → (Route+Fix → Verify) × retry_n
    results_writer.py                        [NEW]        diagnostic_results_v3_opt.csv
    results_html.py                          [NEW]        diagnostic_results_v3_opt.html (imports CSS/JS from reporting.py)
    aggregator.py                            [NEW]        unresolved_cases_v3_opt.json
    regression.py                            [NEW]        post-loop — re-score v3_opt vs v2

scripts/
  diagnose_failures.py                       [NEW]        CLI entry point (--phase, --diagnose-only,
                                                          --reset-rules, --retry-n, etc.)

tests/
  test_diagnose_failures.py                  [NEW]        loader / verify / harness / rules-store / attempts-store /
                                                          router-separation / specialist-routing / assembler /
                                                          regression / entry-point / settings — all mocked, no API key
```

### Data artefacts (under `evaluation/`)

All produced by the harness. Naming convention `_v3_opt` everywhere.

```
evaluation/
  pydantic_predictions_v2.csv                [READ-ONLY]  existing input — failing cases come from here
  pydantic_predictions_v3_opt.csv            [GENERATED]  produced by post-loop regression subprocess (eval-api --version v3_opt)

  # Per-case loop artefacts (gitignored):
  case_results_v3_opt.jsonl                  [GENERATED]  one CaseResult per case (router diagnosis + all attempts); written incrementally for resumability
  diagnostic_results_v3_opt.csv              [GENERATED]  one row per (report_id, turn_index, attempt_id)
  diagnostic_results_v3_opt.html             [GENERATED]  dark-theme HTML view of the CSV
  unresolved_cases_v3_opt.json               [GENERATED]  ambiguous + after-3-attempts failures

  # Source-of-truth stores (COMMITTED):
  rules_triage_v3_opt.jsonl                  [NEW]        verified passes for triage         ── source for v3_opt.py
  rules_preprocess_v3_opt.jsonl              [NEW]        verified passes for preprocess     ── source for v3_opt.py
  rules_retriever_v3_opt.jsonl               [NEW]        verified passes for retriever      ── source for v3_opt.py
  rules_calculator_v3_opt.jsonl              [NEW]        verified passes for calculator     ── source for v3_opt.py

  # Attempt history (COMMITTED — makes future runs smarter):
  rule_attempts_triage_v3_opt.jsonl          [NEW]        every triage attempt (pass + fail)
  rule_attempts_preprocess_v3_opt.jsonl      [NEW]        every preprocess attempt (pass + fail)
  rule_attempts_retriever_v3_opt.jsonl       [NEW]        every retriever attempt (pass + fail)
  rule_attempts_calculator_v3_opt.jsonl      [NEW]        every calculator attempt (pass + fail)

  # Post-loop regression (gitignored):
  regression_v3_opt.csv                      [GENERATED]  per-case delta v2 → v3_opt
  model_accuracy_comparison_v3_opt.csv       [GENERATED]  one-row summary (fixed, regressed, net_delta, accuracies)
```

### What this feature adds — at a glance

| Category | Count | Where |
|---|---:|---|
| New Python modules | 12 | `src/convfinqa/diagnosis/*` (11) + `scripts/diagnose_failures.py` |
| New test files | 1 | `tests/test_diagnose_failures.py` |
| Modified existing files | 3 | `config.py`, `backends/pydantic.py`, `evaluation/reporting.py` |
| Generated source files | 1 | `src/convfinqa/prompts/v3_opt.py` |
| Committed data stores | 8 | 4 × `rules_<agent>_v3_opt.jsonl` + 4 × `rule_attempts_<agent>_v3_opt.jsonl` |
| Gitignored data artefacts | 7 | `case_results`, `diagnostic_results.{csv,html}`, `unresolved_cases`, `regression`, `model_accuracy_comparison`, `pydantic_predictions_v3_opt` |
| New LM-backed agents | 5 | `diagnostic_router_agent` + 4 `<agent>_fix_agent` specialists (all `LM_MAX`) |
| New system prompts | 5 | `DIAGNOSTIC_ROUTER_SYSTEM_PROMPT` + `FIX_<AGENT>_SYSTEM_PROMPT` × 4 |
| New CLI flags | 3 | `--stage`, `--reset-rules`, `--retry-n` (plus existing-style `--diagnose-only`, `--limit`, `--force`, `--skip-regression`, `--version`) |
| New env vars | 3 | `RETRY_N`, `MAX_PRIOR_ATTEMPTS_IN_PAYLOAD`, `RULES_DIR` (plus existing `DEEPSEEK_API_KEY`, `LM_MAX_MODEL`, `PROMPTS_VERSION`) |

## Implementation Steps

### Step 0 — Settings + CSS/JS extraction (prerequisite)

- `Settings.lm_max_model: str = "deepseek-reasoner"` (NOT `"deepseek-v4-pro"` — that string was a typo).
- `Settings.rules_dir: Path = Path("evaluation").resolve()` — overridable via `RULES_DIR`.
- `Settings.retry_n: int = 1` — overridable via `RETRY_N`. Validate `1 ≤ N ≤ 3`; raise on out-of-range. Default 1 means single-pass (no refinement loop).
- `Settings.max_prior_attempts_in_payload: int = 50` — overridable via `MAX_PRIOR_ATTEMPTS_IN_PAYLOAD`. Caps how many prior attempts per agent are surfaced to the diagnostic agent, bounding prompt size. Most-recent-N policy.
- No new `load_dotenv` calls anywhere — `Settings()` already loads `~/.env`.
- In `evaluation/reporting.py`: extract `PREDICTIONS_CSS`, `PREDICTIONS_JS`, and `render_filter_bar(extra_selects=...)` to module level. `diagnosis/results_html.py` imports them. No duplicated CSS in the repo.

### Step 1 — Router + 4 specialist fix agents (LM_MAX)

- In `backends/pydantic.py`: `LM_MAX = OpenAIChatModel(settings.lm_max_model, provider=_deepseek_provider)`. All five agents share this model.
- **Pydantic AI conformance**: the 5 diagnosis agents are built with `pydantic_ai.Agent(model, output_type=..., instructions=...)` exactly like the four production sub-agents in `src/convfinqa/backends/pydantic.py` (`triage_agent`, `preprocess_agent`, `retriever_agent`, `calculator_agent`). Same constructor signature, same provider object (`_deepseek_provider`), same OpenAI-compat chat-completions transport, structured output via `output_type=<BaseModel>`. The only differences are: (a) `LM_MAX` model id (`deepseek-reasoner`) vs `LM_MINI` (`deepseek-chat`), (b) no `.tool_plain` calls (the diagnosis agents don't expose tools — they return a single structured object), and (c) `instructions` strings are different. Treating the diagnosis stack as "just five more pydantic-ai agents wired up the same way as the production four" is intentional — it keeps Logfire instrumentation, retry semantics, and JSON-output validation uniform across the codebase. No bespoke LLM client, no raw `openai.AsyncOpenAI`.

#### `diagnosis/prompts.py` contains 5 system prompts:

The prompts are pure strings (or f-strings composing shared blocks) defined at module level. They are passed verbatim into `Agent(instructions=...)` at import time. Each prompt is structured the same way:

```
1. Role / scope            — one paragraph saying "you are X, you only do Y"
2. Domain knowledge        — what this agent needs to know to do its job
3. Investigation protocol  — step-by-step procedure to follow on each call
4. Output contract         — the BaseModel schema reminded inline so the LM doesn't drift
5. Hard constraints        — explicit forbids (no code, no cross-agent edits, etc.)
6. Worked example(s)       — at least one full input→output trace
```

##### 1. `DIAGNOSTIC_ROUTER_SYSTEM_PROMPT` — for `diagnostic_router_agent`

The router's job is to compare the gold answer/program against what each sub-agent produced and pin the blame on exactly one sub-agent (or `ambiguous`). It must NEVER write a `system_prompt_fix` — the router's `output_type=RouterDiagnosis` schema has no rule field, so the LM literally cannot, but the prompt also says so explicitly.

Sections:

1. **Role**: "You are the diagnostic router for a 4-agent ConvFinQA pipeline. Your only job is to classify which sub-agent caused the failure and explain why. You do NOT propose fixes — a specialist agent will do that downstream."

2. **Pipeline topology** (verbatim from this spec, §The Four Sub-Agents): Triage → (if program) Preprocess → Retriever → Calculator. Triage and Retriever run on every turn; Preprocess and Calculator only on program turns.

3. **IO log format guide**: what the `*_io` JSON blobs look like. `triage_io.output = {turn_type, conv_type}`. `preprocess_io.output = {sub_questions, program}`. `retriever_io.output = {values: [...]}` for program turns or `{answer: ...}` for number turns. `calculator_io.trajectory = [{tool_name, args, result}, ..., {tool_name: "finish", ...}]`.

4. **Investigation protocol — backward causal walk** (the user's mental model). Inspect each stage's output in reverse chronological order, asking at each layer "is this stage's output consistent with the gold?" The first stage whose output is INCONSISTENT with gold (AND whose upstream inputs were CONSISTENT) is the failed agent. Concretely, for a **program turn** the chain is:

   ```
   Triage → Preprocess → Retriever → Calculator → pred_answer
   ```

   Walk it backwards:

   - **Step A. Compare `pred_answer` vs `gold_answer`.** If equal, the case wasn't actually wrong — exit `ambiguous` with confidence 0.3. (Shouldn't happen given the loader filter, but guard.)
   - **Step B. Calculator check.** Look at `calculator_io.trajectory`. Given the retrieved values (Calculator's input) and the program (also Calculator's input), did Calculator execute correctly? Use this rule: if you replayed `pred_program` over `retriever_io.output.values` mentally and got `pred_answer`, Calculator did its job — move upstream. If you'd have gotten a different number, Calculator is at fault (`failed_agent="calculator"`, mode ∈ `wrong_tool_order | spurious_multiply_100 | missing_multiply_100_in_calc | wrong_operand_assignment | execution_error`). Stop.
   - **Step C. Retriever check.** Look at `retriever_io.output.values` against the table/text and against `preprocess_io.output.sub_questions`. Two sub-cases:
     - If each `sub_question` is well-specified (year + entity + metric) but a returned value is wrong or empty → Retriever fault (`wrong_retrieved_value`). Stop.
     - If a `sub_question` is vague (missing year, entity, or metric — e.g., "what was the share-based compensation cost?" with no year) → Retriever can't be blamed for guessing wrong. Walk upstream to Preprocess. (This is the Preprocess/Retriever boundary; see below.)
   - **Step D. Preprocess check.** Compare `preprocess_io.output.program` (op multiset) vs `gold_program` (op multiset, extracted via the regex in the Investigation Protocol section). Multisets differ → Preprocess fault, mode ∈ `missing_multiply_100 | wrong_subtract_direction | extra_or_missing_op | wrong_op`. Multisets match but sub-questions were the vague ones flagged in Step C → Preprocess fault, mode `vague_sub_questions`.
   - **Step E. Triage check.** Compare `pred_turn_type` vs `gold_turn_type`. Different → Triage fault, mode `wrong_turn_type`. (You should already have noticed this at the top of the trace because a wrong turn_type would mean Preprocess was skipped or run incorrectly.)
   - **Step F. Nothing matches.** No single stage looks clearly wrong → `failed_agent="ambiguous"`, confidence 0.3–0.5. Don't guess.

5. **Forward-walk fallback**: if the backward walk is inconclusive (e.g., values look right but program looks right but answer is still wrong — usually a numeric-tolerance edge case), do one forward pass: Triage output → Preprocess output → Retriever output → Calculator output, and report the first stage where output ≠ what gold would imply.

6. **Number-turn shortcut**: if `gold_turn_type == "number"`, Preprocess and Calculator are skipped in production. The only candidates are Triage (wrong turn_type) or Retriever (`wrong_direct_lookup`). Skip Steps B and D.

7. **Preprocess/Retriever boundary rules** (verbatim from §Investigation Protocol of this spec): sub-question missing year/entity/metric → Preprocess; clearly specifies but value wrong → Retriever; partially specified → ambiguous. This is the hardest judgment; bias toward Preprocess when in doubt because a vague sub-question makes Retriever a coin flip.

8. **Failure-mode quick-reference table**: list the canonical `failure_mode` tags per agent (the ones already enumerated in §Investigation Protocol). The router must pick one of these tags, not invent a new one.

9. **Output format spec**: remind the LM that `RouterDiagnosis = {failed_agent, failure_mode, failure_explanation (2–4 sentences), supporting_evidence (2–5 quoted IO snippets), confidence (0..1)}`. No `system_prompt_fix` field exists. Quote at least two IO snippets in `supporting_evidence` to ground the diagnosis.

10. **Worked example — diagnosis half only**: the "what was the net change … in percentage?" case from §Worked Example. Show the gold program `multiply(divide(subtract(B,A),A),100)` vs `pred_program=divide(subtract(B,A),A)`, walk Steps A–E, conclude `failed_agent="preprocess", failure_mode="missing_multiply_100"`. Do NOT include the fix half — that belongs in `FIX_PREPROCESS_SYSTEM_PROMPT`.

11. **Hard forbids**: "Do not output a rule, fix, or system_prompt patch. Do not name a sub-agent that does not exist. Do not invent a `failure_mode` tag outside the canonical list."

##### 2. `FIX_TRIAGE_SYSTEM_PROMPT` — for `triage_fix_agent`

- **Role**: "You write `system_prompt` rules for the Triage agent only. The Triage agent classifies `turn_type ∈ {number, program}` and `conv_type ∈ {Type I, Type II}` — nothing else."
- **Input layout**: the user message is `FixPayload.model_dump_json()`. Read `router_diagnosis.failure_mode` and `router_diagnosis.failure_explanation` to understand what to fix. Read `current_prompt` (the Triage v2 prompt) — anchor every rule to phrasing and structure already in v2, never invent new section headers.
- **Prior Rule Attempts** (shared block — extracted as `_PRIOR_ATTEMPTS_BLOCK` so all four specialists stay in sync): "Before proposing, scan `prior_rule_attempts`. (1) If a near-identical rule appears with `verify_result='passed'`, that rule is already in the in-memory baseline you're being asked to extend — DO NOT re-propose. Try a different angle or return a low-confidence `FixProposal` so the harness routes to unresolved. (2) If a near-identical rule appears with `verify_result='failed'` for the same `failure_reason`, that exact shape is known not to work — propose a meaningfully different rule. (3) `failure_reason='did_not_fix'` → broaden trigger or add a concrete example; `caused_regression` → narrow trigger with a guard condition."
- **Hard Constraint** (shared block — `_HARD_CONSTRAINT_BLOCK`): only `system_prompt` changes; no code, no other-agent edits, no model/tool swaps.
- **Output format**: `FixProposal{rule, fix_type, confidence (0..1), rationale (1–3 sentences)}`. The `rationale` must cite the specific clause of `router_diagnosis.failure_explanation` the rule addresses.

##### 3. `FIX_PREPROCESS_SYSTEM_PROMPT` — for `preprocess_fix_agent`

Same skeleton, scoped to Preprocess. Domain knowledge section covers: the DSL (`add/subtract/multiply/divide/exp/greater`), sub-question specification rules (year + entity + metric), the percentage convention (`multiply(..., 100)` outermost when the question asks for a percentage), and operand ordering for `subtract` and `divide`. Includes the worked example's **fix half** (`missing_multiply_100` → rule and rationale). Reuses `_PRIOR_ATTEMPTS_BLOCK` and `_HARD_CONSTRAINT_BLOCK`.

##### 4. `FIX_RETRIEVER_SYSTEM_PROMPT` — for `retriever_fix_agent`

Same skeleton, scoped to Retriever. Domain knowledge: table-cell vs text-span lookup discipline, never inventing sub-questions, number-turn direct-answer behaviour (when Preprocess is skipped). Reuses shared blocks.

##### 5. `FIX_CALCULATOR_SYSTEM_PROMPT` — for `calculator_fix_agent`

Same skeleton, scoped to Calculator. Domain knowledge: tool-call ordering, operand assignment (first retrieved value = A, second = B, …), the `multiply(..., 100)` override authority, the `finish` tool contract. Reuses shared blocks.

All four specialist prompts compose `_PRIOR_ATTEMPTS_BLOCK` and `_HARD_CONSTRAINT_BLOCK` via f-string includes so the four stay in sync; only sections 1 (Role) and 2 (Domain knowledge) and the worked-example half are agent-specific.

#### `diagnosis/agents.py`:

Same constructor shape as the production sub-agents (`src/convfinqa/backends/pydantic.py`). The agents are module-level singletons constructed at import time; `instructions=` is bound on the `Agent` object (immutable for the lifetime of the process), and the user-supplied payload is JSON-rendered via `payload.model_dump_json()` and passed to `.run()` as the user message. No tools attached — diagnosis agents always emit a single structured `BaseModel` and never call tools.

```python
from pydantic_ai import Agent
from convfinqa.backends.pydantic import LM_MAX
from convfinqa.diagnosis.models import (
    FixPayload, FixProposal, RouterDiagnosis, RouterPayload,
)
from convfinqa.diagnosis.prompts import (
    DIAGNOSTIC_ROUTER_SYSTEM_PROMPT,
    FIX_TRIAGE_SYSTEM_PROMPT,
    FIX_PREPROCESS_SYSTEM_PROMPT,
    FIX_RETRIEVER_SYSTEM_PROMPT,
    FIX_CALCULATOR_SYSTEM_PROMPT,
)

diagnostic_router_agent = Agent(
    LM_MAX, output_type=RouterDiagnosis,
    instructions=DIAGNOSTIC_ROUTER_SYSTEM_PROMPT,
)
triage_fix_agent = Agent(
    LM_MAX, output_type=FixProposal,
    instructions=FIX_TRIAGE_SYSTEM_PROMPT,
)
preprocess_fix_agent = Agent(
    LM_MAX, output_type=FixProposal,
    instructions=FIX_PREPROCESS_SYSTEM_PROMPT,
)
retriever_fix_agent = Agent(
    LM_MAX, output_type=FixProposal,
    instructions=FIX_RETRIEVER_SYSTEM_PROMPT,
)
calculator_fix_agent = Agent(
    LM_MAX, output_type=FixProposal,
    instructions=FIX_CALCULATOR_SYSTEM_PROMPT,
)

FIX_AGENTS: dict[str, Agent] = {
    "triage":     triage_fix_agent,
    "preprocess": preprocess_fix_agent,
    "retriever":  retriever_fix_agent,
    "calculator": calculator_fix_agent,
}

async def route_case(payload: RouterPayload) -> RouterDiagnosis:
    """Step 1 — Diagnose: classify-only. One LM_MAX call per case (called once per case, cached across retries)."""
    result = await diagnostic_router_agent.run(payload.model_dump_json())
    return result.output

async def propose_fix(failed_agent: str, payload: FixPayload) -> FixProposal:
    """Step 2 — Route+Fix: dispatched to one of the four specialists by failed_agent. One LM_MAX call per attempt (1..retry_n)."""
    result = await FIX_AGENTS[failed_agent].run(payload.model_dump_json())
    return result.output
```

`route_case` is called once per case (Step 1). `propose_fix` is called once per attempt within a case (Step 2; up to `retry_n` times, default 1). Both are `async` to stay consistent with the rest of the pydantic-ai code paths, but the harness awaits them sequentially — no `asyncio.gather` across cases or across attempts within a case. **Parity with production**: this is structurally identical to how `triage_agent` / `preprocess_agent` / `retriever_agent` / `calculator_agent` are built with `LM_MINI` in `backends/pydantic.py` — same provider, same `Agent` class, same `output_type=<BaseModel>` pattern, same `.run(json_str)` invocation. Operationally there is no new code path; the diagnosis stack is "five more pydantic-ai agents with a bigger model and different prompts."

### Step 2 — Verify (patch + replay-to-failed-turn + check)

`diagnosis/verify.py` (no LLM calls of its own — but it does execute the four production sub-agents, which call DeepSeek). Key contract:

#### What system prompt does the verify replay actually use?

This is the critical question, because the in-memory prompt has THREE layered pieces:

1. **Baseline v2** — the verbatim string from `src/convfinqa/prompts/v2.py` for the failed agent.
2. **Historical passed rules** — the union of all `rule_text` values in `rules_<failed_agent>_v3_opt.jsonl` (after `supersedes` filtering) that earlier cases in this run — or earlier runs — have already verified as passing. These are pulled in by `_assemble_current_prompts()` before each case, so the sub-agent's prompt grows monotonically as the case loop progresses.
3. **The proposed new rule** — the `FixProposal.rule` string just returned by the specialist fix agent for THIS case. Not yet committed to the rules store; it will only be appended on a passing verify.

The patched prompt fed to the production sub-agent during replay is:

```
<v2 baseline>
\n\n## Additional Rules (automated patch)
1. (<rule_id_1>) <historical_passed_rule_1>
2. (<rule_id_2>) <historical_passed_rule_2>
…
N. (proposed) <FixProposal.rule>     ← the candidate being verified
```

Pieces (1) + (2) form the sub-agent's "current baseline" — that is the prompt the case would have run with if no new rule were proposed. Piece (3) is the candidate the verify is testing. Acceptance therefore means: *adding this candidate on top of the already-passing baseline still fixes turn k and still doesn't regress turns 0..k-1.* The patched prompt includes both historical passed rules AND the proposed rule. The verify never tests the proposed rule against v2 in isolation.

A consequence: **a candidate rule that would have passed against bare v2 can fail in verify** if it conflicts with a rule already in that sub-agent's rules store. That's the desired behaviour — the source of truth is what the assembled v3_opt.py actually ships, not what a clean-room test would have accepted. It also explains why case order in the loop is deterministic-sequential: the baseline that case `n+1` sees depends on which rules cases `0..n` promoted.

#### Key contract

- `build_patched_prompt(failed_agent, patch, current_prompts)`: returns a new dict where ONLY `failed_agent`'s prompt has `\n\n## Additional Rules (automated patch)\n{patch}` appended. Other three are byte-identical. `current_prompts` is the dict returned by `_assemble_current_prompts()` — i.e., v2 + historical passed rules already merged in. `patch` is the single proposed candidate appended on top.
- `verify_patch(...)`: builds patched agents via `make_agents(modified_prompts)`, then **replays turns 0..k** of conversation `report_id`, where `k` is the originally-failed `turn_index`:
  - Start with an empty `ConversationHistory`.
  - For each turn `t` in `0..k`: call `run_turn(question_t, report_id, conversation, agents=agents, capture=capture_t)`; append `(question_t, pred_answer_t)` to history; record `correct_t = numeric_match(pred_answer_t, gold_answer_t)`.
  - Replay must run through `k` (not just turn `k` in isolation) because turns 1..k may reference earlier Q&A — the patched agents need to rebuild that history themselves. Predicted answers feed forward (this is the production behaviour we're validating).
  - Turns `k+1..N` are NOT replayed — they were cascade-poisoned in the original prediction and aren't part of the accept criterion.
  - Accept iff `correct_k == True` AND `correct_t == True` for every `t in 0..k-1`. (Turns 0..k-1 were correct in the original run by construction — `k = min(turn_index where correct==False)`. If the patch makes any of them flip to wrong, that's a regression and we reject.)
  - Returns a `FixAttempt` containing per-turn results for 0..k, the IOs of the **first failing turn in 0..k** (or turn `k`'s IOs if everything passed), and an overall `correct` flag.

GOTCHAs:
- `run_turn` accepts `agents=` and `capture=`. `make_agents` rebuilds all four agents even though only one prompt changed (`pydantic_ai.Agent.instructions` is bound at construction). `ConversationHistory.append` uses kwargs (see `data/schemas.py:42`).
- Cost: each verify executes `k+1` turn-runs. For the median case (`k≈2`) that's 3 turn-runs per verify. With default `retry_n = 1` that's 3 turn-runs per case; with `retry_n = 3` it can reach 9 turn-runs per case. The case loop is fully sequential — wall time is the sum of per-case work, not the max.
- Stop early on the first failing turn within a single replay — but record which turn failed so the next attempt's specialist call sees the right IOs in `prior_attempts`.

**Proposal-time evidence (not a retry signal)**: there is no special "iteration 2/3 retry signal" — every verify result (pass or fail, with `first_failing_turn` and `failure_reason`) is appended to `rule_attempts_<agent>_v3_opt.jsonl` and is fed back into the specialist agent on the *next case* it sees. The specialist is guided by its system prompt to do an explicit three-way review before emitting a `FixProposal`:

1. **Read the current prompt** (`fix_payload.current_prompt` — the live baseline for that sub-agent: v2 + all already-passing rules concatenated). Understand what rules already exist; do not re-state them. The new rule must add information not already covered.
2. **Read all passed rules** (entries in `prior_rule_attempts` with `verify_result="passed"`). These rules are in the baseline above. Treat them as constraints — any new rule must be consistent with them. Do not propose anything that contradicts a passed rule.
3. **Read all failed rules** (entries in `prior_rule_attempts` with `verify_result="failed"`). These are known dead ends — do not re-propose. Read `failure_reason`:
   - `did_not_fix` → that exact phrasing/trigger didn't address the failure mode. A new attempt must take a meaningfully different angle (different trigger words, a worked example, a different rule type).
   - `caused_regression` → that rule was too aggressive. A new attempt must narrow the trigger with a guard condition.
   - `duplicate_patch` / `ambiguous_followup` → housekeeping flags, not technical failures.

The specialist then asks itself: *"If I append this proposed rule to the current prompt, what does that prompt now say, and could any phrasing in the new rule conflict with an existing instruction or with another passed rule?"* If yes, narrow the trigger or revise. This forward-impact reasoning is what the system prompt's Hard Constraint + Prior Rule Attempts blocks ask the LM to do — see Step 1 §`FIX_TRIAGE_SYSTEM_PROMPT` and the shared `_PRIOR_ATTEMPTS_BLOCK`. The `rationale` field on `FixProposal` should cite which passed rules the new rule complements (or, on a refinement, which failed rule it intentionally diverges from).

When `retry_n > 1`, iterations 2 and 3 of the *same case* additionally see within-case `prior_attempts` in `FixPayload`, but no additional retry-signal machinery is needed — the attempt history already includes them via `append_attempt(...)`, which is written on every verify regardless of pass/fail.

**Source-of-truth for what the system "should have" produced**: use `gold_program` as the canonical reference for `turn_type=program` cases. `gold_program` encodes both the operands (the numbers that go in, expressed as positional references A, B, C, … to retrieved values) and the calculation required (the DSL ops and their nesting). The conversation history's `(question, answer)` pairs are not the source of truth — they are a thin trace of what production saw. The patched-agent replay must therefore be evaluated against `gold_program`'s logical structure, not against a reconstructed history.

In practice this means:

- For `turn_type=program` turns, the verify success criterion is "executed answer matches gold_answer", but the diagnostic and specialist agents reason about correctness by comparing `pred_program` and the retrieved values against `gold_program` decomposed into (ops, operand positions). A passing verify with a different program shape than gold is still a pass (execution accuracy), but the specialist should flag low confidence if the patched program structurally diverges from gold.
- For `turn_type=number` turns, gold_program is empty/trivial; the source of truth is `gold_answer` and the gold cell location in the table/text.
- The replay does NOT need a richer `ConversationHistory` containing prior sub-questions/programs to be faithful. Gold_program tells the diagnostic and specialist agents exactly what the chain *should* have produced; the (question, answer) trace is only used as production sees it during replay. The two are complementary: replay reproduces production behaviour, gold_program tells us what production should have done.

### Step 3 — Loader

`diagnosis/loader.py`: `load_first_wrong_cases(csv_path) -> (list[DiagnosticPayload], pd.DataFrame)`. Filter: `correct == False`, then `min(turn_index)` per `report_id`. Parse `*_io` strings with a tolerant `_parse_io` that returns `None` on empty/invalid JSON. Inject the four current v2 prompts into each payload.

### Step 3.5 — Rules store + assembler

`diagnosis/rules_store.py`:
- `rules_path(agent) = settings.rules_dir / f"rules_{agent}.jsonl"`.
- `attempts_path(agent) = settings.rules_dir / f"rule_attempts_{agent}.jsonl"`.
- `read_rules(agent)`: returns active rules (filters out any `rule_id` referenced in a later rule's `supersedes`). Reads `rules_<agent>_v3_opt.jsonl` only.
- `read_attempts(agent, *, limit=settings.max_prior_attempts_in_payload)`: returns the last `limit` attempts (both passed and failed) from `rule_attempts_<agent>_v3_opt.jsonl`, most-recent last. Used to feed `prior_rule_attempts` into the diagnostic payload.
- `append_rule(agent, rule_text, fix_type, confidence, report_id, turn_index, supersedes=None) -> str`: appends one JSON line to `rules_<agent>_v3_opt.jsonl`; returns `rule_id = f"{agent[:4]}-{ts}-{uuid6}"`.
- `append_attempt(agent, rule_text, fix_type, confidence, report_id, turn_index, verify_result, first_failing_turn=None, failure_reason=None, promoted_rule_id=None) -> str`: appends one JSON line to `rule_attempts_<agent>_v3_opt.jsonl`. Always called after every verify (pass or fail). Returns `attempt_id`.
- `reset_rules(agent=None)`: truncates one or all `rules_<agent>_v3_opt.jsonl` files AND the matching `rule_attempts_<agent>_v3_opt.jsonl` files in the same call. Single function, single switch — the two stores are always reset together because a rules wipe without an attempts wipe leaves orphan `promoted_rule_id` pointers in the attempt history.

`diagnosis/assembler.py`:
- `assemble_prompts(base, rules_by_agent)`: for each agent, if rules exist, append `\n\n## Additional Rules (automated patch)\n\n1. (rule_id) rule…\n2. ...` to the v2 base.
- `write_v3_opt_module(prompts)`: writes `src/convfinqa/prompts/v3_opt.py` exposing the four constants. Escape any `"""` in bodies (current v2 has none, defensive). Idempotent.
- `assemble_v3_opt()`: reads stores via `all_rules()`, calls assemble + write, `importlib.reload` the v3_opt module. Loaded downstream via `convfinqa.prompts.load("v3_opt")` — **do not** write a parallel loader.

### Step 4 — Harness loop (per-case, fully sequential)

`diagnosis/harness.py`:

- `run_case(router_payload, full_df, *, diagnose_only) -> CaseResult`. Implements the **per-case three-step flow** end-to-end:

  **Step 1 — Diagnose** (called once per case, even on retries):
  - `_assemble_current_prompts()` (cheap; 4 small file reads — picks up rules added by earlier cases).
  - Inject the four current prompts into `router_payload`.
  - `router_diagnosis = await route_case(router_payload)` — one `LM_MAX` call.
  - Persist `(case, router_diagnosis)` to `case_results_v3_opt.jsonl` immediately so a crash mid-loop doesn't lose work.
  - If `router_diagnosis.failed_agent == "ambiguous"`: append a placeholder `FixAttempt` with `correct=False` and route to unresolved. No fix attempt is made — by definition there is no single subagent to send to a specialist.
  - If `diagnose_only=True`: stop here. Append placeholder `FixAttempt(iteration=1, turn_results=[], correct=False, first_failing_turn=None, *_io=None)`; no fix, no verify, no rule writes.

  **Steps 2 + 3 — Route+Fix and Verify** (attempt loop, 1..`settings.retry_n`):

  ```python
  prior_attempts: list[FixAttempt] = []
  for attempt_idx in range(1, settings.retry_n + 1):
      agent = router_diagnosis.failed_agent
      # Step 2 — ROUTE + FIX
      fix_payload = FixPayload(
          ...,
          router_diagnosis=router_diagnosis,           # cached from Step 1, not re-called
          current_prompt=current_prompts[agent],        # v2 + already-passing rules
          prior_rule_attempts=read_attempts(agent),     # cross-run history for THIS sub-agent only
          prior_attempts=list(prior_attempts),          # within-case history
      )
      fix = await propose_fix(agent, fix_payload)      # one LM_MAX call into FIX_AGENTS[agent]
      if not fix.rule.strip():
          break                                         # empty rule → route to unresolved
      if fix.rule in {a.patch_applied for a in prior_attempts}:
          break                                         # within-case duplicate-fix guard
      # Step 3 — VERIFY
      attempt = await verify_patch(agent, fix.rule, ...)  # replay turns 0..k; no LLM of its own
      prior_attempts.append(attempt)
      append_attempt(agent, fix.rule, fix.fix_type, fix.confidence,
                     report_id, turn_index, attempt.verify_result,
                     first_failing_turn=attempt.first_failing_turn,
                     failure_reason=attempt.failure_reason)
      if attempt.correct:
          rule_id = append_rule(agent, fix.rule, fix.fix_type, fix.confidence,
                                report_id, turn_index)
          # backfill promoted_rule_id on the attempt line we just wrote:
          update_attempt(attempt_id, promoted_rule_id=rule_id)
          break                                         # resolved → stop attempt loop
      # else: verify failed → loop back to Step 2 (router NOT re-called)
  ```

  Notes on the loop:
  - The router is called exactly once per case. On retry, only the specialist Fix LLM and the verify replay re-run. This is the cost model the user opted for.
  - `prior_attempts` grows monotonically within the case — that's the within-case retry signal the specialist sees in `FixPayload.prior_attempts` on iteration 2/3.
  - `append_attempt(...)` is called after every verify (pass or fail). `append_rule(...)` is called only on a passing verify. The attempt line is written AFTER the rule so the `promoted_rule_id` is known at attempt-write time.

- `run_harness(payloads, full_df, *, diagnose_only)`:
  - Sequential `for payload in payloads: await run_case(payload, full_df, diagnose_only=diagnose_only)`. **Fully sequential** — one case completes (or unresolves) before the next begins. No `asyncio.gather` anywhere. Rationale: rules added by case `n` must be in the live baseline for case `n+1`'s `_assemble_current_prompts()` call; concurrent cases would race on the rules store.
  - After the case loop completes, call `assemble_v3_opt()` then `run_regression()` (unless `--skip-regression`).

GOTCHAs:
- Logs are linear by design — one case completes before the next begins; one attempt completes before the next attempt within a case. Prefix every log line with `[<failed_agent>]` for easy filtering. NEVER use `asyncio.gather` across cases or attempts.
- The same `prior_attempts` list grows between iterations within a case — that's the within-case retry signal (only active when `retry_n > 1`).
- `case_results_v3_opt.jsonl` is appended incrementally so partial runs are resumable on re-invocation (the loader skips cases already present unless `--force`).

### Step 5 — Results writer + HTML clone

`diagnosis/results_writer.py`: emit one CSV row per `(case, attempt)` pair. Group A columns are joined from `full_df` by `(report_id, turn_index)`. In `--diagnose-only`, Group C cells are emitted as empty strings (not missing columns).

`diagnosis/results_html.py`: render an HTML clone of `pydantic_predictions_v2.html`. Import `PREDICTIONS_CSS`, `PREDICTIONS_JS`, `render_filter_bar` from `evaluation.reporting`. Extend the filter bar with `<select id="fa-filter">` (failed_agent) and `<select id="att-filter">` (attempt_id); extend `applyFilters()` to honour `data-fa` and `data-att`. Row class: `row-correct` / `row-wrong` / `row-pending` (blank in diagnose-only). Add `.placeholder` style for `—` cells and `.fix-box` accent style for `system_prompt_fix`. JSON cells use the existing `<details><summary>view</summary><pre>...</pre></details>` pattern; long text > 200 chars collapses similarly.

`diagnosis/aggregator.py`: `build_unresolved_cases(results, unresolved_name)` — writes router-`ambiguous` cases + cases that exhausted `retry_n` without a passing verify. Not called in `--diagnose-only`. Rule lines in JSONL stores REPLACE the old `prompt_patches.json` design.

### Step 5.5 — Post-loop regression

`diagnosis/regression.py`: subprocess-invoke `uv run convfinqa-eval-api --version v3_opt` with `PROMPTS_VERSION=v3_opt`, `REUSE_CACHE=0` in the env (the child re-instantiates `Settings()` — don't mutate `settings` in the parent). Read the resulting `pydantic_predictions_v3_opt.csv`, join with v2 on `(report_id, turn_index)`, classify each row's `delta` ∈ {`fixed`, `regressed`, `unchanged_right`, `unchanged_wrong`}. Write `regression_v3_opt.csv` + `model_accuracy_comparison_v3_opt.csv` (with `fixed`, `regressed`, `v2_accuracy`, `v3_opt_accuracy`, `net_delta`). Expensive — `--skip-regression` allowed for local dev only.

### Step 6 — Entry point

`scripts/diagnose_failures.py`. Flags:
- `--diagnose-only` — run Step 1 (diagnose) only for every case; no Step 2/3, no rule writes, no v3_opt regeneration.
- `--stage {all,assemble,regression}` (default `all`) — short-circuit to a single post-loop stage. `--stage assemble` and `--stage regression` skip the harness entirely.
- `--reset-rules` — wipes BOTH `rules_<agent>_v3_opt.jsonl` and `rule_attempts_<agent>_v3_opt.jsonl` for all four agents. Default (flag absent): both stores are preserved across runs.
- `--force`, `--limit N`, `--version v2`, `--skip-regression`
- `--retry-n N` (1–3; default `settings.retry_n` which itself defaults to 1). Overrides the setting for this run only.
- `--no-diagnose-cache` — ignore any existing `case_results_v3_opt.jsonl` and re-call the router for every case. Default (flag absent) reuses prior router diagnoses keyed by `(report_id, turn_index)`. See §Diagnose Cache.

Behaviour:
- `--stage assemble` / `--stage regression`: short-circuit; no harness invocation.
- Default (`--stage all`): load the diagnose cache from `case_results_v3_opt.jsonl` (if present and `--no-diagnose-cache` not set), then run the per-case loop (Step 1 reuses cache if hit, else LLM call; Step 2/3 attempt loop, sequential across all cases), then write CSV/HTML, build unresolved, assemble, regression (unless `--skip-regression`). `case_results_v3_opt.jsonl` is rewritten incrementally during the loop — the cache is held in memory once loaded.
- `--diagnose-only`: Step 1 only for every case (also reuses cache); no fix proposals, no verifies, no rule writes, no v3_opt regeneration. Use this to *populate* the cache cheaply and review router output before paying Step 2+3 cost.
- Cache promotion flow: a `--diagnose-only` run followed by a full run reuses every diagnosis from the first run — zero router LLM cost on the second run. See §Diagnose Cache → Promotion path.

Print a summary: cases processed, resolved / unresolved counts, per-attempt counts (how many resolved on attempt 1 vs 2 vs 3), list of artefacts.

### Step 7 — Tests (`tests/test_diagnose_failures.py`)

All tests must pass without `DEEPSEEK_API_KEY` — mock `diagnostic_router_agent.run`, each `FIX_AGENTS[agent].run`, and `run_turn`. Coverage groups:

- **Loader**: first-wrong = min(turn_index) per report_id; one entry per report_id; excludes fully-correct convs; `_parse_io` tolerates empty/invalid.
- **Verify**: `build_patched_prompt` only changes failed agent; appends under `## Additional Rules (automated patch)`; replay covers turns 0..k where k is originally-failed turn_index; predicted answers feed forward (not gold); `correct` is True iff turn k now matches gold AND every prior turn 0..k-1 still matches gold (no regression); `first_failing_turn` is the lowest-index mismatch in 0..k; stage IOs captured come from the first failing turn (or turn k when replay is fully correct); replay stops early on first failing turn; turns k+1..N are not replayed.
- **Per-case loop, default `retry_n=1`**: exactly one router call + one specialist call + one verify per case (resolved or unresolved). Router is not called more than once per case under any `retry_n` setting.
- **Per-case loop, `retry_n=3`**: resolves on attempt 1 / 2 / 3 and terminates unresolved after 3; attempt 2 sees `prior_attempts` of length 1; attempt 3 sees length 2; duplicate-patch terminates early; ambiguous terminates early (no fix attempt at all); cases processed sequentially.
- **Harness diagnose-only**: one router call per case, no fix, no verify; placeholder `attempt_id=1` with blank harness cols; no retries on ambiguous; no rule writes.
- **Results writer / HTML**: one row per attempt; `--diagnose-only` blank Group C; full mode populated; Group A joined from full_df; HTML contains dark-theme CSS, filter selects, `—` placeholder.
- **Constraint**: `FixProposal.rule` has no `def `, `import `, `Agent(`, `model=`, `temperature=`, `tools=`, `class `, `pipeline`. Enforced by a post-output regex check inside `propose_fix`.
- **Router separation**: `diagnostic_router_agent`'s `RouterDiagnosis` schema has no `system_prompt_fix` / `rule` field; the router cannot accidentally propose a fix. `route_case` calls the router exactly once per case (regardless of `retry_n`) and never invokes a specialist.
- **Specialist routing**: `propose_fix("triage", ...)` invokes `triage_fix_agent` (not any other); same for the other three. A test parametrised over the four agents asserts the mock for the correct specialist was called and the mocks for the other three were NOT.
- **FixPayload scoping**: `prior_rule_attempts` passed to `triage_fix_agent` contains ONLY triage attempts, not preprocess/retriever/calculator attempts. Verified by inspecting the JSON sent to the mock.
- **Rules store**: `append_rule` creates file; `read_rules` preserves order; `supersedes` filters; `reset_rules(agent)` vs `reset_rules()` — both also wipe the matching `rule_attempts_<agent>_v3_opt.jsonl` (single switch).
- **Attempts store**: `append_attempt` writes on both pass and fail; `read_attempts(limit=N)` returns most-recent-N; pass entries have `promoted_rule_id` set, fail entries have `first_failing_turn` and `failure_reason` set; `--reset-rules` clears attempts as part of the same operation; default (flag absent) leaves attempts intact.
- **Specialist awareness**: when `prior_rule_attempts[<agent>]` is non-empty, the rendered `FixPayload` JSON contains a section listing them; tests can mock the specialist agent's `.run` to assert the rendered text includes the prior-attempts JSON when present.
- **Assembler**: empty stores → `v3_opt.py == v2.py`; one rule per agent → header appears; written module imports cleanly; triple quotes round-trip; `prompts.load("v3_opt")` reflects assembly.
- **Sequencing**: cases run strictly one-at-a-time (no `asyncio.gather` anywhere); attempts within a case run strictly one-at-a-time; rules added by case `n` are visible to case `n+1`'s `_assemble_current_prompts()` call; `append_rule` only on a passing verify; `append_attempt` on every verify.
- **Regression**: delta classification; summary contents.
- **Entry point**: cache reuse without `--force`; `--limit` truncation; `--diagnose-only` does not write rules or v3_opt; `--reset-rules` calls reset before the case loop; `--stage assemble` / `--stage regression` short-circuit; `--skip-regression` skips regression; no `load_dotenv` import.
- **Diagnose cache**: with a pre-existing `case_results_v3_opt.jsonl` containing diagnoses for `(report_id=R, turn_index=k)`, a subsequent run on the same case skips the router LLM call and reuses the cached `RouterDiagnosis` (verified by asserting `route_case` mock was not awaited for the cached key). `--no-diagnose-cache` forces a re-diagnose for every case (mock awaited per case). A `--diagnose-only` run that writes diagnoses to the JSONL, followed by a full-mode run, reuses every diagnosis (mock not called for any case in the second run). Cases not present in the cache file still pay the router call. Empty-file or absent-file cache is a no-op, not an error.
- **Settings**: `lm_max_model` default is `"deepseek-reasoner"`; env override works; `rules_dir` default ends in `evaluation`; `retry_n` default is `1`; `RETRY_N=2` override works; out-of-range (0 or 4) raises.

## Integration Points

```yaml
ENV (via settings — no load_dotenv anywhere):
  DEEPSEEK_API_KEY   required
  LM_MAX_MODEL       default "deepseek-reasoner"
  RULES_DIR          default "evaluation"
  PROMPTS_VERSION    set to "v3_opt" by post-loop regression subprocess only
  RETRY_N  default 1 (range 1..3)

MODIFY:
  src/convfinqa/config.py                  # add lm_max_model + rules_dir
  src/convfinqa/backends/pydantic.py       # LM_MAX using settings
  src/convfinqa/evaluation/reporting.py    # extract CSS/JS/filter_bar

INPUT (read-only):
  evaluation/pydantic_predictions_v2.csv
  src/convfinqa/prompts/v2.py
  evaluation/rules_<agent>_v3_opt.jsonl           # if present
  evaluation/rule_attempts_<agent>_v3_opt.jsonl   # if present — feeds specialist agent's prior_rule_attempts

OUTPUT:
  evaluation/diagnostic_results_v3_opt.{csv,html}
  evaluation/case_results_v3_opt.jsonl
  evaluation/rules_<agent>_v3_opt.jsonl           # source of truth for v3_opt.py (×4) — passes only
  evaluation/rule_attempts_<agent>_v3_opt.jsonl   # full attempt history (×4) — passes AND failures
  evaluation/unresolved_cases_v3_opt.json
  evaluation/regression_v3_opt.csv
  evaluation/model_accuracy_comparison_v3_opt.csv
  evaluation/pydantic_predictions_v3_opt.csv
  src/convfinqa/prompts/v3_opt.py          # GENERATED

ONLY THE HARNESS WRITES:
  src/convfinqa/prompts/v3_opt.py          # via diagnosis.assembler
  evaluation/rules_<agent>_v3_opt.jsonl           # via diagnosis.rules_store (append_rule)
  evaluation/rule_attempts_<agent>_v3_opt.jsonl   # via diagnosis.rules_store (append_attempt)

COMMIT: rules_<agent>_v3_opt.jsonl, rule_attempts_<agent>_v3_opt.jsonl, prompts/v3_opt.py
GITIGNORE: case_results_v3_opt.jsonl, diagnostic_results_v3_opt.{csv,html},
           unresolved_cases_v3_opt.json, regression_v3_opt.csv,
           pydantic_predictions_v3_opt.csv, model_accuracy_comparison_v3_opt.csv
```

## Validation

```bash
# Level 1 — Syntax & style
uv run ruff check scripts/diagnose_failures.py src/convfinqa/diagnosis/ tests/ --fix
uv run mypy src/convfinqa/diagnosis/ scripts/diagnose_failures.py

# Level 2 — Unit tests (no API key)
uv run pytest tests/test_diagnose_failures.py -v

# Level 3 — Smoke (no LLM)
uv run python -c "
from pathlib import Path
from convfinqa.diagnosis.loader import load_first_wrong_cases
payloads, df = load_first_wrong_cases(Path('evaluation/pydantic_predictions_v2.csv'))
assert len({p.report_id for p in payloads}) == len(payloads)
print(f'First-wrong: {len(payloads)} / Full df: {len(df)}')
"

# Level 4 — Diagnose-only smoke (needs DEEPSEEK_API_KEY)
uv run python scripts/diagnose_failures.py --diagnose-only --limit 5 --force --reset-rules

# Level 5 — Full pipeline smoke (skip regression for speed)
uv run python scripts/diagnose_failures.py --limit 3 --force --reset-rules --skip-regression
diff src/convfinqa/prompts/v2.py src/convfinqa/prompts/v3_opt.py | head -40

# Level 5b — Same smoke with one retry enabled
uv run python scripts/diagnose_failures.py --limit 3 --force --reset-rules --skip-regression --retry-n 2

# Assemble alone (after editing a JSONL by hand)
uv run python scripts/diagnose_failures.py --stage assemble

# Regression alone
uv run python scripts/diagnose_failures.py --stage regression
```

## Anti-Patterns

- **DO NOT** parallelise anything in the case loop — neither cases (rules must compound sequentially: rules added by case `n` must be visible to case `n+1`) nor attempts within a case (each attempt's `prior_attempts` depends on the previous attempt's verify result). No `asyncio.gather` in `run_harness` or `run_case`.
- **DO NOT** diagnose all wrong rows. Filter to first-wrong-per-conversation; downstream wrongs are cascade-poisoned.
- **DO NOT** re-call the router on retry — the diagnosis is cached for the lifetime of the case. Only Step 2 (specialist Fix) and Step 3 (verify) repeat. This is intentional: a router that flip-flops between agents across retries makes attempts impossible to interpret and doubles router cost.
- **DO** feed predicted answers forward during verify replay — that's the production behaviour we're trying to validate. The patch must hold up under realistic cascade conditions, not against an idealised gold-history environment.
- **DO NOT** swap predicted answers for gold answers during verify replay — that would isolate the patched turn from the rest of the conversation and accept patches that look good in a vacuum but don't unblock the real conversation flow.
- **DO NOT** replay turns past `k` (the originally-failed turn) during verify — turns `k+1..N` were cascade-poisoned in the original run and their correctness in the patched replay isn't part of the accept criterion. The post-loop regression catches dataset-wide regression.
- **DO NOT** skip turns 0..k-1 and call `run_turn` for turn k in isolation — the patched agents must rebuild conversation history themselves so multi-turn references in turn k resolve against history the patched pipeline produced, not against gold.
- **DO NOT** use `LM_MINI` for diagnosis or fix — both router and four specialists must use `LM_MAX` from `settings.lm_max_model`.
- **DO NOT** hardcode `"deepseek-v4-pro"` (typo) or any model name — go through settings.
- **DO NOT** call `load_dotenv` in `diagnosis/` or `scripts/diagnose_failures.py`.
- **DO NOT** create a separate refiner agent — the same specialist Fix agent handles all retries via `prior_attempts` in `FixPayload`.
- **DO NOT** propose anything but a `system_prompt` change.
- **DO NOT** apply the same patch twice in one case — duplicate-fix guard terminates the attempt loop.
- **DO NOT** retry on router `"ambiguous"` — terminate the case immediately and route to unresolved (no fix attempt is made at all).
- **DO NOT** treat `retry_n` as a number of retries — by user choice it is the **total attempts cap**. `retry_n=1` ⇒ 1 attempt, no retries (the default). `retry_n=3` ⇒ up to 3 attempts (i.e. up to 2 retries).
- **DO NOT** auto-promote `rules_<agent>_v3_opt.jsonl` into `prompts/v3.py` — a human reviews regression + rule lists, then hand-writes `v3.py` if desired.
- **DO NOT** modify `prompts/v2.py` from the harness.
- **DO NOT** hand-edit `prompts/v3_opt.py` — edit the JSONL store and re-run `--stage assemble`.
- **DO NOT** duplicate `evaluation/reporting.py` CSS into `diagnosis/results_html.py`.
- **DO NOT** silently skip the regression in CI — `--skip-regression` is for local dev only.
- **DO NOT** write a parallel `load_v3_opt_prompts()` — use `convfinqa.prompts.load("v3_opt")`.
- **DO** commit `prompts/v3_opt.py`, the four `rules_<agent>_v3_opt.jsonl`, and the four `rule_attempts_<agent>_v3_opt.jsonl` once a run has produced meaningful accumulation. The attempts log is cheap to store and makes future runs of the specialist agents strictly smarter.
- **DO NOT** write failed attempts into `rules_<agent>_v3_opt.jsonl` — that file is the source of truth for `v3_opt.py` and must contain only verified passes. Failed attempts live in `rule_attempts_<agent>_v3_opt.jsonl`.
- **DO NOT** read `rule_attempts_<agent>_v3_opt.jsonl` from the assembler — the attempts log is for the specialist agents' awareness only and must never influence the generated `v3_opt.py`.

## Confidence: 7.5 / 10

Per-case three-step flow (Diagnose → Route+Fix → Verify, with an attempt loop on Step 2+3) — fully sequential by design across cases and across attempts within a case. Only one sub-agent's `system_prompt` is being updated and tested at a time, so there is no benefit to concurrency. The attempt loop is bounded by `retry_n`: one router call per case (cached across retries) + up to `retry_n` specialist calls + up to `retry_n` verify replays. **Default `retry_n = 1`** means each case costs exactly one router + one specialist + one replay; operators can opt into 2 or 3 for harder cases. Replay cost is **(k+1) turns × pipeline cost per turn**, where k is the originally-failed turn index — typically 1–3, occasionally higher in Type II conversations. Main risks:

- **Gold-program as source of truth**: for `turn_type=program` cases the router and specialist agents are instructed to treat `gold_program` as the canonical reference — it specifies both the numbers that should flow in (operand positions A, B, C, … mapped to retrieved values) and the calculation chain that should execute (DSL ops + nesting). Conversation history's `(question, answer)` pairs only capture what production observed; they are not what the system *should* have produced. The post-loop regression catches any residual divergence between gold_program-faithful patches and full multi-turn execution. For `turn_type=number` turns gold_program is empty/trivial; gold_answer + the gold cell location serve the same role.
- **Cached diagnosis on retry**: by user choice, the router is not re-called between retries within a case. Misattributed cases (router fingered the wrong sub-agent) cannot be recovered by the retry loop; they will simply exhaust `retry_n` and route to unresolved. The post-loop regression and the human review of unresolved cases is the safety net.
- **Rule conflicts**: two rules in the same store can contradict. `supersedes` makes conflicts explicit but the agent doesn't emit it automatically — manual curation likely needed past ~20 rules per agent.
