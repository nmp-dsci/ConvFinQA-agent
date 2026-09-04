# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Full architecture reference, file layout, pipeline details, Logfire tracing, eval system, and all dev commands are in [AGENTS.md](AGENTS.md). Read it before making changes.**

## Quick Start

```bash
# Install dependencies
uv sync

# Start backend (port 8765)
uv run python -m uvicorn convfinqa.serving.app:create_app --factory --workers 1 --port 8765

# Start frontend (port 5173)
cd frontend && npm run dev

# Run tests
uv run pytest
```

## Key Invariants

- **`--workers 1` is required** for the backend — in-memory session state breaks with multiple workers.
- **`convfinqa/llm.py` is the only place a model may be constructed.** The demo gate and the retry/timeout policy live there; a model built anywhere else silently bypasses both. Backends expose `lm_mini()` / `lm_max()` factories, never module-level model objects — importing a module must never require an API key, because the demo container has none.
- **Nothing may build a model at import time.** This broke the deployment twice (`backends.pydantic`, then `backends.dspy`): a read-only route returned 500 purely because reading a dataset fact imported a module that constructed an LM. `tests/test_demo_mode.py::test_every_module_imports_without_a_key` pins it.
- **A campaign is the unit of optimisation work**: up to 5 experiments against one fixed gate split, with a target rotating off any subagent that failed twice in a row. Both caps are enforced in `evalloop/campaign.py`, not left to discipline. Each experiment changes **exactly one** subagent's prompt, so the diff between consecutive champions is one prompt and a champion move has a named cause.
- **The teacher and prompt writer run on the Claude Agent SDK (Opus 5, subscription)**; the four pipeline agents stay on `deepseek-v4-flash`. `llm.subscription_env()` is the only place the child environment is built, and it must **blank** `ANTHROPIC_API_KEY` rather than omit it — the SDK merges its `env` over `os.environ`, and `config.load_dotenv` puts the key there at import time. It also strips the `CLAUDE_CODE_*` session variables, or a loop driven from inside a Claude Code session bills that session. Pinned by `tests/test_llm.py`.
- **Attribution is gold-derived, not judged.** `stage_scores.first_fault()` walks triage → preprocess → retriever → calculator and returns the first failing check; that is what picks each experiment's target. The teacher is *told* the attribution, must justify it in a required `attribution_reason`, may dissent, and a dissent is recorded as `attribution_disputed` (they agreed on 17 of 30 cases when measured — the teacher over-called preprocess and under-called triage).
- **Targeting pools every train draw that ran the same prompt** (2026-09-04). One draw is ~50 first-wrong cases split four ways and the top two agents sit within a couple of cases, so a single draw's ranking is noise: three `v2` draws put preprocess at 18/26/14 against retriever's 15/15/16, and the draw that chose retriever for c01-e03 did so on a 16-vs-14 gap while the pooled evidence favoured preprocess 58–46. `ledger.fault_history(base_version)` accumulates per agent, keyed on **that agent's prompt hash** — legitimate because attribution walks the pipeline in order, so an upstream agent's fault count cannot be changed by a downstream rewrite. Contributing run sets therefore differ in length (an agent untouched since `v2` has three draws; one rewritten last cycle has none), which is why `pick_target` ranks on the fault **rate**, never raw totals — totals would favour whichever prompt had gone longest without a rewrite, exactly backwards. `ledger.merge_draw` folds the current draw in explicitly so a just-rewritten agent with no history is still eligible, and so the draw counts exactly once regardless of tracking-store read timing. Caveat: the denominator is first-wrong cases, so improving one agent mechanically raises every other agent's share — that is the right quantity for *targeting* (what share of remaining failures is this agent worth?) but it means rates are not comparable across compositions in absolute terms.
- **Train is resampled every cycle** from `pool − gate` via `splits.draw_train(seed=…)`, with the seed and drawn ids logged as a run artifact. Train passes early-stop at the first wrong turn (`--stop-at-first-wrong`); both flags are **refused on the gate split**, where the comparison is paired per question.
- **"Held out" means `data.loader.optimizer_split()`, not `train_report_ids`.** Both are 60/40 splits seeded 42, but they agree on only 78 of 120 conversations, and GEPA ran against the former. The 770-question scored set spans conversations the optimizer saw; the never-seen subset is 309 questions. Report `holdout_accuracy` alongside — never blended into — the overall figure.
- **Promotion requires net positive AND one-sided cluster-corrected McNemar p < 0.05** (campaign protocol, 2026-09-03, supersedes the net-positive-only rule). One-sided because the gate only ever promotes improvements, so half the rejection region is spent on a direction it never acts in; cluster-corrected (Durkalski, `Z = Σdₖ/√Σdₖ²` over conversations) because a report's turns share a history and usually an error, so four fixed turns in one report are one piece of evidence, not four. Every verdict also carries a cluster bootstrap CI on Δ. Implemented in `tracking/comparator.py` (`promotable_significant`, `cluster_p_one_sided`, `cluster_bootstrap_ci`) and applied by `evalloop/gate.py`. `promotable` (net positive alone) survives only for the legacy CI gate and for display. **The per-agent metric is no longer a second route to promotion** — under M2 it was, and that is how v3_1/v4/v5 were promoted on evidence whose intervals contained zero; all three were rolled back to v2 on 2026-09-03. **Promotion evidence must come from the unseen test split** (protocol 2026-09-02): train runs optimise, test runs promote — both gate CLIs refuse `--promote` on train evidence. **Prompt versioning is per subagent** (M2.5): each agent has its own lineage in `registry.json → agent_prompts` (`t3.p3.r4.c3` compositions; content hash = truth, seq = human handle); bundles are lockfiles of four components; run names/params/traces carry the composition; `convfinqa-evalloop backfill-prompts` seeds it and `mirror-prompts` mirrors each agent into MLflow's prompt registry. Eval runs log a per-agent metric panel (`acc_triage_turn_type`, `acc_preprocess_skeleton`, `retriever_operand_recall`, `acc_calculator_exec`, `calculator_acc_given_full_recall` — `evalloop/stage_scores.py`, derived from gold, zero API calls) and `gate-targeted` judges the target on its deterministic metric, attribution as fallback.
- **The prompt writer sees the whole record for the prompt it is replacing** (2026-09-04). `ledger.diagnoses_for_agent` gathers every failure ever filed against the exact prompt text — keyed on the per-agent hash, not the bundle version, since `v2` and `v8` share a preprocess prompt but not a retriever prompt — and reaches the writer as `failures_same_prompt` beside `failures_this_run`. Every gate writes `flips.json` (the individual questions fixed and broken, with the answer before and after), so a rejected attempt reports *which* questions it broke rather than only how many; nearly every c01 rejection was net-positive-but-not-significant, i.e. collateral damage, which counts alone cannot be written against. `convfinqa-evalloop backfill-flips` repairs gate runs recorded before this existed and **refuses any run whose recomputation disagrees with the stored verdict** — a wrong flip record would be read as history by every later writer.
- **The Agent SDK must be traced by hand** (2026-09-04). `mlflow.pydantic_ai.autolog()` covers the four pipeline agents because they run in this process; the teacher and prompt writer do not — `claude_agent_sdk` spawns the `claude` CLI as a **subprocess**, no in-process client is constructed, and there is no `mlflow.claude_agent_sdk` integration. The failure mode is silent: traces still appear, with a single `UNKNOWN` wrapper span and no prompt, reply, model, tokens or cost. `evalloop/sdk.py::run_structured` is the one chokepoint every SDK call passes through and opens an `LLM` span there (one **per attempt**, so a transient empty reply shows as a failed call then a successful one rather than a slow success). Any second call path to the SDK needs the same or it records nothing. `tracing.span()` takes `span_type` and its handle takes `inputs`/`outputs` — the trace UI leads with those, so a span that sets only attributes renders as an empty box.
- **Traces store prompts by reference, never raw text** (`evalloop/prompt_refs.py`). A system prompt is a constant repeated on every span of a run and the writer's prompt carries a whole subagent prompt plus its failure history, so dumping text put megabytes of duplication in the store and — at the old 20k cap — still recorded a *truncated* prompt, which is neither cheap nor faithful. Four ref kinds: `teacher_prompt` (module constant, by name), `agent_prompt` (the ledger's own `p2@4bc21f75` identity), `run_artifact` (text logged once on the run — `diagnose_memory.txt` is identical for every case in a pass, `writer_prompt.txt` is one per cycle), `diagnose_case` (a committed CSV row, rebuilt through `case_payload`). **Every ref carries the sha256 of what it stands for and `resolve` refuses on mismatch** — two kinds resolve against code, so without the check an edited `TEACHER_PROMPT` would be handed back as though it were the one that ran. The writer's user prompt is deliberately an artifact, not an id: `ledger_text` reads MLflow live, so that prompt is genuinely not reconstructible from ids. Read any of it back with `convfinqa-evalloop show-prompt --trace tr-…`.
- **Diagnosis runs concurrently** under the same semaphore the eval runner uses (`--concurrency`, default 8), and results are reassembled in case order so the JSONL artifact and the printed log never depend on completion order. It was ~70% of a cycle's wall clock as sequential calls, for no reason — no diagnosis reads another's result.
- **MLflow logging lives inside the runners**, not beside them. An operator who forgets to wrap a run produces an unrecorded result, and a history with silent gaps is worse than none.
- **MLflow tracing follows the same rule** (added 2026-09-02): `tracking/tracing.py` owns it. The evalloop runner always calls `tracing.enable()` — every LLM call lands as a span (run → report → question → named agent stage → `Agent.run`) linked to the MLflow run; serving opts in with `MLFLOW_TRACING=1`. `tracing.span()` is a free no-op until `enable()` succeeds, so imports stay cheap and the demo container needs no tracking server. Never set `MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false` to merge MLflow into Logfire's tracer provider — it crashes pydantic-ai runs; the two providers coexist by staying separate.
- **Vite proxy** (`frontend/vite.config.ts`'s `BACKEND_PREFIXES`) must list every backend path prefix (`/healthz`, `/reports`, `/sessions`, `/eval`, `/admin`, `/traces`, `/demo`). Missing entries cause silent HTML-404 failures in the browser.
- **`pred_program` column** must be present in every predictions CSV. `evaluate_cached` auto-detects its absence and forces a re-run.
- **Cached evaluations are committed**, not regenerated. `evaluation/predictions/` (prediction CSVs, HTML reports, joined CSVs), `evaluation/diagnostics/` (s7 `rules_*_<variant>.jsonl` + `rule_attempts_*_<variant>.jsonl` + diagnostic results), and `runs/<gepa_name>/` (GEPA artifacts including `optimized_runner.json`) are tracked in git so v1/v2 accuracy and GEPA outcomes reproduce across machines with `REUSE_CACHE=1` and no API calls. Do not add these directories to `.gitignore`. If you change pipeline behaviour and need fresh numbers, regenerate the CSVs and commit them — do not leave the working tree dirty or rely on `REUSE_CACHE=0` to mask stale state.
- **`archive/` holds retired experiment by-products** (GEPA iteration logs, DSPy/API parity CSVs, the abandoned s7 `v3_2` round), moved with `git mv` on 2026-08-31. Nothing reads it and `.dockerignore` excludes it; `archive/README.md` lists what moved and what stayed. Keep the GEPA prompts (`runs/*/optimized_runner.json`) and s7 stores (`rules_*_v3_1.jsonl`) where they are — code reads them.
- **`.dspy_cache/`** (~366 MB DSPy LM cache), **`mlruns/`** and **`.traces/`** stay gitignored — they are dev state. What ships is the committed export: `evaluation/mlflow_snapshot.json` + `evaluation/registry.json`, regenerated with `convfinqa-mlflow snapshot`. Sync `.dspy_cache/` via `rsync` to share warm caches; never commit it.
- **`DEMO_MODE` is baked into the Docker image**, not set in Terraform. Infrastructure must not be able to turn the public URL into a billable one. The demo container holds no API key at all.
- **The demo pack is rebuilt from committed CSVs**, never from fresh API calls: `uv run convfinqa-demo-pack`. Its events must stay identical to what `pipeline/runner.py::turn_events` emits — `serving/demo_pack/cli.py::events_from_row` is the other half of that contract.
- **The variant module `src/convfinqa/prompts/<variant>.py` (e.g. `v3_1.py`) is generated**, never hand-edited. The s7 harness writes it from the four `rules_<agent>_<variant>.jsonl` stores via `convfinqa.diagnosis.assembler`. To change it, edit the JSONL (or run the harness) and re-run `scripts/diagnose_failures.py --stage assemble`. The variant defaults to `settings.variant` (`v3_1`); override with `--variant`/`VARIANT`.
- **Diagnostic and predictions HTML use a sticky inspector panel**, not inline `<details>`. Each viewable cell renders a `view` button + adjacent hidden `<pre>`; clicking pops the content into the panel above the table. The shared mechanics (theme CSS, viewer panel/JS, `render_cell`, `render_page`) live in `convfinqa.reporting.html_report`; `evaluation/reporting.py` and `diagnosis/results_html.py` only supply their own columns, summary/pivot blocks, and filter JS. Change the look in one place — `html_report.py`.

## DSPy Pipeline Commands

```bash
# Baseline eval (no GEPA)
RUN_GEPA= uv run convfinqa-optimize

# GEPA smoke run (~30 min)
RUN_GEPA=1 GEPA_MODE=smoke uv run convfinqa-optimize

# GEPA real run (5–9 hr)
RUN_GEPA=1 GEPA_MODE=real uv run convfinqa-optimize

# Resume a prior GEPA run
RUN_GEPA=1 GEPA_MODE=real RESUME_GEPA=latest uv run convfinqa-optimize

# Re-score a prior optimized run (skips GEPA entirely)
RUN_GEPA=1 GEPA_NAME=gepa_real_<ts> uv run convfinqa-optimize

# Pydantic AI evaluation
uv run convfinqa-eval-api
```

## Tracking & demo commands

```bash
uv run convfinqa-mlflow status            # tracking config, aliases, versions
uv run convfinqa-mlflow compare v2 v3_1   # exit 1 when not promotable
uv run convfinqa-mlflow promote v3_1      # refused unless the comparator passes
uv run convfinqa-mlflow backfill          # rebuild history from committed artifacts
uv run convfinqa-mlflow snapshot          # export what the demo image reads
uv run python -m convfinqa.tracking.gate  # the CI eval-regression gate
uv run convfinqa-demo-pack --n 8          # rebuild the recorded demo pack

DEMO_MODE=1 uv run python -m uvicorn convfinqa.serving.app:create_app \
  --factory --workers 1 --port 8765       # runs with no API key
docker compose up demo                    # exactly what ships
./scripts/demo_smoke.sh http://localhost:8080
```

## Optimisation campaigns (the current loop)

```bash
# One experiment, end to end: train draw -> diagnose -> rewrite -> gate -> decide
EVAL_MANIFEST=eval_loop_v2 MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  uv run convfinqa-evalloop cycle --campaign c01
uv run convfinqa-evalloop cycle --campaign c01 --baseline-gate-csv <base.csv>  # reuse the baseline arm
uv run convfinqa-evalloop cycle --campaign c01 --no-promote                    # gate and record, don't move the champion
uv run convfinqa-evalloop campaign-status --campaign c01                       # used / promoted / blocked

# The published write-up, built from the tracking store and the registry
uv run convfinqa-evalloop story              # -> evaluation/story.json + docs/optimization/index.html
uv run python -m convfinqa.evalloop.story_check   # CI: fails when the page has gone stale

# Cutting a manifest by report count, as a superset of an existing one
uv run convfinqa-evalloop make-splits --name eval_loop_v2 --extend eval_loop_v1 \
  --train-reports 100 --test-reports 100
```

`EVAL_MANIFEST` selects the manifest for a whole session — set it once so every run, gate
and diagnosis agrees on what "the gate split" means. The **Campaigns page**
(`/admin/campaigns`, backend `GET /eval/campaigns`) and the published page both read
`evaluation/story.json`, so they cannot disagree; rebuild both with `story`.

## Eval loop (M1) & teacher (M2) — the underlying commands

```bash
uv run convfinqa-evalloop make-splits                  # committed split manifest (train/test/holdout)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  uv run convfinqa-evalloop run --split train --version v4 --n-reports 10
uv run convfinqa-evalloop run --split train --version v4 --n-questions 50  # cumulative-question budget, mutually exclusive with --n-reports
uv run convfinqa-evalloop gate --baseline-csv A.csv --candidate-csv B.csv \
  --baseline-version v3_1 --candidate-version v4 --promote   # M1 net-positive rule

# M2: diagnose first-wrong per report -> propose ONE-subagent challenger -> targeted gate
uv run convfinqa-evalloop diagnose --csv <run.csv> --version v3_1
uv run convfinqa-evalloop propose  --diagnoses <diagnoses.jsonl> --base-version v3_1 --new-version v4
uv run convfinqa-evalloop gate-targeted --target-agent retriever \
  --baseline-csv A.csv --candidate-csv B.csv --baseline-version v3_1 --candidate-version v4 --promote
  # --promote requires test-split evidence; diagnoses args optional (attribution fallback)

uv run convfinqa-evalloop backfill-prompts            # seed per-agent lineages from committed modules
uv run convfinqa-evalloop mirror-prompts --version v4 # per-agent prompts into MLflow's Prompts tab

# M2 trust: label ~30 cases, then score teacher-vs-human agreement (bar: κ ≥ 0.7)
uv run convfinqa-evalloop kappa --make --diagnoses evaluation/diagnostics/evalloop/diagnoses_*.jsonl
uv run convfinqa-evalloop kappa --labels <filled_sheet.csv>

# M3 release gate: opens the SEALED holdout once for the current champion.
# Never run casually — every opening is recorded and burns unseen-ness.
uv run convfinqa-evalloop release --i-know-this-opens-the-holdout
```

The **Dataset page** (`/admin/dataset`, backend `GET /eval/dataset?split=`) shows every
split's questions beside gold answer + gold program — where `gold_suspect` flags get
settled by a human. The teacher's failure taxonomy is frozen in `teacher.py::TEACHER_PROMPT`
(`new:<label>` marks gaps). `scripts/demo_smoke.sh` asserts served bundle == champion.

Teacher runs (diagnose/propose) log to the `convfinqa-optimization` MLflow experiment with
full tracing; eval runs stay in `convfinqa`. Prior diagnoses are read back from MLflow as
memory for the next teacher pass. Challenger prompt modules (`prompts/v4.py`, …) are
generated by `convfinqa-evalloop propose` — do not hand-edit.

## Prompt-Improvement Harness (s7)

Per-case **Diagnose → Route+Fix → Verify** loop over first-wrong-per-conversation cases in `pydantic_predictions_v2.csv`. Default `retry_n=1` (no retries; hard cap 3). Spec: `ai_specs/s7-prompt-optimisation.md`.

```bash
# Smoke: first failing case, single attempt, fresh stores
uv run python scripts/diagnose_failures.py --limit 1 --reset-rules --skip-regression

# Up to 2 retries per case
uv run python scripts/diagnose_failures.py --limit 10 --retry-n 3

# Re-assemble prompts/<variant>.py (default v3_1) from current rules JSONL
uv run python scripts/diagnose_failures.py --stage assemble
```

Outputs land under `evaluation/diagnostics/`, suffixed by the active `<variant>` (default `v3_1`): `diagnostic_results_<variant>.{csv,html}`, `case_results_<variant>.jsonl`, `rules_<agent>_<variant>.jsonl` × 4, `rule_attempts_<agent>_<variant>.jsonl` × 4, `unresolved_cases_<variant>.json`. The input predictions CSV is read from `evaluation/predictions/`.

## ConvFinQA Dataset Characteristics

Source: Chen et al. 2022 — "CONVFINQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering"

### Core Task

Given a financial report (text + table), answer a **sequence** of questions `{Q_0, Q_1, ..., Q_n}` where later questions **depend on previous questions and answers**. The target is to generate a **reasoning program** that can be executed to produce the final answer.

### Key Dataset Properties

- **3,892 conversations**, **14,115 questions** across **2,066 report pages**
- Average **3.67 questions per conversation**, avg question length **10.59 tokens**
- Split: 3,037 train / 421 dev / 434 test
- Two conversation types: **simple** (2,715) and **hybrid** (1,177)

### Question Types (turn_type)

1. **Number selection questions** (34.73%): Directly retrieve a value from the table or text (e.g., "what was the share-based compensation cost in 2010?" → `18.1`)
2. **Program questions** (65.27%): Require computation via a reasoning program
   - 1-step programs: 35.10%
   - 2-step programs: 25.41%
   - 3+ step programs: 4.75%

### Answer Types

- **59.18%** rely on supporting facts from the **table only**
- **25.56%** rely on facts from the **text only**
- **15.26%** rely on **both** text and table

### Calculation Operations (DSL)

The reasoning programs use a domain-specific language with these operations:

| Operation | Arguments | Output | Description |
|-----------|-----------|--------|-------------|
| `add` | number1, number2 | number | number1 + number2 |
| `subtract` | number1, number2 | number | number1 - number2 |
| `multiply` | number1, number2 | number | number1 × number2 |
| `divide` | number1, number2 | number | number1 / number2 |
| `exp` | number1, number2 | number | number1^number2 |
| `greater` | number1, number2 | bool | number1 > number2 |

- Calculation breakdown: ~18.80% additions, ~40.49% subtractions, ~6.92% multiplications, ~33.43% divisions
- Intermediate results are referenced with `#0`, `#1`, etc. (e.g., `add(18.1, -6.3), add(14.6, -5.2), subtract(#0, #1)`)

### Multi-Turn Dependency Chain

This is the hardest aspect of ConvFinQA. Over **60% of questions have dependencies on previous questions**:

- ~30% depend on 1 previous question
- ~28% depend on 2 previous questions
- ~19% depend on 3 previous questions
- ~12% depend on 4+ previous questions

**Implication for the agent**: The preprocessing agent must correctly resolve references to prior Q&A turns. Questions like "and what was **that** sum in 2009?" or "what percentage did **this change** represent?" require understanding what "that" and "this" refer to from conversation history.

### Conversation Types type_2

**Type I — Simple conversations**: A single multi-hop FinQA question decomposed into sequential single-step turns. Each reasoning step becomes one conversation turn.

**Type II — Hybrid conversations**: Two multi-hop FinQA questions about the same report are decomposed and concatenated. These have longer cross-question dependency chains and are significantly harder (Exe Acc: 52.38% vs 72.37% for simple).

### Key Challenges (from paper's error analysis)

1. **Long reasoning chains degrade accuracy**: Later turns in conversations are harder because errors compound — if any intermediate answer is wrong, subsequent answers will likely be wrong too.
2. **Domain knowledge gaps**: Missing financial domain knowledge leads to wrong value retrieval and incorrect mathematical operations.
3. **Context reference resolution**: Models struggle to correctly reference previous conversation context, especially in hybrid conversations where questions switch between different aspects of the same report.
4. **Program format matters**: Using natural math notation (e.g., `a1 + a2 → a3`) outperforms the original DSL format for LLM-based approaches.

### Evaluation Metrics

- **Execution Accuracy (Exe Acc)**: Whether the final executed result matches the gold answer
- **Program Accuracy (Prog Acc)**: Whether the generated program is equivalent to the gold program

Human expert performance: Exe Acc 89.44%, Prog Acc 86.34%


## Development Commands

```bash
# Install dependencies
uv sync

# Baseline eval only (no GEPA). RUN_GEPA defaults to "1" in the file —
# unset it explicitly to skip GEPA.
RUN_GEPA= uv run convfinqa-optimize

# GEPA smoke run (~30 min) — wiring check, NOT a transferable optimization
RUN_GEPA=1 GEPA_MODE=smoke uv run convfinqa-optimize

# GEPA real run (5–9 hr) — the real optimization
RUN_GEPA=1 GEPA_MODE=real uv run convfinqa-optimize

# Resume a prior GEPA run (must match mode/trainset/valset/num_preds)
RUN_GEPA=1 GEPA_MODE=real RESUME_GEPA=latest uv run convfinqa-optimize
RUN_GEPA=1 GEPA_MODE=real RESUME_GEPA=runs/gepa_real_<ts> uv run convfinqa-optimize

# Re-score a prior optimized run on the current test set (skips GEPA entirely;
# writes predictions.csv + predictions_joined.csv + accuracy slices)
RUN_GEPA=1 GEPA_NAME=gepa_smoke_20260429_204159 uv run convfinqa-optimize
RUN_GEPA=1 GEPA_NAME=gepa_real_<ts> uv run convfinqa-optimize

# Tests
uv run pytest
```

### Running on a VM

For long real runs, run GEPA unattended on a VM and do comparisons locally where the cache is warm. `runs/` is tracked in git so cloning the repo on a VM already gives you prior optimization artifacts — only `.dspy_cache/` needs an rsync.

```bash
# 1. On VM: clone (gets evaluation/ + runs/ for free), then rsync the LM cache
git clone <repo> ConvFinQA-agent && cd ConvFinQA-agent
rsync -av user@laptop:~/git/.../ConvFinQA-agent/.dspy_cache/ .dspy_cache/

# 2. Kick off the real run
RUN_GEPA=1 GEPA_MODE=real uv run convfinqa-optimize

# 3. When done: commit the new run dir back into git (preferred over rsync)
git add runs/gepa_real_<ts>/ && git commit -m "GEPA real run <ts>"
git push

# 4. Locally: pull and score (uses warm local cache)
git pull
RUN_GEPA= uv run convfinqa-optimize
RUN_GEPA=1 GEPA_NAME=gepa_smoke_20260429_204159 uv run convfinqa-optimize
RUN_GEPA=1 GEPA_NAME=gepa_real_<ts> uv run convfinqa-optimize
```
