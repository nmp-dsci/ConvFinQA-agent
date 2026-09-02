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
- **"Held out" means `data.loader.optimizer_split()`, not `train_report_ids`.** Both are 60/40 splits seeded 42, but they agree on only 78 of 120 conversations, and GEPA ran against the former. The 770-question scored set spans conversations the optimizer saw; the never-seen subset is 309 questions. Report `holdout_accuracy` alongside — never blended into — the overall figure.
- **Promotion requires a net-positive paired comparison** — more questions fixed than broken on the shared set — with every pass→fail flip listed and the exact McNemar p recorded on the verdict (flagged when not significant at α=0.05). Individual flips no longer veto on their own (rule changed 2026-09-02 at the owner's direction); they are evidence on the promotion record, and the diagnoser's first targets. Enforced in `tracking/comparator.py`, gated in CI by `tracking/gate.py`. The M2 teacher loop adds a second deliberate path (2026-09-02): a targeted challenger that changes ONE subagent promotes when that agent's first-fault count drops AND overall paired accuracy does not regress — recorded via `registry.promote(force=True, reason=...)` with the comparison attached, never silently. **Promotion evidence must come from the unseen test split** (protocol 2026-09-02): train runs optimise, test runs promote — both gate CLIs refuse `--promote` on train evidence. **Prompt versioning is per subagent** (M2.5): each agent has its own lineage in `registry.json → agent_prompts` (`t3.p3.r4.c3` compositions; content hash = truth, seq = human handle); bundles are lockfiles of four components; run names/params/traces carry the composition; `convfinqa-evalloop backfill-prompts` seeds it and `mirror-prompts` mirrors each agent into MLflow's prompt registry. Eval runs log a per-agent metric panel (`acc_triage_turn_type`, `acc_preprocess_skeleton`, `retriever_operand_recall`, `acc_calculator_exec`, `calculator_acc_given_full_recall` — `evalloop/stage_scores.py`, derived from gold, zero API calls) and `gate-targeted` judges the target on its deterministic metric, attribution as fallback.
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

## Eval loop (M1) & teacher (M2)

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
