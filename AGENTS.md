# AGENTS.md

Architecture reference, development commands, and coding guidelines for coding agents working in this repository.

## Current Status

The project uses a `src/convfinqa/` package layout. All code lives under `src/convfinqa/`, `scripts/`, or `tests/` — no Python modules remain at the repo root.

The research pipeline has grown into a production system: an LLM choke point (`llm.py`), MLflow-backed tracking/registry/promotion (`tracking/`), a FastAPI server split into routers (`serving/routes/`), a recorded keyless demo (`serving/demo_pack/`), a React operator console (`frontend/`), and a container + Terraform deploy path (`Dockerfile`, `infra/terraform/`). See the [README](README.md) for the full narrative (evaluation results, production surfaces, tracking/promotion contract, demo mode, deploy) — this file stays terse and reference-shaped.

`serving/app.py` was split into per-concern routers and is now 188 LOC. Three modules still exceed the 400-LOC target from the streamlining PRP: `diagnosis/harness.py` (~613 — the `run_case` async loop), `optimization/gepa.py` (~511), `backends/dspy.py` (~485). Shared HTML-report mechanics live in `convfinqa/reporting/html_report.py`, and the s7 CLI lives in `diagnosis/cli.py` with a thin script shim. See `ai_specs/s6-project-streamlining.md` for the planned splits.

## Project Overview

A multi-agent system that answers multi-turn questions about financial reports (text + table), built out into a production gen-AI system: tracked/versioned model bundles, a champion/challenger promotion contract, per-turn tracing, an automated prompt-research loop, and a keyless public demo alongside a live dev deployment. The main layers are:

1. **DSPy research pipeline** — `src/convfinqa/backends/dspy.py` (signatures, predictors, LM constructors) and `src/convfinqa/optimization/gepa.py` (GEPA training + re-scoring).
2. **Pydantic AI pipeline** — `src/convfinqa/backends/pydantic.py` (per-stage `Agent` instances), `src/convfinqa/pipeline/runner.py` (`run_turn`, `stream_turn`, `ConversationRunner`), `src/convfinqa/evaluation/runner.py` (cached evaluation harness).
3. **LLM choke point** — `src/convfinqa/llm.py`. The only place a model is constructed. Owns the demo gate (`guard_llm_call()`), the retry/timeout policy, and `lm_mini()`/`lm_max()` factories. Backends never hold module-level model objects, and agents are built lazily — importing any module must never require an API key.
4. **Tracking & registry** — `src/convfinqa/tracking/`. Bundle fingerprinting, MLflow logging + tracing (`tracing.py`), trace store, champion/challenger comparator, registry, per-agent prompt lineage (`prompt_ledger.py`), backfill, snapshot export, and the CI eval-regression gate.
5. **Eval loop (M1/M2/M2.5) & campaigns** — `src/convfinqa/evalloop/`. A second, self-improving promotion path over the committed train/test/holdout splits: `runner.py` (traced eval runs), `teacher.py` (first-wrong diagnosis + one-subagent challenger proposal, running on the Claude Agent SDK via `sdk.py`), `gate.py` (net-positive and significance-gated targeted gates), `stage_scores.py` (per-agent gold-derived metrics, gold-derived attribution via `evaluation/program_exec.py::bind_and_execute`), `ledger.py` (pooled fault history, Wilson-bound targeting), `campaign.py`/`cycle.py` (bounded multi-experiment campaigns), `prompt_refs.py` (trace prompt references, not raw text), `story.py`/`story_page.py`/`story_check.py` (the published write-up), `kappa.py` (teacher-vs-human agreement), `release.py` (sealed M3 holdout gate), `splits.py`, `cli.py` (`convfinqa-evalloop`).
6. **FastAPI server** — `convfinqa.serving.app:create_app`, routed through `serving/routes/` (`chat`, `evaluation` incl. `/eval/dataset`, `traces`, `admin`), backed by `serving/sessions.py` (in-memory session store), `serving/limits.py` (rate limiting), `serving/research.py` (s7/GEPA launch), and `serving/demo_pack/` (recorded replay for `DEMO_MODE`).
7. **React frontend** — `frontend/`. The portfolio's house UI in blue (root `DESIGN.md`; kit vendored in `frontend/src/kit/` from `nmp-central-ai/ui`): a sticky header with eight tabs — Overview (`/`), Filings & questions (`/filings`), Scoring (`/scoring`), Leaderboard (`/leaderboard`), Runs (`/runs` — every test pass, train behind `?train=1`; `/runs/legacy`, `/runs/traces`), Optimise (`/optimise` — the Story; `/optimise/campaigns`, `/optimise/runtimes`, `/optimise/fine-tuning`, `/optimise/research`), Agent (`/agent`, `/agent/readiness`, chat at `/agent/chat/*`) and Gold review (`/gold`). Every page is visible read-only in the public demo, gated by a real `<fieldset disabled>` and a server 501/403. The Console's `/admin/*` and `/chat` addresses redirect (`frontend/src/routes.tsx`).
8. **Container & infra** — `Dockerfile` (serves API + built SPA from one origin), `docker-compose.yml` (`demo` / `dev` toggle; `dev` joins the central platform's `nmp-central` network for MLflow), `infra/terraform/` (`bootstrap/` OIDC role, `demo/` ECR + App Runner + alarm), `.github/workflows/deploy-aws.yml` (keyless deploy chained on CI).

## File Layout

| Path | Purpose |
|------|---------|
| `src/convfinqa/config.py` | Settings and dotenv loading. Importing `convfinqa` loads settings first. |
| `src/convfinqa/data/loader.py` | Dataset loading, canonical `qa_data`, `_DOCS`, and `load_conv_examples_test()`. |
| `src/convfinqa/data/schemas.py` | Shared models: `ConvExample`, `QAPair`, `ConversationHistory`, `Document`, etc. |
| `src/convfinqa/pipeline/tools.py` | Calculator tools: `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`. |
| `src/convfinqa/pipeline/stages.py` | Pydantic output models for triage/preprocess/retriever/calculator stages. |
| `src/convfinqa/pipeline/wire_format.py` | DSPy ChatAdapter-compatible input rendering. Do not change format casually. |
| `src/convfinqa/pipeline/runner.py` | `run_turn`, `stream_turn`, `ConversationRunner` — framework-agnostic orchestration. |
| `src/convfinqa/pipeline/prompts_loader.py` | Prompt resolution: optimized artifact loading, deep-merge overlays. |
| `src/convfinqa/backends/pydantic.py` | Pydantic AI agents built lazily via `make_agents()`/`default_agents()`, using `llm.lm_mini()`/`llm.lm_max()` (`settings.lm_max_model`, default `deepseek-v4-pro`) rather than module-level model objects. |
| `src/convfinqa/backends/dspy.py` | DSPy signatures, predictors, LM constructors. Sets `LITELLM_MERGE_REASONING_CONTENT_IN_CHOICES` at import time. |
| `src/convfinqa/optimization/gepa.py` | GEPA training, resume, re-scoring. Sole prompt-optimisation path. |
| `src/convfinqa/evaluation/` | Metrics/cache plus package import paths for evaluation, reporting, joining, and API evaluation. `program_exec.py` is the DSL evaluator (`bind_and_execute`) that binds a symbolic `pred_program`'s placeholders to the retriever's answers and executes it, used by `evalloop/stage_scores.py` for gold-derived attribution — returns `None` (not `False`) when a plan cannot be bound. |
| `src/convfinqa/prompts/` | Versioned prompts (`v1.py`, `v2.py`, `v3_1.py`) and `load/latest` helpers. The variant module `prompts/<variant>.py` (e.g. `v3_1.py`) is generated by the s7 harness from JSONL rule stores. |
| `src/convfinqa/diagnosis/` | s7 prompt-improvement harness (`models.py`, `loader.py`, `prompts.py`, `agents.py`, `verify.py`, `rules_store.py`, `assembler.py`, `harness.py`, `results_writer.py`, `results_html.py`, `aggregator.py`, `cli.py`). |
| `src/convfinqa/reporting/` | Shared HTML-report mechanics (`html_report.py`): theme CSS, sticky inspector panel + viewer JS, `render_cell`, `render_page`. Used by `evaluation/reporting.py` and `diagnosis/results_html.py`. |
| `src/convfinqa/llm.py` | **The single LLM choke point.** `guard_llm_call()` (demo gate), `_RetryTransport` (retry/timeout policy), `get_provider()`/`get_model()`, `lm_mini()`/`lm_max()` factories, `dspy_lm_kwargs()`. Nothing else may construct a model. |
| `src/convfinqa/tracking/` | `bundle.py` (fingerprint: prompts + GEPA overlay + model ids + dataset hash + code SHA), `mlflow_log.py`, `tracing.py` (MLflow trace spans: run → report → question → agent stage → `Agent.run`; `enable()`/`span()`), `span_trim.py` (`span_processors` hook that strips duplicated tool/message payload from autologged pipeline spans, ~74% smaller), `traces.py` (per-stage IO trace store), `comparator.py` (promotion contract: net-positive + one-sided cluster-corrected McNemar `promotable_significant`, cluster bootstrap CI), `registry.py` (champion/challenger aliases, per-agent prompt lineage in `agent_prompts`, append-only history), `prompt_ledger.py` (per-agent prompt versioning: content-hash identity, `t3.p3.r4.c3` compositions), `backfill.py`, `snapshot.py` (demo image export), `gate.py` (CI eval-regression gate), `cost.py` (token/cost accounting), `cli.py` (`convfinqa-mlflow`). |
| `src/convfinqa/evalloop/` | `runner.py` (traced eval runs over a split), `teacher.py` (first-wrong-per-report diagnosis + one-subagent challenger proposal on the Claude Agent SDK, `TEACHER_PROMPT` taxonomy, `resolve_ambiguous` adjudication), `sdk.py` (the one chokepoint every Agent SDK call passes through — traces each attempt, requires a prompt ref), `prompt_refs.py` (ref kinds for traced prompts: `teacher_prompt`/`agent_prompt`/`run_artifact`/`diagnose_case`/`adjudicate_case`, each carrying a sha256 `resolve()` checks), `tools.py` (read-only MLflow tools exposed to the teacher as an in-process MCP server), `ledger.py` (pooled per-agent fault history via `fault_history()`, `flips.json`), `gate.py` (M1 net-positive gate + M2 `gate-targeted`, both requiring `promotable_significant`), `stage_scores.py` (per-agent gold-derived metric panel; `first_fault()` attribution via `evaluation/program_exec.py`), `campaign.py` (bounded campaign: 5 experiments, Wilson-bound `pick_target`, rotate target after 2 consecutive rejections; for the SDK arm `pick_target_class` over failure classes and `single_area_mode` instead of a block; `history` reads the gates ledger first), `cycle.py` (one campaign experiment end to end; `run_cycle(runtime="agent_sdk")` → `run_sdk_cycle`), `sdk_gate.py` / `sdk_teacher.py` / `ledgers.py` (the s10 Agent SDK arm — see the rows above), `story.py`/`story_page.py`/`story_check.py` (`evaluation/story.json` + `docs/optimization/index.html`, CI staleness check), `kappa.py` (teacher-vs-human agreement), `release.py` (M3 sealed holdout gate), `splits.py` (train/test/holdout manifest, `draw_train`), `cli.py` (`convfinqa-evalloop`). |
| `src/convfinqa/backends/agent_sdk.py` | The s10 single-session runtime (`runtime="agent_sdk"`): one Claude Agent SDK session per conversation with the six calculator functions as its only tools, reporting its triage/preprocess/retriever/calculator trail so the same panel and attribution apply. Same `run_conversation` signature as `ConversationRunner`. Lazy SDK import; nothing built at import. One of exactly two modules allowed to construct `ClaudeSDKClient` (with `evalloop/sdk.py`). |
| `src/convfinqa/prompts/sdk_v*.py` | The single-session prompt lineage (`SDK_PROMPT`): `sdk_v1` distilled from a bundle by `convfinqa-evalloop sdk-distil`, later versions generated by the SDK teacher. Never hand-edited. `prompts.load_sdk` / `sdk_versions` / `latest_sdk` / `is_sdk_version`. Registered under `registry.json → sdk_prompts` (`s1, s2, …`, hash = truth) by `tracking/prompt_ledger.py::ensure_sdk`. |
| `src/convfinqa/evalloop/ledgers.py` | The three append-only ledgers under `evaluation/diagnostics/evalloop/` — `diagnoses.jsonl`, `rewrites.jsonl`, `gates.jsonl` — with a frozen `COLUMNS` tuple each, `append` (open-append-fsync, never rewrite), `load(name, runtime=, version=, campaign=)`, `trace(question_id=|edit_id=)` joins, row builders, `log_rows_to_run` (mirrors each appended batch onto its MLflow run as `ledger_rows.jsonl`), and `backfill_ledgers`. Both arms write through it; `ledger.py`'s MLflow searches are the fallback when the files are silent. |
| `src/convfinqa/evalloop/sdk_teacher.py` | The SDK arm's loop agents on Opus via `sdk.py::run_structured`: `diagnose_run` (stage-by-stage diagnosis of each first-wrong case, filed under a failure class, one diagnoses-ledger row each), `rank_classes` (Wilson-bound ranking pooled over SDK draws of the same prompt hash — never a pipeline draw), `propose_version` (one prompt, one tagged edit per class addressed, `max_areas` cap; one rewrites-ledger row per edit), `distil_prompt` (writes `sdk_v1` from a bundle's four prompts). |
| `src/convfinqa/evalloop/sdk_gate.py` | The SDK arm's gate: `gate_overall` (paired one-sided cluster-corrected McNemar on overall accuracy — there is no single target agent — with the per-stage panel beside it) and `log_gate_verdict` (MLflow gate run + gates-ledger row stamped `runtime=agent_sdk`; separate from `teacher.log_gate_verdict`, whose ledger row is hard-wired to `multi_agent`). |
| `src/convfinqa/serving/app.py` | Package FastAPI entry point (`create_app`, `app`); mounts `serving/routes/`. |
| `src/convfinqa/serving/routes/` | `chat.py` (turns, SSE streaming), `evaluation.py` (splits, eval runs, `/eval/*` incl. `/eval/dataset`), `traces.py` (`/traces/*`), `admin.py` (owner-token-gated promotion/research writes), `metrics.py` (`/metrics/production` — turn-level counts, latency/cost/accuracy, hourly series, per `serving`/`demo`/`eval` source group). |
| `src/convfinqa/error_codes.py` | Closed `ErrorCode` vocabulary (`llm_unavailable`, `not_available_demo`, `no_recording`, `rate_limited`, `timeout`, `unknown`) a failed turn is classified into, alongside its free-text message. |
| `src/convfinqa/serving/sessions.py` | In-memory session store. Requires `--workers 1`. |
| `src/convfinqa/serving/limits.py` | Global in-flight cap + per-IP rate window; rejects rather than queues. |
| `src/convfinqa/serving/models.py` | Shared Pydantic request/response models for the API. |
| `src/convfinqa/serving/research.py` | Launches s7 / GEPA smoke runs from the admin Research page (`/admin/research`) and streams progress. |
| `src/convfinqa/serving/evaldata.py` | Backs the admin Evaluations page (`/admin/evaluations`): splits, per-question gold vs. per-version answers. |
| `src/convfinqa/serving/demo_pack/` | `pack.json` (recorded conversations, rebuilt from committed prediction CSVs), `store.py`, `replay.py` (paced SSE replay), `cli.py` (`convfinqa-demo-pack`, and `events_from_row` — the other half of the `pipeline/runner.py::turn_events` contract). |
| `src/convfinqa/serving/cli.py` | Package Typer CLI implementation. |
| `scripts/` | Installed console script entry points. |
| `data/convfinqa_dataset.json` | Source dataset. |
| `evaluation/predictions/` | Cached prediction CSVs + HTML reports + joined CSVs (`pydantic_predictions_v{1,2,3_1}`). **Tracked in git** so v1/v2 accuracy reproduces across machines without re-running. Served by the API and consumed by `REUSE_CACHE`. The DSPy/API parity outputs are archived. |
| `evaluation/diagnostics/` | s7 harness stores: `rules_*`, `rule_attempts_*`, `case_results_*`, `diagnostic_results_*`, `unresolved_cases_*`. Default `RULES_DIR`. `evalloop/` subdir holds teacher `diagnoses_*.jsonl` and the kappa `labelling_sheet_30cases.csv`. |
| `evaluation/splits/` | Committed eval-loop split manifest (`eval_loop_v1.json`: train/test/holdout `report_id`s, seeded). **Tracked in git.** Written by `convfinqa-evalloop make-splits`. |
| `evaluation/predictions/evalloop/` | Eval-loop run CSVs (`evalloop-<split><N>-<version>·<composition>-<ts>.csv`). **Tracked in git.** These, not `pydantic_predictions_<version>.csv`, are the evidence behind evalloop-promoted champions. |
| `evaluation/registry.json` | Bundle registry: champion/challenger aliases, per-agent prompt lineage (`agent_prompts`), and append-only promotion history. **Tracked in git.** |
| `evaluation/mlflow_snapshot.json` | Exported MLflow run/experiment history, produced by `convfinqa-mlflow snapshot`. **Tracked in git**; baked into the demo image so the Experiments tab works with no tracking server. |
| `runs/` | GEPA optimization artifacts — the optimized prompt (`optimized_runner.json` / `dspy_optimized_runner.json`), `config.json`, stats/summary. **Tracked in git** so prior optimization results are usable on any clone. Iteration logs and `gepa_state.bin` are archived. |
| `archive/` | Retired experiment by-products (GEPA iteration logs, DSPy/API parity CSVs, the abandoned s7 `v3_2` round). Nothing reads it; `archive/README.md` lists what moved and what stayed. |
| `.dspy_cache/` | DSPy LM response cache (~366 MB). Gitignored; rsync between machines for warm scoring. |
| `mlruns/`, `.mlflow/`, `.traces/` | Archived pre-2026-09-21 MLflow stores (read-only, not migrated; tracking now goes to the central server in `../nmp-central-ai`) and the local trace DB. Gitignored — the committed snapshot/registry is what ships. |
| `evaluation/traces_snapshot.jsonl.gz` | The trace store, exported verbatim (8.4k turns, 6 MB gz) by `convfinqa-mlflow traces-snapshot`. **Tracked in git**; seeds a store with no turns of its own, so the demo container's Traces page and production metrics show the recorded runs rather than only what it has replayed. |
| `infra/terraform/bootstrap/` | Run once by hand: the GitHub OIDC deploy role. Not applied by CI. |
| `infra/terraform/demo/` | ECR + App Runner + a 5xx alarm. Reconciled by `deploy-aws.yml` after each push to `main`. |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | The demo image (`DEMO_MODE` baked in, not set via Terraform), and the local `demo`/`dev` toggle (`dev` reaches the central MLflow over the external `nmp-central` network; `make -C ../nmp-central-ai up`). Every committed artifact a read-only route opens must be `COPY`d — a missing one empties a page rather than failing anything; `tests/test_demo_image.py` pins the list and `scripts/demo_smoke.sh` asserts the payloads are not their empty states. |
| `.github/workflows/ci.yml`, `.github/workflows/deploy-aws.yml` | CI (lint, mypy, pytest, frontend checks, eval-regression gate, Docker build, `terraform fmt`/`validate`) and the keyless AWS deploy chained on CI passing. |
| `frontend/` | Vite + React + react-query + Zustand on the house UI kit (plain CSS on tokens, blue accent, IBM Plex serif/sans/mono, follows the OS theme): eight tabs from `src/app/nav.ts`, routes as data in `src/routes.tsx`, chat at `/agent/chat/*`. |
| `docs/optimization/` | The published campaign write-up (`index.html` + `story.json`) and the Agent SDK experiment page (`agent-sdk.html`, linked from the index), both built by `convfinqa-evalloop story` from the tracking store, the ledgers and `evaluation/registry.json`; `evalloop/story_check.py` fails CI when either has gone stale. |
| `tests/` | pytest suite (221 tests), including `test_demo_mode.py` (pins the no-model-at-import-time invariant), `test_tracking.py`, `test_evalloop.py`, `test_llm.py`, `test_limits.py`. |

## Four-Stage Pipeline

Both DSPy and Pydantic AI implementations use the same four-stage logic per turn:

1. **Triage** — Classifies the question as `turn_type in {number, program}` and `conv_type in {Type I, Type II}`.
2. **Preprocess** — Only runs for `program` turns. Produces `sub_questions` and a DSL `program`.
3. **Retriever** — Retrieves raw values from the document. For `number` turns, this is the final answer; for `program` turns it returns one `QAPair` per sub-question.
4. **Calculator** — Only runs for `program` turns. Executes the DSL using calculator tools and returns the final numeric answer.

`ConversationHistory` stores prior turns as `(question, answer, report_id)` and renders them with `as_text()` for prompts. Multi-turn reference resolution depends on this text format.

## Commands

Install dependencies:

```bash
uv sync
```

Run tests/lint/typecheck:

```bash
uv run pytest tests/ -q
uv run ruff check src/ scripts/ tests/
uv run mypy
```

`packages = ["convfinqa"]` (via `mypy_path = "src"`) is what makes this resolve real types — the old `packages = ["src"]` silently pointed at a non-package and let every `convfinqa.*` import fall through to `Any`.

Evaluate cached Pydantic prompt versions:

```bash
REUSE_CACHE=1 uv run convfinqa-eval
PROMPTS_VERSION=v2 uv run convfinqa-eval
REUSE_CACHE=0 uv run convfinqa-eval
```

Start the backend:

```bash
uv run python -m uvicorn convfinqa.serving.app:create_app --factory --workers 1 --port 8765
```

Critical: keep `--workers 1`. Session state and per-session locks are in memory — deliberately, since App Runner runs the demo at max-size 1; a shared store (e.g. Redis) behind `SessionStore` is a later swap, not a prerequisite.

Start the backend keyless, exactly like the demo deployment:

```bash
DEMO_MODE=1 uv run python -m uvicorn convfinqa.serving.app:create_app --factory --workers 1 --port 8765
```

Evaluate the running API:

```bash
uv run convfinqa-eval-api
PROMPTS_VERSION=v2 uv run convfinqa-eval-api
```

Tracking, registry and promotion:

```bash
make -C ../nmp-central-ai up              # central MLflow server; UI at http://localhost:5000
uv run convfinqa-mlflow status
uv run convfinqa-mlflow compare v2 v3_1   # exit 1 if not promotable
uv run convfinqa-mlflow promote v3_1      # refused unless the comparator passes
uv run convfinqa-mlflow backfill          # rebuild history from committed artifacts
uv run convfinqa-mlflow snapshot          # export what the demo image reads
uv run convfinqa-mlflow traces-snapshot   # export the trace store the demo image seeds from
uv run python -m convfinqa.tracking.gate  # the CI eval-regression gate
```

Eval loop (M1) and teacher (M2/M2.5) — see the "Eval loop" section below and
the README's §Eval loop (M1) & teacher (M2/M2.5) for the full command set
(`make-splits`, `run`, `gate`, `diagnose`, `propose`, `gate-targeted`,
`backfill-prompts`, `mirror-prompts`, `kappa`, `release`), all under
`uv run convfinqa-evalloop <subcommand>`. See the "Optimisation
campaigns" section and the README's §Campaigns for `cycle`, `campaign-status`,
and `story`.

Demo pack and container:

```bash
uv run convfinqa-demo-pack --n 8          # rebuild the recorded demo pack
docker compose up demo                    # exactly what ships: no keys, replayed chat
docker compose up dev                     # same image, live model, your key
./scripts/demo_smoke.sh http://localhost:8080
```

Run DSPy/GEPA paths through the console entry point:

```bash
RUN_GEPA= uv run convfinqa-optimize
RUN_GEPA=1 GEPA_MODE=smoke uv run convfinqa-optimize
RUN_GEPA=1 GEPA_MODE=real uv run convfinqa-optimize
RUN_GEPA=1 GEPA_MODE=real RESUME_GEPA=latest uv run convfinqa-optimize
RUN_GEPA=1 GEPA_NAME=gepa_real_<ts> uv run convfinqa-optimize
```

Run the prompt-improvement harness (s7):

```bash
uv run python scripts/diagnose_failures.py --limit 1 --reset-rules --skip-regression
uv run python scripts/diagnose_failures.py --retry-n 3 --limit 10
uv run python scripts/diagnose_failures.py --stage assemble
```

Per-case **Diagnose → Route+Fix → Verify** loop over first-wrong-per-conversation cases. Five `LM_MAX` agents (one router + four specialists), default model `deepseek-v4-pro` (`LM_MAX_MODEL`), using `pydantic_ai.output.PromptedOutput` because DeepSeek reasoning models don't support tool-based structured output. Full spec: `ai_specs/s7-prompt-optimisation.md`.

Frontend:

```bash
cd frontend
npm run dev
```

### The confidence judge (s12)

A Haiku 4.5 judge reads each served agent_sdk turn's trace and bands it: `high` means
the trail verifies, `low` means it does not. The band is **advisory** by default — the
answer is shown with the judge's caution beside it — because on the sealed split it did
not earn the right to gate (s13); `JUDGE_MODE=gate` withholds instead and stays tested.
Trained by diagnose-with-gold → distil-without-gold on a 50/50 optimise split, scored at
natural prevalence on a calibrate split, tested once on the gate split. Full invariants
in "The confidence judge" below.

```bash
uv run convfinqa-evalloop judge-dataset --optimise-csv A.csv --calibrate-csv B.csv --test-csv gate.csv
TEACHER_MODEL=claude-sonnet-5 uv run convfinqa-evalloop judge-diagnose --split optimise
TEACHER_MODEL=claude-sonnet-5 uv run convfinqa-evalloop judge-distil --new-version judge_j1
uv run convfinqa-evalloop judge-score --version judge_j1 --split calibrate
uv run convfinqa-evalloop judge-gate --baseline-scores a.csv --candidate-scores b.csv \
  --baseline-version judge_j1 --candidate-version judge_j2 --promote
# Serving: SERVING_RUNTIME=agent_sdk (default) + JUDGE_ENABLED=1 (default); pipeline
# via SERVING_RUNTIME=pipeline. /healthz reports runtime, champion, champion_agent
# (model + effort the champion runs at), pipeline_champion, judge_champion.
```

### Agent SDK experiment (s10)

The single-session arm runs the same loop behind `--runtime agent_sdk` with
`sdk_vN` versions; a version from the other arm is refused at parse time.

```bash
# Once: distil the first single-session prompt from the pipeline champion's four
uv run convfinqa-evalloop sdk-distil --source-version v8 --new-version sdk_v1

# Eval one sdk prompt on a split (run name sdk-evalloop-<split><N>-sdk_vN·sN-<model slug>-<stamp>)
uv run convfinqa-evalloop run --split test --version sdk_v1 --runtime agent_sdk --concurrency 4

# The same prompt on another model: a scoring pass, all else equal, no optimisation.
# The slug lands in the run name, the id in the `sdk_model` param and tag; the story pairs
# it against the reference model's run (`sdk_model_comparison`) without touching the arm.
uv run convfinqa-evalloop run --split test --version sdk_v1 --runtime agent_sdk \
  --sdk-model claude-haiku-4-5-20251001

# Finish a pass the CLI refused part way through (see "Refusals" below)
uv run convfinqa-evalloop run --split test --version sdk_v1 --runtime agent_sdk \
  --resume-from evaluation/predictions/evalloop/<partial>.csv

# One SDK experiment: draw -> diagnose -> rank classes -> rewrite -> gate -> decide
EVAL_MANIFEST=eval_loop_v2 uv run convfinqa-evalloop cycle --campaign s01 --runtime agent_sdk
uv run convfinqa-evalloop cycle --campaign s01 --runtime agent_sdk --baseline-gate-csv <sdk_test.csv>
uv run convfinqa-evalloop campaign-status --campaign s01        # runtime, targets (failure classes), single_area_mode

# The underlying commands
uv run convfinqa-evalloop diagnose --csv <sdk_run.csv> --version sdk_v1 --runtime agent_sdk
uv run convfinqa-evalloop propose --diagnoses <d.jsonl> --base-version sdk_v1 --new-version sdk_v2 \
  --runtime agent_sdk [--max-areas 1]
uv run convfinqa-evalloop gate-targeted --runtime agent_sdk --target-class <label> \
  --baseline-csv A.csv --candidate-csv B.csv --baseline-version sdk_v1 --candidate-version sdk_v2 --promote
  # judges overall accuracy; --promote moves the champion's prompt (registry.promote_sdk),
  # keeping its model and effort; refused on train evidence

# The ledgers
uv run convfinqa-evalloop backfill-ledgers [--no-mlflow]      # seed the three files from per-run diagnoses + MLflow
uv run convfinqa-evalloop ledger-trace --question-id <id>     # diagnoses -> rewrites -> gates for one case
uv run convfinqa-evalloop ledger-trace --edit-id <id>         # one edit, the cases behind it, its verdicts
uv run convfinqa-evalloop story                               # also renders docs/optimization/agent-sdk.html
```

`story.json` carries `champion` + `champion_agent` (the universal champion, 2026-10-15),
`pipeline_champion`, `sdk_champion`, `sdk_campaigns` (targets are failure
classes) and `runtime_comparison` (`pipeline` / `agent_sdk` arms on the gate
split + the cross-runtime `gate` row, every field `None` until an SDK run
exists); `GET /eval/campaigns` serves them as additive keys.

**Refusals: a rate-limited turn is unscored, never wrong** (2026-09-05). The
`claude` CLI answers a spent account with prose (`You've hit your session limit
· resets 5:40pm`, `Credit balance is too low`), which is *no* answer — a live
349-question pass scored 176 of them as wrong and reported 44.4%.
`backends/agent_sdk.py::RATE_LIMIT_MARKERS` matches those texts on any reply
that did not arrive as a structured object and raises `SdkRateLimitError`,
which is **never retried** (the correction buys the same refusal) and **aborts
the conversation** (the session is spent). Three consequences, all pinned by
`tests/test_sdk_resume.py`:

- **Unscored, not wrong.** Those rows carry `unscored=True` (trailing column),
  empty `pred_answer` and `error` prefixed `rate_limited: `; accuracy,
  `n_wrong`, `program_accuracy`, the per-agent panel and the `accuracy_by_*`
  slices are all computed over the scored subset, so an unattempted turn is
  absent from the numerator *and* the denominator.
- **Incomplete passes cannot be gated.** Metrics `n_scored` / `n_unscored` /
  `n_rate_limited` / `complete`, MLflow tag `incomplete=true` and param
  `unscored_rows`, `complete` + `n_unscored` in the summary, and a loud closing
  line. `gate.load_run_csv` — the one door `gate.py` and `sdk_gate.py` both
  come through — raises `IncompleteRunError` naming the file and the count.
  There is no override. A CSV predating the column loads unchanged.
- **A conversation is reused whole or re-run whole.** `run --resume-from <csv>`
  copies every conversation the prior CSV answered completely through verbatim
  (original `run_id`/`trace_id`, plus `resumed_from_run_id`) and re-runs the
  rest from turn 0 — a session's later turns depend on its earlier ones, so a
  half-finished conversation is never stitched. Split, version, runtime and the
  report set are checked first; params `resumed_from` /
  `n_reused_conversations` and metric `n_reused_questions` are logged, and the
  panel is re-scored over the whole frame.

## Validation Baseline

The canonical cached evaluator smoke is:

```bash
REUSE_CACHE=1 uv run convfinqa-eval
```

Expected overall results (770-question scored set — **not** held out; see below):

- `v1`: `73.0%` (`562/770`)
- `v2`: `77.1%` (`594/770`) — champion
- `v3_1`: `76.2%` (`587/770`)

These numbers come from the `evaluation/predictions/pydantic_predictions_v{1,2,3_1}.csv` files committed in-repo. Anyone cloning the repo can reproduce them with `REUSE_CACHE=1` (no API calls). To force a true re-evaluation, set `REUSE_CACHE=0` — but prefer reusing the committed cache so accuracy figures stay comparable across machines and over time.

**"Held out" means `data.loader.optimizer_split()`, not `train_report_ids`** — GEPA trained on 120 of the 200 scored conversations, so the 770-question figure above mixes seen and unseen data. The genuinely never-seen subset is 309 questions; report `holdout_accuracy` alongside it, never blended into the overall number: `v1` 72.8%, `v2` 77.7% (champion), `v3_1` 73.5%. See the README's "A note on what held out means here" for the full derivation.

Server smoke:

```bash
uv run python -m uvicorn convfinqa.serving.app:create_app --factory --workers 1 --port 8765
curl http://127.0.0.1:8765/healthz
curl http://127.0.0.1:8765/eval/runs
```

Expected `/eval/runs` (auto-discovered from `prompts/`, grows as variants are added): `["v1","v2","v3_1"]`.

## Reproducibility Policy

The repo treats cached evaluation outputs as first-class artifacts, not throwaway runtime state:

- **`evaluation/predictions/`** — committed. Holds `pydantic_predictions_v{1,2,3_1}.csv`, their joined CSVs, and HTML reports. Re-running with `REUSE_CACHE=1` rehydrates results from these files. (The s3–s5 DSPy/API parity outputs — `dspy_predictions_*`, `api_predictions_*`, `parity_report_*`, `model_accuracy_comparison_*` — are under `archive/evaluation/predictions/`.)
- **`evaluation/diagnostics/`** — committed. Holds the s7 source-of-truth stores `rules_<agent>_<variant>.jsonl` + `rule_attempts_<agent>_<variant>.jsonl` (currently `_v3_1`) plus `case_results_*`, `diagnostic_results_*`, `unresolved_cases_*`. The abandoned `_v3_2` round is under `archive/evaluation/diagnostics/`.
- **`runs/<gepa_name>/`** — committed. Holds the optimized prompt (`optimized_runner.json` / `dspy_optimized_runner.json`), `config.json` and stats from each GEPA optimization. Lets anyone re-score a prior run with `RUN_GEPA=1 GEPA_NAME=<name> uv run convfinqa-optimize`. The per-iteration logs and `gepa_state.bin` are under `archive/runs/<gepa_name>/`; restore `gepa_state.bin` with `git mv` before using `RESUME_GEPA` on a run.
- **`evaluation/registry.json`, `evaluation/mlflow_snapshot.json`** — committed. Bundle registry (promotion history) and exported MLflow run/experiment history, regenerated with `convfinqa-mlflow backfill` / `snapshot`. Baked into the demo image so the Experiments tab works with no tracking server.
- **`.dspy_cache/`** — gitignored (~366 MB). Local LM response cache. Sync between machines via `rsync -av .dspy_cache/ user@host:~/ConvFinQA-agent/.dspy_cache/` rather than committing.
- **`mlruns/`, `.mlflow/`, `.traces/`** — gitignored. The first two are the archived pre-2026-09-21 MLflow stores (read-only, not migrated; new runs go to the central server in `../nmp-central-ai`), the third the local trace DB; dev state, not shipped. The trace store's committed export is `evaluation/traces_snapshot.jsonl.gz` (`convfinqa-mlflow traces-snapshot`), which seeds an empty store — a dev machine with its own history is never touched.

Rule of thumb: if regenerating it costs an API call, commit it. If it can be rebuilt locally without network, leave it ignored.

## Frontend Proxy Invariant

Every backend path prefix must be listed in `BACKEND_PREFIXES` in `frontend/vite.config.ts`, which feeds both `server.proxy` and `preview.proxy`:

```ts
const BACKEND_PREFIXES = ['/healthz', '/reports', '/sessions', '/eval', '/admin', '/traces', '/demo', '/metrics']
const proxy = Object.fromEntries(BACKEND_PREFIXES.map((p) => [p, API_BASE]))
```

If a new backend route prefix is added, add it to `BACKEND_PREFIXES` or the browser can receive HTML 404s instead of JSON.

`/admin` is an API prefix and, until s16, was also the UI's page prefix; the old `/admin/*` page addresses now redirect client-side, so the proxy must still hand document requests (`sec-fetch-dest: document` / `accept: text/html`) back to Vite's client router instead of forwarding them to FastAPI — otherwise a browser navigating to `/admin/evaluations` gets proxied, matches no backend route, and falls through to a built `index.html` that references dist asset hashes the dev server doesn't serve. `bypassDocumentRequests()` in `vite.config.ts` is what does this.

## Environment Variables

| Variable | Required | Description |
|---|---:|---|
| `DEEPSEEK_API_KEY` | Yes for any LLM call | DeepSeek via LiteLLM/OpenAI-compatible provider. Optional at boot — only `Settings.require_deepseek_api_key()`, called from `llm.py`, demands it, and only when a call is actually about to happen. The demo container has none. |
| `TEACHER_OAUTH_TOKEN` | No | Explicit override for the teacher/prompt-writer's Claude Agent SDK login, under its own name rather than `CLAUDE_CODE_OAUTH_TOKEN` — a general-purpose token in a shared dotfile would otherwise silently replace the CLI's working subscription login. Set in `llm.subscription_env()`. |
| `LOGFIRE_TOKEN` | No | Enables remote tracing. |
| `LITELLM_MERGE_REASONING_CONTENT_IN_CHOICES` | For DSPy | Set automatically by `convfinqa.backends.dspy` at import time for DeepSeek reasoning output compatibility. |
| `PROMPTS_VERSION` | No | Pin prompt version, otherwise latest prompt module is used. |
| `PROMPTS_OVERLAY_PATH` | No | JSON overlay path used by prompt optimisation harness. |
| `REUSE_CACHE` | No | Defaults on; set `0` to force evaluation reruns. |
| `LM_MAX_MODEL` | No | DeepSeek model used by the s7 diagnosis agents and the GEPA/dspy backend. Default `"deepseek-v4-pro"`. |
| `RULES_DIR` | No | Directory for s7 rules/attempts JSONL stores. Default `evaluation/diagnostics/`. |
| `RETRY_N` | No | s7 per-case total attempts cap (1..3). Default `1` (no retries). |
| `MAX_PRIOR_ATTEMPTS_IN_PAYLOAD` | No | Cap on prior `rule_attempts` surfaced to a specialist (default 50). |
| `DEMO_MODE` | No | Baked into the demo Docker image, not set via Terraform. When set, `llm.py::guard_llm_call()` refuses every LLM call and chat is served from `serving/demo_pack/` instead. Default `false`. |
| `OWNER_TOKEN` | For admin writes | Gates promotion and research-launch routes (`serving/routes/admin.py`); compared timing-safely. Unset means admin writes are refused outright, not left open. |
| `TRUSTED_PROXY` | No | Whether `X-Forwarded-For` is trusted for per-client rate limiting (true behind App Runner, which always sets it). Default `true`. |
| `MAX_INFLIGHT_TURNS` | No | Global in-flight turn cap; rejects rather than queues. Default `4`. |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | No | Per-IP rate limit. Default `30` requests / `60` seconds. |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_ATTEMPTS` | No | Per-call timeout and retry cap, enforced in `llm.py`. Default `120.0` / `4`. |
| `MLFLOW_TRACKING_URI` | No | Default `http://localhost:5000`, the central platform server (`make -C ../nmp-central-ai up`; `http://mlflow:5000` from inside compose). The demo image never contacts it and reads the committed snapshot instead. Point it at `sqlite:///mlruns/mlflow.db` only to read the archived pre-2026-09-21 store. |
| `MLFLOW_EXPERIMENT` / `REGISTERED_MODEL_NAME` | No | Default `"convfinqa"` / `"convfinqa-pipeline"`. |
| `MLFLOW_TRACING` | No | Opt serving into per-call MLflow trace spans via `tracking/tracing.py`. The evalloop runner always traces regardless of this flag. Default `false`. |
| `TRACE_CAPTURE_ENABLED` | No | Persist per-stage IO for every serving turn. Default `true`; off in tests. |

Never commit `.env` files.

## Development Notes

- All imports go through `convfinqa.*`; no legacy root modules exist.
- Do not change `render_chat_inputs()` output without an evaluation smoke; prompts were tuned against that exact wire format.
- Do not change backend route paths or SSE event shapes without updating frontend types/tests.
- Do not hand-edit the generated variant module `src/convfinqa/prompts/<variant>.py` (e.g. `v3_1.py`) — it is generated by `convfinqa.diagnosis.assembler` from the `rules_<agent>_<variant>.jsonl` stores. Edit the JSONL or re-run the harness, then `--stage assemble`. The output variant defaults to `settings.variant` (`v3_1`); override with `--variant v3_2` / `VARIANT=v3_2`.
- Predictions HTML and diagnostic HTML use a sticky inspector-panel viewer (button per JSON/long-text cell pops content into a panel above the table) instead of inline `<details>` expansion. The shared mechanics live in `convfinqa/reporting/html_report.py`; change the theme/viewer there once rather than editing `evaluation/reporting.py` and `diagnosis/results_html.py` in parallel.
- `diagnosis/harness.py`, `optimization/gepa.py`, and `backends/dspy.py` are still over the 400-LOC target — split before adding new features inside them. `harness.py`'s `run_case` is the prime candidate but is the untested core async loop, so decompose it as a focused, separately-verified change. `serving/app.py` was already split into `serving/routes/` and is no longer on this list.
- **Never construct a model outside `convfinqa/llm.py`.** Backends expose `lm_mini()`/`lm_max()` factories, never module-level model objects, and agents are built lazily. This is what makes the demo gate real and lets the keyless demo container import the whole package; two prior deployments broke (`backends.pydantic`, then `backends.dspy`) because a module built an LM at import time. `tests/test_demo_mode.py::test_every_module_imports_without_a_key` pins the invariant.
- **Promotion requires a net-positive paired comparison** — strictly more questions fixed than broken on the shared question set. Pass→fail flips no longer veto on their own (rule changed 2026-09-02): each one is listed on the verdict together with the exact McNemar p over the discordant pairs, flagged when the sample cannot support significance (α=0.05), so a small-sample promotion is recorded as what it is. Enforced in `tracking/comparator.py`, gated in CI by `tracking/gate.py`.
- MLflow logging lives *inside* the eval/GEPA/s7 runners, not beside them, so a run cannot happen without being recorded.
- MLflow tracing (2026-09-02): `tracking/tracing.py` — evalloop always traces (autolog + report/question spans, run-linked, tagged `model_version_id`/`split`/`run_name`); serving opts in via `MLFLOW_TRACING=1`. `tracing.span()` no-ops until `enable()`. Do not merge providers with `MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false` — crashes pydantic-ai under Logfire.
- "Held out" means `data.loader.optimizer_split()`, not `train_report_ids` — see §Validation Baseline above.

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
- **Attribution is gold-derived, and the one judgement gold cannot make is asked as a binary question.** `stage_scores.first_fault()` walks triage → preprocess → retriever → calculator and returns the first failing check. Rewritten 2026-09-04 after hand-attributing seven failures and disagreeing with the previous rule on five; re-scoring all twelve committed runs moved **37.4% of 554 first-wrong cases** and cut the retriever's share by 62% (152 → 58). Three things were wrong. (1) **Plan shape was judged before operand coverage**, so a skeleton mismatch short-circuited before the retriever check ran and preprocess won ties it should have lost. The order now follows the data dependency: `pred_program` is **symbolic** (`divide(A, B)`, placeholders bound to preprocess's own sub-questions in order), so the plan genuinely cannot be evaluated until coverage is known. (2) **Skeleton equality is not a valid test for a symbolic plan** — it punished correct plans shaped differently from gold and passed wrong plans that shared a shape. `evaluation/program_exec.py::bind_and_execute` replaces it: bind the placeholders to the retriever's answers, run the DSL, compare to gold. It returns `None`, never `False`, when the plan cannot be bound, because "unreadable" and "wrong" are different claims. (3) **Every case had to name an agent** — the fallback charged the calculator, so a bad gold label was billed to an innocent agent. `NON_AGENT = (gold_suspect, ambiguous, unscorable)` exists so that stops, and those verdicts are excluded from targeting's numerator *and* denominator (`ledger.fault_history` keys off `n_attributed`, falling back to the sum of fault counts, which is exact for pre-rewrite runs). The `gold_suspect` check only fires when the document text is actually held — with no text every operand reads as absent, and an unmatched report id would empty the fault counts. **`ambiguous` is the ~7% the record cannot settle**: a missing operand that *is* in the document, where the retriever declined a sub-question. `Double_BLK` (values sat in the table; the retriever reasoned itself out of them) and `Double_IPG` (preprocess asked for a quantity the report never states) leave byte-identical evidence with opposite correct verdicts, so `teacher.resolve_ambiguous` asks the SDK one binary question — *did any sub-question ask for this number?* — and never shows it the gold answer, because it is not attributing. Yes → retriever had its chance; no → preprocess. A failed adjudication leaves the case unresolved rather than guessed. Known limitation, pinned by a test: gold may spell out an operand the pipeline never needed because an earlier turn produced the value it feeds into (`Single_UNP/2013 q1`), and that reads as missing. The teacher is still *told* the attribution, must justify it in `attribution_reason`, may dissent, and a dissent is recorded as `attribution_disputed`. **`convfinqa-evalloop backfill-attribution` recomputes past diagnose runs from their committed `source_csv`** — required, because pooling counts from two different rules under one Wilson bound averages two measurements instead of enlarging one; a run whose CSV is missing is skipped, never estimated. Its guard compares each run's stored `attribution_rule` tag against `stage_scores.attribution_rule_id()` — a hash of the whole `stage_scores`/`program_exec`/`evaluation.metrics` module sources, not a hand-bumped constant, so it notices the rule changing underneath it rather than just whether a run was ever recomputed; `--force` recomputes regardless. Recomputation reads back any `adjudicated`/`derived_agent` verdicts already settled in the run's own `diagnoses.jsonl` and reuses them for cases the recompute marks `ambiguous`, so re-running it cannot discard a verdict a prior run already paid a model call to settle.
- **Targeting pools every train draw that ran the same prompt** (2026-09-04). One draw is ~50 first-wrong cases split four ways and the top two agents sit within a couple of cases, so a single draw's ranking is noise: three `v2` draws put preprocess at 18/26/14 against retriever's 15/15/16, and the draw that chose retriever for c01-e03 did so on a 16-vs-14 gap while the pooled evidence favoured preprocess 58–46. `ledger.fault_history(base_version)` accumulates per agent, keyed on **that agent's prompt hash** — legitimate because attribution walks the pipeline in order, so an upstream agent's fault count cannot be changed by a downstream rewrite. Contributing run sets therefore differ in length (an agent untouched since `v2` has three draws; one rewritten last cycle has none), which is why `pick_target` ranks on the **Wilson lower bound** of the fault rate (`ledger._score`), never raw totals and never the bare rate. Totals would favour whichever prompt had gone longest without a rewrite; the bare rate re-creates the noise problem pooling exists to solve, because the agents carry unequal evidence — c02-e01 picked the retriever at 18/45 = 40.0% over preprocess at 71/195 = 36.4%, the retriever's figure being one draw of a prompt rewritten the cycle before, and it gated at exactly +0.00pp. The Wilson bound needs no tuned threshold and stays positive whenever any fault is recorded, so a freshly rewritten agent is penalised for thin evidence but never unreachable. `ledger.merge_draw` folds the current draw in explicitly so a just-rewritten agent with no history is still eligible, and so the draw counts exactly once regardless of tracking-store read timing. Caveat: the denominator is first-wrong cases, so improving one agent mechanically raises every other agent's share — that is the right quantity for *targeting* (what share of remaining failures is this agent worth?) but it means rates are not comparable across compositions in absolute terms.
- **Train is resampled every cycle** from `pool − gate` via `splits.draw_train(seed=…)`, with the seed and drawn ids logged as a run artifact. Train passes early-stop at the first wrong turn (`--stop-at-first-wrong`); both flags are **refused on the gate split**, where the comparison is paired per question.
- **"Held out" means `data.loader.optimizer_split()`, not `train_report_ids`.** Both are 60/40 splits seeded 42, but they agree on only 78 of 120 conversations, and GEPA ran against the former. The 770-question scored set spans conversations the optimizer saw; the never-seen subset is 309 questions. Report `holdout_accuracy` alongside — never blended into — the overall figure.
- **Promotion requires net positive AND one-sided cluster-corrected McNemar p < 0.05** (campaign protocol, 2026-09-03, supersedes the net-positive-only rule). One-sided because the gate only ever promotes improvements, so half the rejection region is spent on a direction it never acts in; cluster-corrected (Durkalski, `Z = Σdₖ/√Σdₖ²` over conversations) because a report's turns share a history and usually an error, so four fixed turns in one report are one piece of evidence, not four. Every verdict also carries a cluster bootstrap CI on Δ. Implemented in `tracking/comparator.py` (`promotable_significant`, `cluster_p_one_sided`, `cluster_bootstrap_ci`) and applied by `evalloop/gate.py`. `promotable` (net positive alone) survives only for the legacy CI gate and for display. **The per-agent metric is no longer a second route to promotion** — under M2 it was, and that is how v3_1/v4/v5 were promoted on evidence whose intervals contained zero; all three were rolled back to v2 on 2026-09-03. **Promotion evidence must come from the unseen test split** (protocol 2026-09-02): train runs optimise, test runs promote — both gate CLIs refuse `--promote` on train evidence. **Prompt versioning is per subagent** (M2.5): each agent has its own lineage in `registry.json → agent_prompts` (`t3.p3.r4.c3` compositions; content hash = truth, seq = human handle); bundles are lockfiles of four components; run names/params/traces carry the composition; `convfinqa-evalloop backfill-prompts` seeds it and `mirror-prompts` mirrors each agent into MLflow's prompt registry. Eval runs log a per-agent metric panel (`acc_triage_turn_type`, `acc_preprocess_plan`, `acc_preprocess_skeleton`, `retriever_operand_recall`, `acc_calculator_exec`, `calculator_acc_given_full_recall` — `evalloop/stage_scores.py`, derived from gold, zero API calls). **Preprocess is targeted on `acc_preprocess_plan`, not the skeleton** (2026-09-05): the skeleton compares op lists, which is invalid for a symbolic plan for the same reason it was dropped from attribution, and across c03 it moved *against* the accuracy it was supposed to be evidence for — v11 gained 1.43pp while the skeleton fell 1.27pp, where the execution metric rose 0.81pp. `acc_preprocess_skeleton` stays in the panel for continuity with recorded runs but decides nothing. `_ensure_scored` re-scores a frame missing **any** `ROW_COLUMNS` entry, not just a sentinel: every committed CSV predates `preprocess_plan_ok`, so a one-column guard skipped scoring and the panel raised `KeyError` on exactly the runs it had to read and `gate-targeted` judges the target on its deterministic metric, attribution as fallback.
- **The prompt writer sees the whole record for the prompt it is replacing** (2026-09-04). `ledger.diagnoses_for_agent` gathers every failure ever filed against the exact prompt text — keyed on the per-agent hash, not the bundle version, since `v2` and `v8` share a preprocess prompt but not a retriever prompt — and reaches the writer as `failures_same_prompt` beside `failures_this_run`. Every gate writes `flips.json` (the individual questions fixed and broken, with the answer before and after), so a rejected attempt reports *which* questions it broke rather than only how many; nearly every c01 rejection was net-positive-but-not-significant, i.e. collateral damage, which counts alone cannot be written against. `convfinqa-evalloop backfill-flips` repairs gate runs recorded before this existed and **refuses any run whose recomputation disagrees with the stored verdict** — a wrong flip record would be read as history by every later writer.
- **The Agent SDK must be traced by hand** (2026-09-04). `mlflow.pydantic_ai.autolog()` covers the four pipeline agents because they run in this process; the teacher and prompt writer do not — `claude_agent_sdk` spawns the `claude` CLI as a **subprocess**, no in-process client is constructed, and there is no `mlflow.claude_agent_sdk` integration. The failure mode is silent: traces still appear, with a single `UNKNOWN` wrapper span and no prompt, reply, model, tokens or cost. `evalloop/sdk.py::run_structured` is the one chokepoint every SDK call passes through and opens an `LLM` span there (one **per attempt**, so a transient empty reply shows as a failed call then a successful one rather than a slow success). Any second call path to the SDK needs the same or it records nothing. `tracing.span()` takes `span_type` and its handle takes `inputs`/`outputs` — the trace UI leads with those, so a span that sets only attributes renders as an empty box.
- **Autologged pipeline spans are trimmed before storage** (`tracking/span_trim.py`, an MLflow `span_processors` hook installed by `tracing.enable()`). The pipeline's traces were 874 MB of an 880 MB store and almost none of it was information: `ToolManager.execute_tool_call` serialised pydantic-ai's whole toolset plus a second copy of the message history and the user prompt (86% of the payload); `Agent.run` wrote `_state` and `_new_messages_serialized` at ~112 KB each beside a 1.9 KB `output`; `InstrumentedModel.request` repeated the four agent system prompts on every call. Measured 73.6% smaller per span (tool-call −84.6%, Agent.run −79.4%, model request −50.8%) with the messages, tool arguments and outputs — the reasoning — untouched. Two traps: the duplicated `ctx.prompt`/`ctx.messages` were verified byte-identical to a model span in the same trace before being dropped (8/8 on a sample), and prompt matching must be on the **stripped** text, because pydantic-ai delivers instructions with the trailing newline removed and hashing the raw module constant matches nothing and fails *silently*. Under Pydantic AI 2.x (MLflow 3.17's autologger) a model span is named after the model class — `OpenAIChatModel.request`, not `InstrumentedModel.request` — and carries the instructions a second time as `model_request_parameters.instruction_parts`; the trimmer matches the `.request` suffix and substitutes both, and tool spans are now named after the tool and carry only the call. A trimmer keyed on the old name stops substituting with no error, so `test_an_autologged_run_stores_prompt_refs_not_prompt_text` drives a real autologged run into a throwaway store and reads the trace back. `MLFLOW_TRACE_FULL=1` disables it. Nothing enforces retention — both the MLflow store and `.traces/traces.db` are insert-only and grow forever.
- **Traces store prompts by reference, never raw text** (`evalloop/prompt_refs.py`). A system prompt is a constant repeated on every span of a run and the writer's prompt carries a whole subagent prompt plus its failure history, so dumping text put megabytes of duplication in the store and — at the old 20k cap — still recorded a *truncated* prompt, which is neither cheap nor faithful. Four ref kinds: `teacher_prompt` (module constant, by name), `agent_prompt` (the ledger's own `p2@4bc21f75` identity), `run_artifact` (text logged once on the run — `diagnose_memory.txt` is identical for every case in a pass, `writer_prompt.txt` is one per cycle), `diagnose_case` (a committed CSV row, rebuilt through `case_payload`). **Every ref carries the sha256 of what it stands for and `resolve` refuses on mismatch** — two kinds resolve against code, so without the check an edited `TEACHER_PROMPT` would be handed back as though it were the one that ran. The writer's user prompt is deliberately an artifact, not an id: `ledger_text` reads MLflow live, so that prompt is genuinely not reconstructible from ids. Read any of it back with `convfinqa-evalloop show-prompt --trace tr-…`. `refs` is a **required argument with no default** on `run_structured`: since spans no longer carry text, a call site that omitted it would record a prompt neither stored nor recoverable — worse than the dump this replaced — so mypy rejects it at the call site. `ClaudeSDKClient` may only be constructed in `evalloop/sdk.py`; a test pins both. Note this covers the Agent SDK only — the pydantic-ai pipeline traces are autologged with full text and are where the store's bulk actually is (~1.2 MB per conversation trace against ~0.4 KB for a teacher span).
- **Diagnosis runs concurrently** under the same semaphore the eval runner uses (`--concurrency`, default 8), and results are reassembled in case order so the JSONL artifact and the printed log never depend on completion order. It was ~70% of a cycle's wall clock as sequential calls, for no reason — no diagnosis reads another's result.
- **MLflow logging lives inside the runners**, not beside them. An operator who forgets to wrap a run produces an unrecorded result, and a history with silent gaps is worse than none.
- **Tracking goes to the central platform** (2026-09-21): `settings.mlflow_tracking_uri` defaults to `http://localhost:5000`, the `nmp-central-ai` server (`make -C ../nmp-central-ai up`; `MLFLOW_TRACKING_URI` overrides, `http://mlflow:5000` inside compose). This repo runs no MLflow of its own any more — the compose `mlflow` service is gone. Because every write in `mlflow_log` is best-effort-silent, the evalloop runner calls `mlflow_log.assert_reachable()` first and refuses to spend model calls against a server that does not answer `/health`; serving never does, so the demo stays server-free. `_mlflow()` also probes `/health` (2s) before touching the REST client: without it a stopped server cost every caller MLflow's retry ladder — **measured 247s per call** — which the demo container's `/admin/experiments` would have paid on every request, since its default URI is now a server it can never reach. Old stores (`mlruns/mlflow.db`, `.mlflow/`) are archived read-only, not migrated. **So the central store does not hold the pre-migration record** — the champion's own evidence included — and two readers must merge rather than replace: `story.collect()` unions the store's eval runs with the committed `mlflow_snapshot.json` (`_with_snapshot_runs`) and carries the committed campaign sections forward when the store has no gate runs (`campaign_record_from` says from when), and `snapshot.build_snapshot()` keeps every committed run the store lacks (`merge_runs`). Rebuilding either from the store alone silently deleted the record.
- **MLflow tracing follows the same rule** (added 2026-09-02): `tracking/tracing.py` owns it. The evalloop runner always calls `tracing.enable()` — every LLM call lands as a span (run → report → question → named agent stage → `Agent.run`) linked to the MLflow run; serving opts in with `MLFLOW_TRACING=1`. `tracing.span()` is a free no-op until `enable()` succeeds, so imports stay cheap and the demo container needs no tracking server. Never set `MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false` to merge MLflow into Logfire's tracer provider — it crashes pydantic-ai runs; the two providers coexist by staying separate.
- **Vite proxy** (`frontend/vite.config.ts`'s `BACKEND_PREFIXES`) must list every backend path prefix (`/healthz`, `/reports`, `/sessions`, `/eval`, `/admin`, `/traces`, `/demo`, `/metrics`). Missing entries cause silent HTML-404 failures in the browser. The UI's own pages (`/filings`, `/scoring`, `/leaderboard`, `/runs`, `/optimise`, `/agent`, `/gold`) must never collide with one.
- **`pred_program` column** must be present in every predictions CSV. `evaluate_cached` auto-detects its absence and forces a re-run.
- **Cached evaluations are committed**, not regenerated. `evaluation/predictions/` (prediction CSVs, HTML reports, joined CSVs), `evaluation/diagnostics/` (s7 `rules_*_<variant>.jsonl` + `rule_attempts_*_<variant>.jsonl` + diagnostic results), and `runs/<gepa_name>/` (GEPA artifacts including `optimized_runner.json`) are tracked in git so v1/v2 accuracy and GEPA outcomes reproduce across machines with `REUSE_CACHE=1` and no API calls. Do not add these directories to `.gitignore`. If you change pipeline behaviour and need fresh numbers, regenerate the CSVs and commit them — do not leave the working tree dirty or rely on `REUSE_CACHE=0` to mask stale state.
- **`archive/` holds retired experiment by-products** (GEPA iteration logs, DSPy/API parity CSVs, the abandoned s7 `v3_2` round), moved with `git mv` on 2026-08-31. Nothing reads it and `.dockerignore` excludes it; `archive/README.md` lists what moved and what stayed. Keep the GEPA prompts (`runs/*/optimized_runner.json`) and s7 stores (`rules_*_v3_1.jsonl`) where they are — code reads them.
- **`.dspy_cache/`** (~366 MB DSPy LM cache), **`mlruns/`**, **`.mlflow/`** and **`.traces/`** stay gitignored — they are dev state. `mlruns/` and `.mlflow/` are the archived pre-2026-09-21 MLflow stores: read-only, not migrated (decision D2); new runs go to the central server. What ships is the committed export: `evaluation/mlflow_snapshot.json` + `evaluation/registry.json`, regenerated with `convfinqa-mlflow snapshot`, and `evaluation/traces_snapshot.jsonl.gz`, regenerated with `convfinqa-mlflow traces-snapshot`. Sync `.dspy_cache/` via `rsync` to share warm caches; never commit it.
- **The demo is a demo of dev, so every committed artifact a read-only route opens must be in the image** (2026-09-08). The failure mode is silent by construction: those routes are written to degrade rather than 500, so an artifact left out of the `Dockerfile` does not fail the build, the deploy, or the smoke test — it empties a page. `evaluation/story.json` and `evaluation/readiness.json` were never copied in, and the public deployment served "no campaign has been recorded yet" for the whole campaign record: the entire Runtimes page, the campaign track, the progression chart, the judge panel, the readiness strip and four of the five landing HUD tiles, while dev showed all of them. `tests/test_demo_image.py::SHIPPED_ARTIFACTS` is the list, keyed by the route that reads each file, and `scripts/demo_smoke.sh` asserts the payloads are not their empty states — an empty state is a legitimate answer in dev and a deployment bug in the container. **The trace store ships the same way.** `.traces/traces.db` is 57 MB of dev state, so a cold container browsed only the turns it had itself replayed; `tracking/trace_snapshot.py` exports it to gzipped JSONL (6 MB, 8.4k turns) and seeds a store that has **no turns of its own** — emptiness is the guard, not the mode, so a dev machine keeps what it measured and a fresh checkout gets the record. Rows are copied **verbatim**, `trace_id` and `created_at` included: re-recording them would stamp every historical turn with the image's build time and report a batch evaluation as a traffic spike inside `/metrics/production`'s 24-hour window. `TRACE_SEED_FROM_SNAPSHOT=0` disables it; the test suite sets it false, or every temp store would load 8.4k rows.
- **`/metrics/production` is all-time, and the sparkline is bucketed rather than windowed** (2026-09-08). Every aggregate always covered every turn the store holds; what was wrong was the label. The response advertised `window_hours: 24` and the UI printed "last 24 h" above figures computed from the whole store, so a page showing 8,408 turns also said "no turns in the last 24 h" — and in the demo, whose committed traces are days old, the series really was twenty-four measured zeros for every source. The response now says `window: "all-time"`, and `_series` draws `SERIES_BUCKETS` (24) bars **ending at that source's newest turn**, widening `hour → day → week` until they span its whole history; each group carries `series_bucket`, `first_turn_at` and `last_turn_at` so the UI can label a bar instead of assuming one. Anchoring on the newest turn rather than on `now` is the load-bearing part: a series ending at `now` is empty on any deployment whose history is committed rather than live.
- **The traces store and MLflow tracing are different things, and the Traces page is the former.** `.traces/traces.db` is one row per *answered turn* — its four stage captures, its gold answer, its cost — written by serving and by the eval runner, and it is what `/traces`, `/metrics/production` and the Traces page read. MLflow tracing (`tracking/tracing.py`) is per *LLM call span* inside an eval run (run → report → question → agent stage → `Agent.run`), lives on the central MLflow server (`../nmp-central-ai`, http://localhost:5000), and needs that server. The in-app view is deliberately not MLflow-backed: the demo container has no tracking server, and the question the page answers — "why did it answer that" — is a turn-level question. What MLflow contributes to the app is the run/experiment history, exported to `evaluation/mlflow_snapshot.json` and read by `/admin/experiments`.
- **The UI's pages no longer live under `/admin`, but the Console's old addresses still do** (s16). `/admin` is an API prefix, and until 2026-09-30 it was also the UI's page prefix, with `/admin/experiments` spelled the same on both sides. The pages now sit at the house's eight slots and every old `/admin/*` address is a client-side redirect loader (`frontend/src/routes.tsx`, pinned by `routes.test.tsx`). So an old bookmark must still reach the SPA shell, not JSON: `serving/app.py` keeps the document-request rule — a `GET`/`HEAD` under `/admin` whose `sec-fetch-dest` is `document` (or whose `Accept` carries `text/html`) gets `index.html` — and Vite's `bypassDocumentRequests` does the same in dev. `fetch()` sends `*/*` and the SSE client sends `text/event-stream`, so no real API call is caught.
- **`DEMO_MODE` is baked into the Docker image**, not set in Terraform. Infrastructure must not be able to turn the public URL into a billable one. The demo container holds no API key at all.
- **The demo pack is rebuilt from committed CSVs**, never from fresh API calls: `uv run convfinqa-demo-pack`. Its events must stay identical to what `pipeline/runner.py::turn_events` emits — `serving/demo_pack/cli.py::events_from_row` is the other half of that contract.
- **The variant module `src/convfinqa/prompts/<variant>.py` (e.g. `v3_1.py`) is generated**, never hand-edited. The s7 harness writes it from the four `rules_<agent>_<variant>.jsonl` stores via `convfinqa.diagnosis.assembler`. To change it, edit the JSONL (or run the harness) and re-run `scripts/diagnose_failures.py --stage assemble`. The variant defaults to `settings.variant` (`v3_1`); override with `--variant`/`VARIANT`.
- **The frontend is the portfolio's house UI in blue** (s16, 2026-09-30; plan `.lavish/s16_ui-migration-and-ui-kit-plan.html`). Read the root `DESIGN.md` before touching `frontend/src`. `frontend/src/kit/` is the house UI kit (`nmp-central-ai/ui/kit`, stamped `nmp-ui-kit <ver>`) vendored byte for byte — never edit it here; change the kit and `make -C ../nmp-central-ai ui-sync PROJECT=P1`. This project owns `theme.css` (the blue accent), `app.css` (its extensions) and its pages. Plain CSS on tokens: no Tailwind, no component-library styling, no icon or chart library (Radix and cmdk stay for behaviour). Eight tabs in the house order — Overview · Filings & questions · Scoring · Leaderboard · Runs · Optimise · Agent · Gold review — are data in `frontend/src/app/nav.ts`; routes are data in `frontend/src/routes.tsx` with a redirect loader for every address the Console published, and `routes.test.tsx` lands them all. Chat is `/agent/chat/*`, the one `wide` route. Every page is a slide (eyebrow → h1 claim with one `<em>` → lead → evidence); every absent value prints `—` and its reason. `npm run lint:design` (`src/kit/design_lint.mjs`, in CI) fails on a hex outside the token files, text under 12.8px, banned fonts and Tailwind.
- **The production-readiness scorecard is a committed, checked file.** `evaluation/readiness.json` scores the repo against the portfolio's nine rubric dimensions (R1–R9) with a status, a `how`, proof paths and in-app routes per row; `GET /eval/readiness` serves it verbatim (keyless, demo-safe) and the landing strip, Architecture §13 and `/admin/readiness` render it from one component. Statuses are judgements and stay hand-maintained; what CI checks (`evalloop/readiness.problems`, run by `story_check` and `tests/test_readiness.py`) is that every proof path exists, every route is served, the vocabulary holds, and the headline score matches the rows. The portfolio's front matter is the second copy — change both. R2 is `partial` on purpose: the judge was tried, measured, and not adopted, which is a result worth showing rather than an `n/a`.
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
make -C ../nmp-central-ai up              # central MLflow server; UI at http://localhost:5000
uv run convfinqa-mlflow status            # tracking config, aliases, versions
uv run convfinqa-mlflow compare v2 v3_1   # exit 1 when not promotable
uv run convfinqa-mlflow promote v3_1      # refused unless the comparator passes
uv run convfinqa-mlflow champion-evidence <test csv>  # the CI gate's floor for the champion agent
uv run convfinqa-mlflow backfill          # rebuild history from committed artifacts
uv run convfinqa-mlflow snapshot          # export what the demo image reads
uv run convfinqa-mlflow traces-snapshot   # export the trace store the demo image seeds from
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
EVAL_MANIFEST=eval_loop_v2 uv run convfinqa-evalloop cycle --campaign c01
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
and diagnosis agrees on what "the gate split" means. The **Experiments page**
(`/admin/experiments`, backend `GET /eval/campaigns`) and the published page both read
`evaluation/story.json`, so they cannot disagree; rebuild both with `story`.

## Agent SDK experiment (s10)

A second runtime answers the same conversations: one Claude Agent SDK session per
conversation with the six calculator functions as its only tools
(`backends/agent_sdk.py`, `runtime="agent_sdk"`), run through the same loop and the
same gate as the four-agent pipeline so the two can be compared on the fixed split.
The invariants, all pinned by tests:

- **One prompt lineage, `sdk_prompts`.** The single-session prompt is `prompts/sdk_vN.py`
  (`SDK_PROMPT`), registered by hash in `registry.json → sdk_prompts` (`s1, s2, …`).
  `sdk_v1` is distilled from a bundle by `convfinqa-evalloop sdk-distil`; every later
  version is written by the SDK teacher. Never hand-edit one. `next_version("sdk_v3")`
  is `sdk_v4` — the lineage never shares a number with a pipeline `vN`.
- **One champion, and it is an agent** (universal champion, owner's decision 2026-10-15;
  supersedes "`sdk_champion`, never `champion`"). `aliases.champion` is `sdk_v1` and
  `registry.json → champion_agent` records how it runs: `{version, runtime: agent_sdk,
  model: claude-sonnet-5, effort: high}` — the configuration gate
  `g-20260905080150-00fce8a3` measured. The pipeline lineage keeps its best as
  `pipeline_champion` (`v8`): `registry.promote` moves only that alias, and every
  pipeline reader (cycle, gate, ledgers, bundle, prompt_ledger, release, demo pack) reads
  `registry.pipeline_champion()`, so a bundle that beats the previous bundle can never
  dethrone an agent it never faced. Three ways the champion moves, each a history event:
  `promote_sdk` (a new prompt; the record's model and effort stay), `promote_agent` (a new
  model/effort on the same prompt — needs a `story.model_swap_verdict` of `beats` or
  `matches` on test evidence), and the one-off `migrate_to_universal_champion`.
  `registry.sdk_champion()` still exists and returns the champion when it is an sdk
  version (the retired `sdk_champion` alias is read only from a pre-migration registry).
  **Serving and eval passes run the record**: `llm.sdk_model_name()` / `sdk_effort_name()`
  resolve setting → `champion_agent` → constant, so `SDK_MODEL`/`SDK_EFFORT` override one
  process and nothing else. Pinned by `tests/test_universal_champion.py`.
  **The CI gate's floor is on the record too** (2026-10-15): `champion_agent` carries
  `accuracy` and `evidence_run`, the exact test pass that measured the champion, and
  `tracking/gate.py` re-scores that pass against that floor. Before, the gate read the
  sdk version entry's metrics — empty for every sdk version — as a 0.00% floor and passed.
  A floor is a property of the agent, not the prompt (sdk_v1 is 90.5% on Sonnet 5, 87.4%
  on Haiku 5.5), so `promote_agent` re-floors on the verdict's `candidate_run`,
  `promote_sdk` on its `evidence_csv` (dropping the old floor when given none), and
  `convfinqa-mlflow champion-evidence <csv>` records one by hand; each refuses a pass
  whose name says it ran another version, model or effort, or an incomplete one. A
  champion with no recorded floor fails the gate — a missing floor is not a floor of zero.
- **Effort is part of the agent** (2026-10-15). Commit `27e11b5` (2026-09-10) pinned
  `effort="low"` on every SDK call, *after* the committed sdk_v1 evidence was measured with
  effort unset — i.e. the SDK default, `high`. So the 90.5% is a `high` figure, and serving
  ran an unscored `low` configuration from 09-10 until the champion record put it back on
  `high`. `run --sdk-effort` pins a pass; the run name carries it after the model slug
  (`…·s1-haiku-5-5-high-<stamp>`), it is the `sdk_effort` param and tag, `--resume-from`
  refuses a mismatch, and a run with no `sdk_effort` param is dated against
  `llm.SDK_EFFORT_PINNED_LOW_AT`. The teacher and judge keep their own `low`.
- **An sdk pass's price is `sdk_session_cost_usd`** (2026-10-15). The CLI's
  `total_cost_usd` is the session's *running total*, so `sdk_cost_usd` — the sum over
  turns — overstated every sdk pass 2.2–2.6×: sdk_v1's test pass recorded $27.62 and
  cost $11.83. And the CLI bundled with SDK 0.2.152 (2.1.259) does not know
  `claude-haiku-5-5`: it prints `unrecognized_model` (the API still serves the model,
  with `effort` and adaptive thinking as sent) and prices it on a $5/$25 fallback card,
  50× Haiku 5.5's ≤100K rate. `tracking/cost.py` corrects both;
  `convfinqa-evalloop backfill-sdk-cost` derived the figure for older passes from their
  traces (archived store → committed snapshot, central store → metric), and
  `story._arm` prices an sdk arm by it alone, showing no cost rather than the inflated
  one. `tests/test_sdk_cost.py` pins the SDK version the repricing table was measured
  on. One harness effect to know: a session's first API call carries 7 tools and every
  later one 65, so the conversations that open a pass together each write that ~50k
  prefix to the cache in one turn, against their own 60k ceiling, before any of them
  can read it. Both arms of s19 paid it — sdk_v1's Sonnet test pass had three such
  conversations peak at 59–61k — and a ceiling error in a pass's first batch is this,
  not the model looping.
- **Calculator tools only.** Arithmetic happens inside the tool server, never inline;
  the trajectory is what the calculator column is scored on.
- **Ledgers are append-only.** `evaluation/diagnostics/evalloop/{diagnoses,rewrites,gates}.jsonl`
  are opened with `"a"`, written, fsynced; nothing rewrites or reorders a line. Frozen
  `COLUMNS` per ledger, extension by trailing columns with defaults. Both arms append
  through `evalloop/ledgers.py`; each run mirrors its batch as `ledger_rows.jsonl`.
  `campaign.history` reads `gates.jsonl` first and falls back to MLflow search.
- **Skipped stages attribute to the skipped stage.** A program turn the session answered
  with an empty program is a preprocess fault; one with no calculator call is a calculator
  fault.
- **SDK draws never pool with pipeline draws, nor across agents.** `sdk_teacher.rank_classes`
  pools on the sdk prompt hash, `runtime="agent_sdk"` **and the answering agent**
  (`sdk_model`, `sdk_effort`; s21, 2026-10-15) — one prompt is a different agent on another
  model (sdk_v1: 90.5% on Sonnet 5, 87.4% on Haiku 5.5), and the ledger held 56 Sonnet
  diagnoses under sdk_v1's hash that would otherwise have steered a rewrite for Haiku. The
  diagnoser's memory filters the same way. Rows written before the model columns existed read
  as `ledgers.LEGACY_SDK_AGENT` (Sonnet 5 · high), which is what they ran on
  (`ledgers.sdk_agent_of`). The pipeline's `ledger.fault_history` filters on `multi_agent`. A
  campaign's cap is counted per runtime (`history(campaign, runtime=…)`).
- **A refusal is one failure class** (`sdk_teacher.GAVE_UP_CLASS`, `stable_class`). The frozen
  taxonomy has no "declined a value the report contains", so the diagnoser filed Haiku 5.5's
  "unavailable" answers — 20 of its 76 first-wrong train turns, its largest failure — under a
  dozen `new:` labels that each ranked as noise. Ranking and the writer's grouping read the
  stable class; the diagnosis keeps its own label, and `TEACHER_PROMPT` is untouched. A refusal
  attributed `ambiguous` still counts: that verdict doubts *which* stage declined, never
  whether the agent failed.
- **A challenger agent is optimised against the champion, never beside a re-run of it** (s21).
  `cycle --runtime agent_sdk --sdk-model M --sdk-effort E` runs every pass of the cycle on that
  agent; when it differs from `champion_agent` the cycle is a *challenger* cycle: its train
  failures are that agent's (`--train-csv`, repeatable, reuses committed early-stopped draws and
  refuses one whose name, model, effort or completeness disagrees), the writer is told its target
  (`target_model`/`target_effort` in the payload — `SDK_WRITER_PROMPT` is unchanged so old trace
  refs still resolve), a `--smoke-reports` pass checks the output contract, the challenger is
  scored on test and re-scored on the champion's own train seeds (`--rescore-seed`, default every
  seed the champion has a draw on), and `sdk_gate.gate_challenger` judges it against the
  champion's committed `evidence_run` and draws under s19's rule (beats, or matches within 3pp;
  train vetoes). A pass promotes prompt, model and effort together
  (`registry.promote_challenger`) — `promote_sdk` would serve the new prompt on a model it was
  never scored on, and `promote_agent` keeps the old prompt. On the champion's own agent the
  cycle reuses the evidence run as its baseline rather than running the champion again. A pass
  cut short stops the cycle with the `--resume-from` command; finish by hand with
  `gate-challenger`. `prompt_ledger.ensure_sdk(parent_version=…)` records what was edited, so
  sdk_v3 descends from s1, not from whichever entry was newest.
- **A gate row says what the registry did** (2026-10-15). The row used to be written before the
  promotion was tried, with `promoted` = "the rule passed", so a refused promotion left one on
  record that never happened. Now every SDK path promotes first and logs second: `promoted` is
  the registry's answer, `gate_passed` the rule's, `verdict_rule` (`house` | `model_swap`) and
  `verdict_outcome` say which rule, and `baseline_/candidate_sdk_model|effort` name both agents;
  diagnoses carry `sdk_model`/`sdk_effort`, rewrites `target_model`/`target_effort` (all
  trailing columns). Campaign streaks read `gate_passed`, falling back to `promoted` on older
  rows where that is what it meant.
- **The whole record is one timeline, test passes only** (`story.test_timeline`,
  `champion_steps`, `train_passes`, `challengers`; s21). Every `*evalloop-test100-*` pass of every
  agent, oldest first, each against the champion of its day, with what judged it — a gates row
  (promoted · passed · rejected), a challenger verdict (beats · matches · falls_short), the
  champion's own pass, a baseline, or `incomplete` (unscored turns, or a pass in
  `story.KNOWN_INCOMPLETE` from before the unscored rule; never compared, never drawn). The step
  line comes from the registry's history and steps at the **pass that measured** each champion,
  not at the promotion's timestamp: sdk_v1's promotion was written at 08:01 UTC, after sdk_v2's
  pass at 06:28, and placed by the event sdk_v2 would be drawn against v8. An sdk *version's*
  accuracy (`_gate_runs_by_version`) is its pass on the champion's agent and on the gate split
  only — keyed on the version alone it became whichever model last scored it, and Runtimes §5
  showed sdk_v1 at 87.39% (a Haiku pass) beside the +8.88pp its Sonnet pass measured; sdk
  experiment rows take both accuracies from their gates-ledger row. Carried campaign sections
  merge **per campaign**: the first gate run a new campaign writes to the central store would
  otherwise drop c01–c03 and s01. Optimise › Story and the Runs ledger draw it; the published
  page lists it and `story_check` fails a page that does not.
- **The SDK arm's experiment cap is 2, the pipeline's is 5**
  (`campaign.MAX_EXPERIMENTS_BY_RUNTIME`, `max_experiments(runtime)`; owner's instruction,
  2026-09-05). A cycle costs subscription time rather than a couple of dollars of DeepSeek, and
  `sdk_v1` left only 33 wrong turns to search. One consequence to know: with a cap of 2 a campaign
  ends before two consecutive rejections can accumulate, so `single_area_mode` is unreachable for an
  SDK campaign unless the cap is raised — the machinery stays, tested against a raised cap.
- **The target is a failure class, not a subagent.** The rewrite may edit several tagged
  areas of the one prompt (one edit per class, one `rewrites.jsonl` row each). Two
  consecutive rejections do not block anything — they switch the lineage to
  `single_area_mode` (`propose_version(..., max_areas=1)`) for the rest of the campaign.
- **The gate judges overall accuracy** (`evalloop/sdk_gate.py::gate_overall`, the same
  one-sided cluster-corrected McNemar as `gate.py`) with the per-stage panel beside it.
  The verdict is recorded by `sdk_gate.log_gate_verdict`, not `teacher.log_gate_verdict` —
  the latter's ledger row is hard-wired to `runtime=multi_agent`, and a wrong runtime on
  an append-only row pools an SDK verdict into the pipeline's history forever.
- **`--runtime agent_sdk`** is on `run`, `diagnose`, `propose`, `cycle` and `gate-targeted`
  (which then takes `--target-class`); `--version sdk_vN` under the wrong runtime fails at
  parse time. New subcommands: `sdk-distil`, `backfill-ledgers [--no-mlflow]`,
  `ledger-trace --question-id|--edit-id`.
- **The open runtime, `--runtime open_agent`, is a record-only challenger** (s25–s27;
  `backends/open_agent.py`, display name `llm_react_v1` from `llm.OPEN_AGENT_NAMES`). It runs
  the champion's single-session contract — the `sdk_v1` prompt, six `mcp__cfq__*` tools,
  `StructuredOutput` schema and limits imported from `agent_sdk`, never copied — in one
  Pydantic AI agent on DeepSeek (`deepseek-flash`, thinking high). It is **not a new prompt
  version** (the prompt is byte-identical to `sdk_v1`; prompts are identified by hash). The
  module docstring lists the only accepted differences. `gate-challenger --runtime open_agent`
  writes its gates row with `runtime=open_agent` (never `agent_sdk` — the ledger is
  append-only) and **refuses `--promote`**: serving has no open runtime. Result o01-e01:
  `falls_short` of the champion. Keys come only from `~/.env`/`.env`. `evalloop/smoke.py`
  picks the 20-conversation smoke from the champion's own committed seed-2027/2028 draws;
  champion numbers come only from its committed runs, nothing new runs on Claude.
- **The SFT student (s28–s30) learns from llm_react_v1's own conversations and is scored as
  an open agent.** `convfinqa-sft` fetches a pass's `trajectories.jsonl.gz`, gates whole
  conversations (F1–F8, N1; `--prefix-rows` also keeps an F1 drop's turns before its first
  wrong one) into `evaluation/sft/<version>/`, and trains a LoRA on Modal; the student is
  Qwen3-4B on vLLM (`STUDENT_URL`), `student-qwen3-4b[-sftN-…]`, effort `none`. **Test 100
  (`eval_loop_v2` test) is its holdout**: never collected on, trained on or used to choose;
  `sft_v1` val picks checkpoints, and a tuned checkpoint opens test only on the owner's word.
  `sft_v2` adds an aimed draw (`sft/aim.py`, ranked by the student's val miss rates, never
  test's). **Two harnesses:** `pydantic_ai` is llm_react_v1's and the teacher's and does not
  change; `pydantic_ai_checked` (`backends/open_checks.py`, H1–H3) is the student's from s29,
  so a student baseline is re-scored under it rather than compared across harnesses. Pydantic
  AI 2.x records a refused call as a `tool-return` with `outcome: "retried"` (not
  `retry-feedback`) and can fold a turn's closing tool return into the next question's
  request; `sft/trajectories.turns` and `sft/rows` handle both (`tests/test_sft_rows.py`).
- **DPO (s33+) builds preference pairs from train-split conversations only** (`convfinqa-dpo`,
  `src/convfinqa/dpo/`, data in `evaluation/dpo/<version>/`; trainer `sft/train/dpo.py`, configs
  `sft/configs/dpo1_*.yaml`). `sample` draws tries per question from the student on vLLM; wrong
  tries pair against right ones (`pairs`, capped per question), and `teacher` backfills the
  questions the student got all wrong with llm_react_v1's correct turns as the chosen side
  (`dpo_v2`). Val and test are never sampled: val scores the tuned LoRA, **test 100 stays sealed**
  unless the owner approves opening it. DPO rounds share `runs/sft/` but carry a `kind` in
  `run.json` and are **not** on the Fine-tuning progression (`sft/progress.py::_sft_runs`).
- **Optimise › Fine-tuning reads one committed record, `evaluation/fine_tuning.json`** (s32).
  `convfinqa-sft progress` (`sft/progress.py`) builds it from committed files only: the
  student's and teacher's pass CSVs, `runs/sft/*/run.json` + `metrics.jsonl`, the dataset
  cards and the `sft_v1` manifest; `tests/test_fine_tuning.py` fails when the committed copy
  is not a byte-for-byte rebuild, so rebuild it after any new pass or training run. **A pass is
  placed by its report ids, never its name**: the val passes ran as `--split train` and are
  named `train100`. A round's pick is its sweep's highest val accuracy (one harness, default
  temperature); a harness change is a stage of its own on the progression, so each paired step
  is one change. `GET /eval/fine-tuning` serves the record verbatim; `GET
  /eval/fine-tuning/questions?pass=…` opens only passes the record lists as complete, on one
  split. Each pass links to `/optimise/campaigns?run=<run_id>`, which opens that run under its
  own split — so the student's runs must be in `evaluation/mlflow_snapshot.json`
  (`convfinqa-mlflow snapshot`) for the link to land in the demo.
- **The write-up is `docs/optimization/agent-sdk.html`**, rendered by `story` from the same
  `story.json` (`runtime_comparison`, `sdk_campaigns`, `sdk_champion`) and covered by
  `story_check`. Every comparison field is `None` until an SDK run exists — the page says
  "not yet run", never zero.
- **A rate-limited turn is unscored, never wrong** (2026-09-05). The CLI answers a spent
  account with prose — `You've hit your session limit · resets 5:40pm`, `Credit balance is
  too low` — which is *no* answer, and a 349-question pass scored 176 of them as wrong and
  reported 44.4% as though it had measured the agent. `agent_sdk.RATE_LIMIT_MARKERS` matches
  those texts (only on a reply that did not arrive as a structured object, so an answer that
  says "quota" is still an answer) and raises `SdkRateLimitError`: **not retried** — the
  correction buys the same refusal — and it **aborts the conversation**, since the session is
  spent. The runner writes those turns with `unscored=True`, `pred_answer` empty and `error`
  prefixed `rate_limited: `, and every consumer of `correct` (accuracy, `n_wrong`,
  `program_accuracy`, the panel, the `accuracy_by_*` slices) reads the scored subset — an
  unscored turn is absent from the numerator *and* the denominator.
- **A pass with unscored rows is incomplete and cannot be gated.** It carries metrics
  `n_unscored` / `n_rate_limited` / `n_scored` / `complete`, the MLflow tag `incomplete=true`
  and the param `unscored_rows`, `complete` and `n_unscored` in the returned summary, and a
  loud closing line naming how many turns were never answered. `gate.load_run_csv` — the
  shared door both `gate.py` and `sdk_gate.py` come through — refuses such a CSV
  (`IncompleteRunError`) naming the file and the count: a paired comparison over a set half of
  whose turns were never attempted is not a comparison. There is no override; the pass is
  finished with `--resume-from`. A CSV predating the column loads unchanged (missing =
  all-False).
- **A conversation is reused whole or re-run whole.** `run --resume-from <csv>` copies every
  conversation the prior CSV answered *completely* (all questions, no unscored, no
  `rate_limited:` row) through verbatim — its own `run_id` and `trace_id` kept, with
  `resumed_from_run_id` naming the run it came from — and re-runs every other conversation
  from turn 0. A half-finished conversation is never stitched: the runtime's premise is that a
  conversation is one session whose later turns depend on its earlier ones. Split, version,
  runtime and the report set (a superset, and the same `--train-seed` draw) are checked
  against the prior CSV before anything runs; params `resumed_from` / `n_reused_conversations`
  and metric `n_reused_questions` are logged, and `stage_scores.score_rows` re-scores the
  whole frame so the panel covers reused and fresh rows alike.

- **The model is part of the run's name and record** (2026-09-06). `run --sdk-model <id>`
  pins the model for one sdk pass; the run name carries its slug between the composition and
  the stamp (`sdk-evalloop-test100-sdk_v1·s1-haiku-4-5-<stamp>`, `llm.sdk_model_slug`), the
  full id is the `sdk_model` param **and** tag (a search key), and `--resume-from` refuses a
  CSV whose name carries a different slug. Runs named before the slug existed ran the default
  model, which their `sdk_model` param records. **A challenger agent is paired against the
  champion's own committed runs, never a re-run of the champion** (owner's rule, s19):
  `story.sdk_model_comparison` keys agents on **(model, effort)** and pairs each against the
  champion's record — on test question by question (the gate's test, from the two committed
  CSVs), and on sdk_v1's own early-stopped train draws (`--train-seed 2027|2028
  --stop-at-first-wrong`, reproduced id for id) **per conversation**, since early stop leaves
  later turns with no counterpart. Each pair carries `story.model_swap_verdict`: `beats` (the
  house gate), `matches` (test 95% CI lower bound ≥ −`MODEL_SWAP_MARGIN_PP` = 3pp, test and
  program turns not significantly worse, no extra token-ceiling errors) or `falls_short`;
  train is a veto, never a substitute. Only `registry.promote_agent` acts on it.
  `runtime_comparison` pins the sdk arm to the run its gate measured
  (`candidate_eval_run_id`), so no later pass — or promotion — re-labels the v8-vs-sdk_v1
  verdict. The page (`docs/optimization/agent-sdk.html`) and
  `/optimise/runtimes` both show the table once a second model exists; `story_check` fails a
  page that hides it.

```bash
uv run convfinqa-evalloop run --split test --version sdk_v1 --runtime agent_sdk \
  --sdk-model claude-haiku-4-5-20251001      # one scoring pass, model swapped, all else equal
uv run convfinqa-evalloop run --split train --version sdk_v1 --runtime agent_sdk \
  --sdk-model claude-haiku-5-5 --sdk-effort high \
  --train-seed 2027 --stop-at-first-wrong   # s19: a challenger on the champion's own train draw
uv run convfinqa-evalloop run --split train --version sdk_v1 --runtime agent_sdk \
  --train-seed 2027 --report-ids-file <ids.txt>   # narrow a seeded draw to listed ids; refuses any id outside it
uv run convfinqa-evalloop sdk-distil --source-version v8 --new-version sdk_v1
uv run convfinqa-evalloop run --split test --version sdk_v1 --runtime agent_sdk
uv run convfinqa-evalloop run --split test --version sdk_v1 --runtime agent_sdk \
  --resume-from evaluation/predictions/evalloop/<partial>.csv   # finish a refused pass
uv run convfinqa-evalloop cycle --campaign s01 --runtime agent_sdk
EVAL_MANIFEST=eval_loop_v2 uv run convfinqa-evalloop cycle --campaign h01 --runtime agent_sdk \
  --sdk-model claude-haiku-5-5 --sdk-effort high --concurrency 4 --diagnose-concurrency 8 \
  --train-csv <haiku draw 2027>.csv --train-csv <haiku draw 2028>.csv   # s21: a challenger cycle
uv run convfinqa-evalloop gate-challenger --candidate-csv <test>.csv --candidate-version sdk_v3 \
  --sdk-model claude-haiku-5-5 --sdk-effort high --candidate-train-csv <2027>.csv \
  --candidate-train-csv <2028>.csv --campaign h01 --label h01-e01   # finish one by hand
uv run convfinqa-evalloop gate-targeted --runtime agent_sdk --target-class <label> \
  --campaign s01 --label s01-e03 --rewrite-id rw-…  ...
uv run convfinqa-evalloop backfill-ledgers --no-mlflow
uv run convfinqa-evalloop ledger-trace --question-id <report>_q<n>
```

## The confidence judge (s12)

The runtime decision (s11) moved serving to the Agent SDK; the judge was the attempt
to make it self-checking. A second, cheaper model reads a finished turn's trace and
returns a band: `high` means the trail verifies, `low` means it does not. **Measured on
the sealed split, the band did not earn the right to gate** (s13) — it is shown as a
caveat instead, and `JUDGE_MODE=gate` keeps the withholding path for a judge that does.
Everything lives in `evalloop/judge.py`; the invariants, all pinned by tests:

- **The judge never sees gold.** `judge.judge_payload` builds its input from the
  same readers the teacher uses (question, history, document, the trail: sub-questions,
  retrieved cells with sources, program, calculator trajectory, answer, reasoning) and
  asserts that no `GOLD_KEYS` entry reached it. `validate_judge_prompt` refuses a
  distilled prompt that mentions gold / ground truth. `row_from_capture` is the bridge
  from a served turn's capture to the row shape, so serving and scoring build
  byte-identical judge inputs.
- **Trained the way `sdk_v1` was: diagnose with gold, distil without it.** The teacher
  (pinned to Sonnet 5 via `TEACHER_MODEL`; the loop's default is Opus) goes through
  *every* case of the optimise split — correct and incorrect — and writes what the trace
  got wrong or right, how each of six named checks reads (`CHECKS`), whether a judge
  without gold could have known, and one rule such a judge could run. One distil call
  writes `prompts/judge_jN.py` (`JUDGE_PROMPT`); round 2 re-diagnoses the judge's own
  misses with its verdict attached and revises. Never hand-edit a judge prompt.
- **Three splits, cut by conversation, from committed run CSVs only**
  (`evaluation/judge/dataset.json`). *optimise* is balanced 50/50 — every attributable
  negative (NON_AGENT attributions excluded) plus as many positives, hard ones first —
  because the teacher's unit is a miss. *calibrate* stays at natural prevalence because
  coverage and high-band precision are prevalence-dependent; it gates versions.
  *test* is the runtime's gate split, scored once by the frozen champion. The manifest
  carries the attribution rule id; a rule change means a rebuild.
- **Band-based metrics are the contract** (`selective_metrics`): coverage, high-band
  accuracy, failures caught, the one-sided 95% Wilson bound on the unseen high-band
  error (≈3/n at zero errors — a bound, never "100%"). `p_correct` is recorded for the
  risk–coverage curve. Operating target: high-band error ≤ 1%.
- **Promotion rule** (`gate_judges`, `registry.promote_judge`, alias `judge_champion`,
  lineage `judge_prompts` `j1, j2, …`): the candidate's high band stays inside the
  target, coverage rises (no exception for a baseline that itself misses target),
  failures caught do not fall. A judge is never promoted on the test split — `gate_judges`
  derives each side's split from its own scores CSV (the trailing `split` column, or the
  `judge-score-<split>N-…` filename for a CSV written before that column existed) rather
  than trusting a caller-supplied argument, refuses a mismatched pair, and refuses outright
  when the evidence is `test`; the derived split is what `registry.promote_judge` checks.
  `registry.promote` also refuses a judge version the same way it refuses an sdk version —
  serving reads `champion` to build the four pipeline agents, and a judge there is a
  champion nothing can serve. The history event records the `runtime_version` it was
  calibrated to.
- **A failed judge call fails closed** — `low` band, error in the record — in scoring
  and in serving. A guard-rail that fails open is not one. **A rate-limited one is
  unscored instead** (2026-09-08), the same rule the eval runner applies to a
  rate-limited turn: a spent subscription is *no verdict*, not a cautious one. Failing
  it closed made the j2 calibrate pass withhold 175 of its 304 turns and report a
  failure capture of 56.7% it had never earned, and the gate decided on it. `score_split`
  catches `TeacherRateLimitError`, writes `unscored=True` with no band, and
  short-circuits the rest of the pass rather than paying wall clock for the same
  refusal; `scored_only` is the door every consumer of `band` comes through, so an
  unscored turn is absent from the numerator *and* the denominator; the run carries
  `n_unscored`/`complete`, the tag `incomplete=true` and a loud closing line; and
  `gate_judges` refuses such a CSV (`IncompleteJudgeScoresError`) naming the file and
  the count, with no override. Serving still fails a rate-limited turn closed — a
  served answer with no judge must be withheld.
- **Serving** (`SERVING_RUNTIME=agent_sdk`, the default; `JUDGE_ENABLED`): one live
  `ClaudeSDKClient` per chat session (`serving/sdk_session.py`), closed on delete,
  eviction and shutdown; `serving/sdk_turn.py` holds the stage frames from the capture
  until the judge has ruled, then emits them, followed by a `judge` frame and an
  `answer` frame with `band`/`withheld`. What a `low` band does to those frames is
  `settings.judge_mode` (`JUDGE_MODE`, default `advisory`): under `advisory` the answer
  is released with the band and the judge's reason/failed checks shown as a caveat;
  under `gate` the stage frames are redacted (`output`/`args`/`result` stripped, since
  a program's calculator trajectory or a number turn's retrieved cell *is* the answer)
  and `answer` is empty on the wire and in the visible history. Either way the trace row
  keeps the value (`capture["judge"]["answer"]`), `judge_band` and `judge_p`, so the
  policy can change without breaking the record. The judge runs
  through `evalloop/sdk.py::run_structured(model=judge_model_name())` — the Agent SDK
  on the subscription — the same chokepoint as the teacher. The demo container replays
  the recorded pack and never runs either.
- **The record**: `evaluation/judge/judge_diagnoses.jsonl` and `judge_gates.jsonl`
  (append-only), `evaluation/judge/scores/*.csv` (one per version × split), MLflow
  experiment `convfinqa-judge` (kinds `judge_diagnose`, `judge_distil`, `judge_score`).
  `story.json → judge` (`judge.summary()`) feeds `/eval/campaigns`, `/scoring`, `/optimise/runtimes`,
  the landing HUD and `docs/optimization/agent-sdk.html`; `story_check` fails a page
  that hides the judge once a champion exists.

```bash
uv run convfinqa-evalloop judge-dataset --optimise-csv <draw A> --calibrate-csv <draw B> --test-csv <gate pass>
TEACHER_MODEL=claude-sonnet-5 uv run convfinqa-evalloop judge-diagnose --split optimise
TEACHER_MODEL=claude-sonnet-5 uv run convfinqa-evalloop judge-distil --new-version judge_j1
uv run convfinqa-evalloop judge-score --version judge_j1 --split calibrate
TEACHER_MODEL=claude-sonnet-5 uv run convfinqa-evalloop judge-diagnose --split calibrate \
  --judge-scores evaluation/judge/scores/<j1 calibrate>.csv --judge-version judge_j1   # round 2: the misses
TEACHER_MODEL=claude-sonnet-5 uv run convfinqa-evalloop judge-distil --new-version judge_j2 --base-version judge_j1 --round 2
uv run convfinqa-evalloop judge-score --version judge_j2 --split calibrate
uv run convfinqa-evalloop judge-gate --baseline-scores <j1 cal> --candidate-scores <j2 cal> \
  --baseline-version judge_j1 --candidate-version judge_j2 --promote
uv run convfinqa-evalloop judge-score --version <champion> --split test      # once
```

## Eval loop (M1) & teacher (M2) — the underlying commands

```bash
uv run convfinqa-evalloop make-splits                  # committed split manifest (train/test/holdout)
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
