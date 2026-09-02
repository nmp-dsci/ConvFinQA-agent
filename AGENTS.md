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
5. **Eval loop (M1/M2/M2.5)** — `src/convfinqa/evalloop/`. A second, self-improving promotion path over the committed train/test/holdout splits: `runner.py` (traced eval runs), `teacher.py` (first-wrong diagnosis + one-subagent challenger proposal), `gate.py` (net-positive and targeted gates), `stage_scores.py` (per-agent gold-derived metrics), `kappa.py` (teacher-vs-human agreement), `release.py` (sealed M3 holdout gate), `splits.py`, `cli.py` (`convfinqa-evalloop`).
6. **FastAPI server** — `convfinqa.serving.app:create_app`, routed through `serving/routes/` (`chat`, `evaluation` incl. `/eval/dataset`, `traces`, `admin`), backed by `serving/sessions.py` (in-memory session store), `serving/limits.py` (rate limiting), `serving/research.py` (s7/GEPA launch), and `serving/demo_pack/` (recorded replay for `DEMO_MODE`).
7. **React frontend** — `frontend/`. A status-board landing at `/`, a sessions/thread-inspector chat at `/chat`, and an instrument-style admin section at `/admin` (Overview, Evaluations, Dataset, Experiments, Traces + detail, Research, System) — all seven admin pages are visible read-only in the public demo, gated by a route filter, a real `<fieldset disabled>`, and a server 501/403.
8. **Container & infra** — `Dockerfile` (serves API + built SPA from one origin), `docker-compose.yml` (`demo` / `dev` toggle, plus an always-on `mlflow` tracking-server service), `infra/terraform/` (`bootstrap/` OIDC role, `demo/` ECR + App Runner + alarm), `.github/workflows/deploy-aws.yml` (keyless deploy chained on CI).

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
| `src/convfinqa/evaluation/` | Metrics/cache plus package import paths for evaluation, reporting, joining, and API evaluation. |
| `src/convfinqa/prompts/` | Versioned prompts (`v1.py`, `v2.py`, `v3_1.py`) and `load/latest` helpers. The variant module `prompts/<variant>.py` (e.g. `v3_1.py`) is generated by the s7 harness from JSONL rule stores. |
| `src/convfinqa/diagnosis/` | s7 prompt-improvement harness (`models.py`, `loader.py`, `prompts.py`, `agents.py`, `verify.py`, `rules_store.py`, `assembler.py`, `harness.py`, `results_writer.py`, `results_html.py`, `aggregator.py`, `cli.py`). |
| `src/convfinqa/reporting/` | Shared HTML-report mechanics (`html_report.py`): theme CSS, sticky inspector panel + viewer JS, `render_cell`, `render_page`. Used by `evaluation/reporting.py` and `diagnosis/results_html.py`. |
| `src/convfinqa/llm.py` | **The single LLM choke point.** `guard_llm_call()` (demo gate), `_RetryTransport` (retry/timeout policy), `get_provider()`/`get_model()`, `lm_mini()`/`lm_max()` factories, `dspy_lm_kwargs()`. Nothing else may construct a model. |
| `src/convfinqa/tracking/` | `bundle.py` (fingerprint: prompts + GEPA overlay + model ids + dataset hash + code SHA), `mlflow_log.py`, `tracing.py` (MLflow trace spans: run → report → question → agent stage → `Agent.run`; `enable()`/`span()`), `traces.py` (per-stage IO trace store), `comparator.py` (promotion contract, net-positive paired comparison + McNemar p), `registry.py` (champion/challenger aliases, per-agent prompt lineage in `agent_prompts`, append-only history), `prompt_ledger.py` (per-agent prompt versioning: content-hash identity, `t3.p3.r4.c3` compositions), `backfill.py`, `snapshot.py` (demo image export), `gate.py` (CI eval-regression gate), `cost.py` (token/cost accounting), `cli.py` (`convfinqa-mlflow`). |
| `src/convfinqa/evalloop/` | `runner.py` (traced eval runs over a split), `teacher.py` (first-wrong-per-report diagnosis + one-subagent challenger proposal, `TEACHER_PROMPT` taxonomy), `gate.py` (M1 net-positive gate + M2 `gate-targeted`), `stage_scores.py` (per-agent gold-derived metric panel), `kappa.py` (teacher-vs-human agreement), `release.py` (M3 sealed holdout gate), `splits.py` (train/test/holdout manifest), `cli.py` (`convfinqa-evalloop`). |
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
| `mlruns/`, `.traces/` | Local MLflow store and trace DB. Gitignored — the committed snapshot/registry is what ships. |
| `infra/terraform/bootstrap/` | Run once by hand: the GitHub OIDC deploy role. Not applied by CI. |
| `infra/terraform/demo/` | ECR + App Runner + a 5xx alarm. Reconciled by `deploy-aws.yml` after each push to `main`. |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | The demo image (`DEMO_MODE` baked in, not set via Terraform), the local `demo`/`dev` toggle, and an always-on `mlflow` tracking-server service (`docker compose up -d mlflow`). |
| `.github/workflows/ci.yml`, `.github/workflows/deploy-aws.yml` | CI (lint, mypy, pytest, frontend checks, eval-regression gate, Docker build, `terraform fmt`/`validate`) and the keyless AWS deploy chained on CI passing. |
| `frontend/` | Vite + React + Zustand + Tailwind operator console ("The Console"): landing status board at `/`, chat at `/chat`, admin section at `/admin` (Overview, Evaluations, Dataset, Experiments, Traces, Research, System), IBM Plex type, terminal-amber accent, dark-first with a light variant. |
| `tests/` | pytest suite (170 tests), including `test_demo_mode.py` (pins the no-model-at-import-time invariant), `test_tracking.py`, `test_evalloop.py`, `test_limits.py`. |

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
uv run convfinqa-mlflow status
uv run convfinqa-mlflow compare v2 v3_1   # exit 1 if not promotable
uv run convfinqa-mlflow promote v3_1      # refused unless the comparator passes
uv run convfinqa-mlflow backfill          # rebuild history from committed artifacts
uv run convfinqa-mlflow snapshot          # export what the demo image reads
uv run python -m convfinqa.tracking.gate  # the CI eval-regression gate
```

Eval loop (M1) and teacher (M2/M2.5) — see CLAUDE.md's "Eval loop" section and
the README's §Eval loop (M1) & teacher (M2/M2.5) for the full command set
(`make-splits`, `run`, `gate`, `diagnose`, `propose`, `gate-targeted`,
`backfill-prompts`, `mirror-prompts`, `kappa`, `release`), all under
`uv run convfinqa-evalloop <subcommand>`.

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
- **`mlruns/`, `.traces/`** — gitignored. Local MLflow store and trace DB; dev state, not shipped.

Rule of thumb: if regenerating it costs an API call, commit it. If it can be rebuilt locally without network, leave it ignored.

## Frontend Proxy Invariant

Every backend path prefix must be listed in `BACKEND_PREFIXES` in `frontend/vite.config.ts`, which feeds both `server.proxy` and `preview.proxy`:

```ts
const BACKEND_PREFIXES = ['/healthz', '/reports', '/sessions', '/eval', '/admin', '/traces', '/demo', '/metrics']
const proxy = Object.fromEntries(BACKEND_PREFIXES.map((p) => [p, API_BASE]))
```

If a new backend route prefix is added, add it to `BACKEND_PREFIXES` or the browser can receive HTML 404s instead of JSON.

`/admin` is both an API prefix and a UI route prefix (the admin console lives at `/admin/*`), so the proxy must hand document requests (`sec-fetch-dest: document` / `accept: text/html`) back to Vite's client router instead of forwarding them to FastAPI — otherwise a browser navigating to `/admin/evaluations` gets proxied, matches no backend route, and falls through to a built `index.html` that references dist asset hashes the dev server doesn't serve. `bypassDocumentRequests()` in `vite.config.ts` is what does this.

## Environment Variables

| Variable | Required | Description |
|---|---:|---|
| `DEEPSEEK_API_KEY` | Yes for any LLM call | DeepSeek via LiteLLM/OpenAI-compatible provider. Optional at boot — only `Settings.require_deepseek_api_key()`, called from `llm.py`, demands it, and only when a call is actually about to happen. The demo container has none. |
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
| `MLFLOW_TRACKING_URI` | No | `file:` store in dev; unset in the demo image, which reads the committed snapshot instead. |
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
