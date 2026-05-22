# ConvFinQA Agent

An AI engineering project for [ConvFinQA](https://github.com/czyssrs/ConvFinQA): multi-turn financial QA over report text and tables. The system evaluates, optimises, serves, and visualises a four-stage agent pipeline.

## Evaluation Results

Cached Pydantic AI evaluator across prompt versions (770-question held-out sample). Reproduces offline from committed `evaluation/pydantic_predictions_<version>.csv` — no API calls when `REUSE_CACHE=1`.

```bash
REUSE_CACHE=1 uv run convfinqa-eval
```

```
[v1] cache hit: 200/200 conversations (770 questions) — skipping
[v1] combined accuracy: 73.0%  (562/770 questions)
Wrote evaluation/pydantic_predictions_v1.html

[v2] cache hit: 200/200 conversations (770 questions) — skipping
[v2] combined accuracy: 77.1%  (594/770 questions)
Wrote evaluation/pydantic_predictions_v2.html

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
question=3                 116        68.1%         69.8%
question=4                  60        61.7%         75.0%
question=5                  24        62.5%         66.7%
question=6                  10        70.0%         90.0%
question=7                   1         0.0%          0.0%
----------------------------------------------------------
```

v2 beats v1 by +4.1 pp overall; biggest gains are on program turns (+5.2 pp), Type II conversations (+6.9 pp), and deeper turns (`question=4`: +13.3 pp, `question=6`: +20.0 pp).

### Version differences

- **v1** — Original baseline. Compact instructions per agent: classify (triage), decompose (preprocess), look up (retriever), compute (calculator). Same four-stage pipeline as v2; differences are purely in the system prompts. See `src/convfinqa/prompts/v1.py`.
- **v2** — GEPA-optimised prompts produced by a full real run (`gepa_real_20260502_005251`). Prompts are longer and more explicit: worked examples, explicit percentage convention (`multiply(..., 100)` outermost), clearer Type I vs Type II conversation guidance, and tighter sub-question specification rules (year + entity + metric). Pipeline structure and tools are unchanged from v1. See `src/convfinqa/prompts/v2.py`.
- **v3_opt** *(generated)* — Assembled by the s7 prompt-improvement harness from v2 + verified `rules_<agent>_v3_opt.jsonl`. Not hand-edited. See §Prompt-Improvement Harness (s7) below.

## Current Layout

The repository uses a `src/convfinqa/` package layout. No Python modules remain at the repo root — everything imports through `convfinqa.*`.

| Path | Purpose |
|---|---|
| `src/convfinqa/config.py` | Settings and dotenv loading. |
| `src/convfinqa/data/loader.py` | Dataset loader, canonical `qa_data`, `_DOCS`, evaluation sample. |
| `src/convfinqa/data/schemas.py` | Shared Pydantic models. |
| `src/convfinqa/pipeline/` | Shared stage models, calculator tools, wire format, runner import path. |
| `src/convfinqa/backends/` | Backend import paths for DSPy and Pydantic AI. |
| `src/convfinqa/evaluation/` | Metrics, cache, evaluation runners, reporting, API evaluation import paths. |
| `src/convfinqa/prompts/` | Versioned prompt modules (`v1`, `v2`, generated `v3_opt`). |
| `src/convfinqa/diagnosis/` | s7 diagnose → route+fix → verify harness (per-case prompt improvement). |
| `src/convfinqa/serving/` | FastAPI app and Typer CLI package paths. |
| `src/convfinqa/optimization/` | GEPA and prompt optimisation entry points. |
| `scripts/` | Installed command entry points. |
| `frontend/` | React/Vite UI. |
| `evaluation/` | Cached prediction CSVs + dark-themed HTML reports. Tracked in git so accuracy reproduces offline. |
| `runs/` | GEPA optimization artifacts (`optimized_runner.json`). Tracked in git so prior runs are usable on any clone. |
| `.dspy_cache/` | DSPy LM response cache (~366 MB). Gitignored; rsync between machines for warm scoring. |

## Pipeline

Every turn flows through the same four stages:

```text
question + report + history
  -> triage      -> turn_type, conv_type
  -> preprocess  -> sub_questions, program        (program turns only)
  -> retriever   -> raw values / direct answer
  -> calculator  -> final computed answer         (program turns only)
```

Calculator tools are `add`, `subtract`, `multiply`, `divide`, `exp`, and `greater`.

## Setup

```bash
uv sync
cd frontend && npm install
```

Required environment for model-backed runs:

```bash
DEEPSEEK_API_KEY=sk-...
```

Optional:

```bash
ANTHROPIC_API_KEY=...
LOGFIRE_TOKEN=...
```

## Evaluation

Run the cached Pydantic AI evaluator across prompt versions:

```bash
REUSE_CACHE=1 uv run convfinqa-eval
```

Expected cached baseline:

| Version | Accuracy |
|---|---:|
| `v1` | `73.0%` (`562/770`) |
| `v2` | `77.1%` (`594/770`) |

Useful variants:

```bash
PROMPTS_VERSION=v2 uv run convfinqa-eval
REUSE_CACHE=0 uv run convfinqa-eval
```

Outputs are written under `evaluation/`, for example:

- `pydantic_predictions_v2.csv`
- `pydantic_predictions_v2_joined.csv`
- `pydantic_predictions_v2.html`

## Serving

Start the backend:

```bash
uv run python -m uvicorn convfinqa.serving.app:create_app --factory --workers 1 --port 8765
```

`--workers 1` is required because session state is stored in memory.

Smoke check:

```bash
curl http://127.0.0.1:8765/healthz
curl http://127.0.0.1:8765/eval/runs
```

Expected `/eval/runs`:

```json
["v1", "v2"]
```

## CLI

Against a running backend:

```bash
uv run python -m convfinqa.serving.cli
uv run python -m convfinqa.serving.cli ask --report <id> --question "..."
uv run python -m convfinqa.serving.cli reports
```

Start the API via installed script:

```bash
uv run convfinqa-serve
```

## API Evaluation

Drive the running FastAPI server over the held-out evaluation sample:

```bash
uv run convfinqa-eval-api
PROMPTS_VERSION=v2 uv run convfinqa-eval-api
```

Outputs include `api_predictions_<version>.csv`, joined CSVs, and model comparison tables when matching DSPy/Pydantic CSVs exist.

## Prompt-Improvement Harness (s7)

Per-case **Diagnose → Route+Fix → Verify** loop over first-wrong-per-conversation cases in `pydantic_predictions_v2.csv`. Promotes verified rules into `prompts/v3_opt.py`.

### Smoke runs

```bash
# Single case, single attempt, fresh stores (fastest end-to-end smoke)
uv run python scripts/diagnose_failures.py --limit 1 --reset-rules --skip-regression

# 10 cases with up to 2 retries (retry_n=3 ⇒ 3 total attempts max)
uv run python scripts/diagnose_failures.py --limit 10 --retry-n 3 --skip-regression
```

### Diagnose-only sweep (Step 1, no fix or verify)

```bash
# Cheap router-only classification of every first-wrong case — no rule writes
uv run python scripts/diagnose_failures.py --diagnose-only --reset-rules
```

### Full corpus run

```bash
# All 95 first-wrong cases (omit --limit). Default retry_n=1.
uv run python scripts/diagnose_failures.py --reset-rules --skip-regression
```

### Post-loop stages (standalone, no harness invocation)

```bash
# Re-assemble prompts/v3_opt.py from the current rules JSONL
uv run python scripts/diagnose_failures.py --stage assemble

# Run the regression eval (stub — re-scores v3_opt vs v2; currently a placeholder)
uv run python scripts/diagnose_failures.py --stage regression
```

### Pointing at a different input or output dir

```bash
uv run python scripts/diagnose_failures.py \
  --input evaluation/pydantic_predictions_v1.csv \
  --out-dir evaluation/v1_run \
  --version v1 \
  --limit 5
```

### Env-var overrides (preferred over CLI flags for persistent config)

```bash
RETRY_N=3                              uv run python scripts/diagnose_failures.py
LM_MAX_MODEL=deepseek-chat             uv run python scripts/diagnose_failures.py --limit 1
RULES_DIR=evaluation/experimental      uv run python scripts/diagnose_failures.py --reset-rules
MAX_PRIOR_ATTEMPTS_IN_PAYLOAD=20       uv run python scripts/diagnose_failures.py
```

### All CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--input PATH` | `evaluation/pydantic_predictions_v2.csv` | Predictions CSV to read failing cases from. |
| `--out-dir PATH` | `evaluation` | Where to write `diagnostic_results_*` and `case_results_*`. |
| `--version` | `v2` | Base prompt version that v3_opt builds on. |
| `--limit N` | (none) | Truncate to the first N first-wrong cases. |
| `--diagnose-only` | off | Step 1 only — router classification, no fix, no verify, no rule writes. |
| `--stage {all,assemble,regression}` | `all` | Short-circuit to a single post-loop stage. |
| `--reset-rules` | off | Truncate `rules_<agent>_v3_opt.jsonl` AND `rule_attempts_<agent>_v3_opt.jsonl` for all four agents. |
| `--retry-n N` | `settings.retry_n` (1) | Total attempts cap per case (1..3). |
| `--force` | off | Reserved for resume semantics (currently no-op — `case_results` is overwritten). |
| `--skip-regression` | off | Skip the post-loop regression subprocess (local dev only). |
| `--verbose / -v` | off | DEBUG logging. |

### Outputs under `evaluation/`

- `diagnostic_results_v3_opt.{csv,html}` (HTML uses the same dark theme + inspector-panel viewer as predictions HTML)
- `case_results_v3_opt.jsonl`
- `rules_<agent>_v3_opt.jsonl` × 4 (verified passes — source of truth for `v3_opt.py`)
- `rule_attempts_<agent>_v3_opt.jsonl` × 4 (pass + fail history — feeds the specialist on future runs)
- `unresolved_cases_v3_opt.json`

Full spec: [`ai_specs/s7-prompt-optimisation.md`](ai_specs/s7-prompt-optimisation.md).

## DSPy / GEPA

Run DSPy and GEPA flows via:

```bash
RUN_GEPA= uv run convfinqa-optimize
RUN_GEPA=1 GEPA_MODE=smoke uv run convfinqa-optimize
RUN_GEPA=1 GEPA_MODE=real uv run convfinqa-optimize
RUN_GEPA=1 GEPA_MODE=real RESUME_GEPA=latest uv run convfinqa-optimize
RUN_GEPA=1 GEPA_NAME=gepa_real_<ts> uv run convfinqa-optimize
```

GEPA artifacts live in `runs/<gepa_name>/`; evaluation outputs live in `evaluation/`.

## Frontend

```bash
cd frontend
npm run dev
```

Vite proxies these backend prefixes and both `server.proxy` and `preview.proxy` must stay in sync:

- `/healthz`
- `/reports`
- `/sessions`
- `/eval`

## Quality Gates

```bash
uv run pytest tests/ -q
uv run ruff check src/ scripts/ tests/
uv run mypy src/convfinqa
```

Current validated state:

- `56 passed`
- Ruff clean
- mypy: 22 pre-existing errors in `pipeline/runner.py`, `serving/app.py`, `evaluation/runner.py`, `evaluation/reporting.py` (untyped functions, str|Model union access); not a regression — track and fix incrementally.
- cached eval reproduces `v1=73.0%`, `v2=77.1%`
- backend health and `/eval/runs` smoke pass under `convfinqa.serving.app:create_app`

## Notes for Contributors

- All imports go through `convfinqa.*`; no legacy root modules exist.
- Do not change `src/convfinqa/pipeline/wire_format.py` without rerunning the cached evaluation smoke.
- Do not change backend routes or SSE event shapes without updating frontend types/tests.
