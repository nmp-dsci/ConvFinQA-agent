# ConvFinQA Agent

A production gen-AI system for [ConvFinQA](https://github.com/czyssrs/ConvFinQA): multi-turn financial QA over report text and tables. A four-stage agent pipeline, plus the surfaces that make it operable — per-turn tracing, experiment tracking, a champion/challenger registry with an enforced promotion contract, an automated prompt-research loop, and a keyless public demo.

Two deployments, one build:

| | dev | demo (public) |
|---|---|---|
| Chat | live against the champion bundle | replayed from recorded conversations |
| Splits, answers, traces, experiments | live | live — same committed artifacts |
| Promote / launch research | owner-token gated | visible, inert |
| API keys present | yes | **none, by construction** |

`DEMO_MODE` is baked into the image, so no infrastructure change can turn the public URL into a billable one.

## Evaluation Results

Cached Pydantic AI evaluator across prompt versions. Reproduces offline from committed `evaluation/predictions/pydantic_predictions_<version>.csv` — no API calls when `REUSE_CACHE=1`.

### A note on what "held out" means here

The scored set is all 200 sampled conversations (770 questions), and **GEPA trained on 120 of them**. So the 770-question figure mixes conversations the optimizer saw with ones it did not, and is reported as *overall*, never as held out. The genuinely never-seen subset is 80 conversations / **309 questions**:

| version | never-seen (309 q) | overall (770 q) |
|---------|-------------------:|----------------:|
| v1      | 72.8%              | 73.0%           |
| **v2**  | **77.7%** ← champion | 77.1%         |
| v3_1    | 73.5%              | 76.2%           |

The v1 → v2 improvement is slightly *larger* on never-seen data (+4.9 pp) than the overall figure suggests (+4.2 pp); v3_1's regression is correspondingly larger. Both numbers are surfaced in the app, and split membership is inspectable under **Data & answers** — `GET /eval/splits`.

Note also that two 60/40 "train" splits exist in the codebase with the same seed but different shuffles (`data.loader.train_report_ids` uses a pandas `.sample()`; `data.loader.optimizer_split()` reproduces the DSPy backend's `random.Random(42).shuffle()`). They agree on only 78 of 120 conversations. **GEPA ran against `optimizer_split()`**, so that is the one every held-out claim is measured against.

```bash
REUSE_CACHE=1 uv run convfinqa-eval
```

```
[v1] cache hit: 200/200 conversations (770 questions) — skipping
[v1] combined accuracy: 73.0%  (562/770 questions)
Wrote evaluation/predictions/pydantic_predictions_v1.html

[v2] cache hit: 200/200 conversations (770 questions) — skipping
[v2] combined accuracy: 77.1%  (594/770 questions)
Wrote evaluation/predictions/pydantic_predictions_v2.html

[v3_1] cache hit: 200/200 conversations (770 questions) — skipping
[v3_1] combined accuracy: 76.2%  (587/770 questions)
Wrote evaluation/predictions/pydantic_predictions_v3_1.html

------------------------------------------------------------------------
Cut                      Count            v1            v2          v3_1
------------------------------------------------------------------------
Overall                    770        73.0%         77.1%         76.2%

turn_type=Number           284        85.2%         87.7%         89.8%
turn_type=Program          486        65.8%         71.0%         68.3%

conv_type=Type I           640        75.2%         78.8%         78.8%
conv_type=Type II          130        62.3%         69.2%         63.8%

question=0                 200        81.0%         82.0%         79.0%
question=1                 199        75.4%         79.4%         82.4%
question=2                 160        70.0%         75.6%         73.1%
question=3                 116        68.1%         69.8%         71.6%
question=4                  60        61.7%         75.0%         71.7%
question=5                  24        62.5%         66.7%         58.3%
question=6                  10        70.0%         90.0%         80.0%
question=7                   1         0.0%          0.0%          0.0%
------------------------------------------------------------------------
```

**v2 is the current best prompt version.** It beats v1 by +4.1 pp overall; biggest gains are on program turns (+5.2 pp), Type II conversations (+6.9 pp), and deeper turns (`question=4`: +13.3 pp, `question=6`: +20.0 pp).

**v3_1 (the first s7 harness output) regressed by −0.9 pp against the v2 baseline it was optimising** — 76.2% vs 77.1%. The regression is not uniform, and the split is the useful signal:

| Cut | v2 → v3_1 | Read |
|---|---:|---|
| `turn_type=Number` | **+2.1 pp** (87.7 → 89.8) | Retrieval-shaped rules landed. |
| `turn_type=Program` | **−2.7 pp** (71.0 → 68.3) | Program-construction rules hurt more than they helped. |
| `conv_type=Type II` | **−5.4 pp** (69.2 → 63.8) | Worst cut; hybrid multi-hop chains are the most brittle under added rules. |
| `conv_type=Type I` | 0.0 pp (78.8 → 78.8) | Flat. |

The likely cause is the shape of the rule store: 24 of the 39 promoted rules target the `preprocess` agent, so v3_1 loads the program-decomposition prompt with case-specific rules that generalise poorly to unseen conversations. Rules verify against the single case that produced them (turns `0..k`), which does not gate against collateral damage elsewhere in the sample.

This result is what motivated [`ai_specs/s8-optimisation-testing.md`](ai_specs/s8-optimisation-testing.md) — a bench comparing three alternative optimisation techniques (TextGrad, ProTeGi, PromptWizard) against the s7 baseline. **s8 is specified but not implemented.**

### Version differences

- **v1** — Original baseline. Compact instructions per agent: classify (triage), decompose (preprocess), look up (retriever), compute (calculator). Same four-stage pipeline as v2; differences are purely in the system prompts. See `src/convfinqa/prompts/v1.py`.
- **v2** — GEPA-optimised prompts produced by a full real run (`gepa_real_20260502_005251`). Prompts are longer and more explicit: worked examples, explicit percentage convention (`multiply(..., 100)` outermost), clearer Type I vs Type II conversation guidance, and tighter sub-question specification rules (year + entity + metric). Pipeline structure and tools are unchanged from v1. See `src/convfinqa/prompts/v2.py`.
- **v3_1, v3_2, v3_N** *(generated)* — Each is a variant produced by the s7 prompt-improvement harness. `v3_1` assembles from `v2` baseline + `rules_<agent>_v3_1.jsonl`; `v3_2` assembles from `v3_1` baseline + `rules_<agent>_v3_2.jsonl`; and so on. Pass `--variant <name>` to the harness to start a new variant; chain via `--prompts-version <prev> --variant <next>`. Never hand-edited. See §Prompt-Improvement Harness (s7) below.
  - **`v3_1`** — Built from `v2` + 39 verified rules (`preprocess` 24, `retriever` 7, `calculator` 5, `triage` 3) over the 95 first-wrong-per-conversation cases. Scored 76.2%, a −0.9 pp regression vs v2. **Not promoted.**
  - **`v3_2`** — *In progress, incomplete.* Round 2 ran diagnose/propose over 94 cases (`case_results_v3_2.jsonl`, `diagnostic_results_v3_2.{csv,html}`) but promoted no rules — the `rules_*_v3_2.jsonl` stores are empty and `prompts/v3_2.py` was never assembled, so there is no v3_2 to evaluate. Resuming means re-running the full loop with `--prompts-version v3_1 --variant v3_2` and letting Step 3 (verify) execute.

### Known state of the s7 case caches

`case_results_<variant>.jsonl` is **overwritten** by each run rather than merged (`--force` is a documented no-op). Both committed `case_results` files were last written by a diagnose/propose pass, so every record carries `verify_result: null` and `resolved: false` — that reflects the last pass, not the full v3_1 run. The verification metadata that actually matters survives in the rules store: every rule in `rules_<agent>_v3_1.jsonl` carries `verified_at` and the `verified_on` case it was verified against.

## Production surfaces

The frontend is the operator console, not just a chat window. Six tabs, all
reading the same backend:

| Tab | What it answers |
|---|---|
| **Chat** | Pick a filing, ask freely or step through the dataset's own questions, watch the four stages stream. |
| **Data & answers** | Which conversations the optimizer saw, and every question with gold beside each version's answer — filterable to just the turns where versions disagree. |
| **Traces** | Every turn the system has answered, stage by stage: inputs, outputs, reasoning, tool loop, tokens, latency, gold comparison. |
| **Experiments** | Every eval / GEPA / research run, the accuracy trend, and a question-by-question diff of any two versions with the pass→fail flip list. |
| **Research** | Launch an s7 round or a GEPA smoke run and watch it stream; browse the rules each round promoted. |
| **Eval** | The per-slice accuracy tables. |

## Tracking, versioning and promotion

A "model version" here is a **bundle**, not a checkpoint — prompts + GEPA
overlay + both model ids + dataset hash + code SHA, versioned together and
stamped on every run, CSV and serving session (`convfinqa.tracking.bundle`).

The promotion contract, enforced in `tracking/registry.py` and
`tracking/comparator.py`:

1. **Every bundle is registered** — hand-edited prompts, an s7 round, a GEPA run,
   no difference. Specs are never deleted, so failed challengers keep their
   evidence as long as champions do.
2. **It is evaluated** on the never-seen split.
3. **The comparator decides.** First version is champion by default; after that,
   promotion needs overall accuracy ≥ champion **and no per-question pass→fail
   flips**. The second condition is the load-bearing one — a change that fixes
   twelve number turns and breaks nine program turns nets out positive and is
   still a regression.
4. **Promotion is append-only** — an alias move recorded with timestamp,
   comparator verdict and the runs behind it.

```bash
uv run convfinqa-mlflow status                    # config, aliases, versions
uv run convfinqa-mlflow compare v2 v3_1           # exit 1 if not promotable
uv run convfinqa-mlflow promote v3_1              # refused unless it passes
uv run convfinqa-mlflow backfill                  # rebuild history from git
uv run convfinqa-mlflow snapshot                  # export for the demo image
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Logging lives *inside* the eval/GEPA/s7 runners rather than beside them, so a
run cannot happen without being recorded — an experiment history with silent
gaps is worse than none.

## Demo mode

```bash
DEMO_MODE=1 uv run python -m uvicorn convfinqa.serving.app:create_app \
  --factory --workers 1 --port 8765          # works with no API key at all
uv run convfinqa-demo-pack --n 8             # rebuild the recorded pack
```

The pack is reconstructed from committed prediction CSVs — every row already
carries the full per-stage IO — so recording it costs **zero API calls**.
Replay emits the same SSE events as the live path, paced so a real 30–60 s turn
plays in about four seconds. Below the fuzzy-match threshold it declines rather
than serving the nearest recording: confidently returning another filing's
number would be the worst failure a system about numerical accuracy could have.

## Container & deploy

```bash
docker compose up demo     # exactly what ships: no keys, replayed chat
docker compose up dev      # same image, live model, your key
./scripts/demo_smoke.sh http://localhost:8080
```

One container serves the API and the built SPA from the same origin. Push to
`main` → CI must pass → `deploy-aws.yml` assumes an AWS role via GitHub OIDC
(no stored keys), builds and pushes to ECR, App Runner auto-deploys `:latest`,
Terraform reconciles, and the smoke test asserts `mode=demo`, a registered
champion, the committed evidence, and a 403 on an admin write.

Infrastructure is ECR + App Runner + a 5xx alarm — no VPC, no database, no
Secrets Manager. Measured 225 MiB RSS at rest, 315 MiB with every prediction CSV
cached, so the 1 GB instance is the honest floor. Roughly $5–15/month; demo mode
is the real cost control, since the public deployment performs no inference.


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
| `src/convfinqa/prompts/` | Versioned prompt modules (`v1`, `v2`, generated `v3_1`, `v3_2`, …). |
| `src/convfinqa/diagnosis/` | s7 diagnose → route+fix → verify harness (per-case prompt improvement). CLI implementation lives in `diagnosis/cli.py`. |
| `src/convfinqa/llm.py` | **The single LLM choke point.** Every model is built here; retry/timeout policy and the demo gate live here and nowhere else. |
| `src/convfinqa/tracking/` | Bundle fingerprint, trace store, MLflow logging, comparator, registry, backfill, snapshot, CI gate. |
| `src/convfinqa/serving/` | FastAPI app, routers (`chat`, `evaluation`, `traces`, `admin`), session store, limits, research runner. |
| `src/convfinqa/serving/demo_pack/` | Recorded conversations + replay, so the keyless demo streams like the live app. |
| `src/convfinqa/optimization/` | GEPA and prompt optimisation entry points. |
| `scripts/` | Installed command entry points. `diagnose_failures.py` is a thin shim over `convfinqa.diagnosis.cli:main`. |
| `ai_specs/` | Design specs — `s7-prompt-optimisation.md` (implemented), `s8-optimisation-testing.md` (specified, not implemented). |
| `frontend/` | React/Vite UI. |
| `evaluation/predictions/` | Cached prediction CSVs + dark-themed HTML reports + joined CSVs. Tracked in git so accuracy reproduces offline. |
| `evaluation/diagnostics/` | s7 harness stores (`rules_*`, `rule_attempts_*`, `diagnostic_results_*`, …). Tracked in git. |
| `runs/` | GEPA optimization artifacts (`optimized_runner.json`). Tracked in git so prior runs are usable on any clone. |
| `evaluation/registry.json`, `evaluation/mlflow_snapshot.json` | Bundle registry + exported experiment history. Tracked, and baked into the demo image. |
| `infra/terraform/` | `bootstrap/` (run once: the OIDC deploy role) and `demo/` (ECR + App Runner + alarm). |
| `Dockerfile`, `docker-compose.yml` | The demo image, and the local dev/demo toggle. |
| `.dspy_cache/` | DSPy LM response cache (~366 MB). Gitignored; rsync between machines for warm scoring. |
| `mlruns/`, `.traces/` | Local MLflow store and trace DB. Gitignored — the committed snapshot is what ships. |

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
| `v3_1` | `76.2%` (`587/770`) |

The sweep auto-discovers every `prompts/v*.py` module via `prompts.latest_all()`, so a new variant appears in the comparison table with no registration step.

Useful variants:

```bash
PROMPTS_VERSION=v2 uv run convfinqa-eval
REUSE_CACHE=0 uv run convfinqa-eval
```

Outputs are written under `evaluation/predictions/`, for example:

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

Expected `/eval/runs` (auto-discovered from `prompts/`, so it grows as variants are added):

```json
["v1", "v2", "v3_1"]
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

Per-case **Step 1 — Diagnose → Step 2 — Propose → Step 3 — Verify** loop over first-wrong-per-conversation cases in `pydantic_predictions_v2.csv`. Promotes verified rules into `prompts/<variant>.py` (default `v3_1`).

Two independent version axes control input/output:
- **`--prompts-version <name>`** — *input* prompts the harness reads — the baseline being optimised. Default `settings.prompts_version` (from `PROMPTS_VERSION` env) or `v2`. Any existing module in `prompts/` is valid (`v1`, `v2`, `v3_1`, …). Same name as `convfinqa-eval`'s `PROMPTS_VERSION` so the version identifier is consistent end-to-end.
- **`--variant <name>`** — *output* variant name (default `v3_1`). Controls every artifact suffix (`rules_<agent>_<variant>.jsonl`, `case_results_<variant>.jsonl`, `diagnostic_results_<variant>.{csv,html}`, `unresolved_cases_<variant>.json`) AND the name of the generated prompts module (`prompts/<variant>.py`).

To iterate, pass a **new** `--variant` each round and chain its `--prompts-version` to the previous variant.

### Run modes

Three operator modes, each a strict subset of the next. All three steps are cached in `case_results_v3_1.jsonl` — a `--diagnose-only` pass primes Step 1, `--propose-fix` adds Step 2, a full run adds Step 3. Subsequent runs reuse cached steps for free; with all three caches hot, a re-run makes **zero LLM calls and zero verify replays**.

| Mode | Flag | Steps | First-run cost per non-ambiguous case | All-cache-hit cost | Writes rule store? | Regenerates `v3_1.py`? |
|---|---|---|---|---|---|---|
| Diagnose-only | `--diagnose-only` | Step 1 | 1 router LLM call | 0 | No | No |
| Propose-fix | `--propose-fix` | Step 1 + Step 2 | 1 router + 1 specialist LLM call | 0 | No | No |
| Full | *(default)* | Step 1 + Step 2 + Step 3 | router + specialist + `k+1` turn-replays | 0 (bookkeeping only, dedup'd) | Yes (on passing verify) | Yes (post-loop) |

`--diagnose-only` and `--propose-fix` are mutually exclusive. Disable specific step caches with `--no-diagnose-cache` / `--no-propose-cache` / `--no-verify-cache`.

The flags below show the explicit `--prompts-version v2 --variant v3_1` form. These are the defaults, so they can be omitted on first iteration — but stating them makes the optimisation direction obvious (`v2 → v3_1`) and matches what subsequent iterations need (`--prompts-version v3_1 --variant v3_2`).

```bash
# Step 1 only — cheap router classification, no propose, no verify
uv run python scripts/diagnose_failures.py \
  --prompts-version v3_1 --variant v3_2 \
  --diagnose-only --reset-rules

# Step 1 + Step 2 — also propose a fix per case, still skip verify.
# Reuses Step 1 diagnose cache if present, so this is "specialist LLM only".
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --propose-fix

# Full loop — Step 1 + Step 2 + Step 3, promotes verified rules into the store
# AND writes prompts/v3_1.py.
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --reset-rules --skip-regression
```

### Iterating with variants

Each variant is a self-contained universe of artifacts (rules JSONL, attempts JSONL, case_results, diagnostic_results, prompts module). Names must be Python-importable — use underscores, not dots (`v3_2`, not `v3.2`).

```bash
# Iteration 1: build v3_1 from v2 baseline (this is the default behaviour).
uv run python scripts/diagnose_failures.py --prompts-version v2 --variant v3_1

# Iteration 2: build v3_2 from v3_1 baseline.
# - Loads prompts/v3_1.py as the input.
# - Writes prompts/v3_2.py + evaluation/*_v3_2.* artifacts.
# - Step caches are per-variant — v3_2 starts fresh, v3_1's cache is untouched.
uv run python scripts/diagnose_failures.py --prompts-version v3_1 --variant v3_2

# Iteration 3 (and beyond):
uv run python scripts/diagnose_failures.py --prompts-version v3_2 --variant v3_3
```

- Each variant lives alongside the others — `v3_1` and `v3_2` artifacts never collide.
- `--reset-rules` only truncates the *current* variant's stores; other variants are untouched.
- `--stage assemble --variant <name>` re-assembles `prompts/<name>.py` from that variant's rules JSONL.
- Never use the same name for `--prompts-version` and `--variant` — it would double-apply the variant's rules onto its own baseline.

### Smoke runs

```bash
# Single case, single attempt, fresh stores (fastest end-to-end smoke)
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --limit 1 --reset-rules --skip-regression

# 10 cases with up to 2 retries (retry_n=3 ⇒ 3 total attempts max)
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --limit 10 --retry-n 3 --skip-regression
```

### Full corpus run

```bash
# All 95 first-wrong cases (omit --limit). Default retry_n=1.
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --reset-rules --skip-regression
```

### Typical workflow

All four commands target the same `(--prompts-version v2, --variant v3_1)` universe — step caches in `case_results_v3_1.jsonl` carry across the four calls, so each step pays its cost once.

```bash
# 1. Cheap router sweep over every first-wrong case (populates Step 1 cache)
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --diagnose-only

# 2. Review router output in evaluation/diagnostics/diagnostic_results_v3_1.html

# 3. Generate proposed fixes using cached diagnoses (zero router cost,
#    one specialist call per non-ambiguous case). Populates Step 2 cache.
#    Group C verify columns render as `—`.
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --propose-fix

# 4. Review proposed system_prompt rules in the HTML, then commit to verify.
#    Full loop reuses Steps 1+2 caches; only Step 3 (verify replay) is live.
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --skip-regression

# 5. Re-run is free — all three step caches hit, dedup'd bookkeeping only.
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --skip-regression

# To re-execute a specific step (e.g. after editing v2 prompts), opt out:
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --no-verify-cache --skip-regression
```

### End-to-end optimisation loop (cross-app)

The full loop alternates between the s7 harness (which proposes + verifies rules to improve a baseline) and `convfinqa-eval` (which measures the result against the held-out test sample). Because both scripts use the same `PROMPTS_VERSION` identifier and the prompts loader auto-discovers `v\d+(_\d+)?` modules, the version name carries through every step:

```bash
# Round 1 — improve v2 → produce v3_1
PROMPTS_VERSION=v2 VARIANT=v3_1 uv run python scripts/diagnose_failures.py
#   Writes: prompts/v3_1.py, rules_<agent>_v3_1.jsonl, case_results_v3_1.jsonl,
#           diagnostic_results_v3_1.{csv,html}, unresolved_cases_v3_1.json

# Round 1 — evaluate v3_1 vs v1/v2
PROMPTS_VERSION=v3_1 uv run convfinqa-eval
#   Writes: pydantic_predictions_v3_1.csv + dark-theme HTML
#   Prints: comparison table with v1, v2, v3_1 side-by-side

# Round 2 — improve v3_1 → produce v3_2
PROMPTS_VERSION=v3_1 VARIANT=v3_2 uv run python scripts/diagnose_failures.py
#   Loads prompts/v3_1.py (which already embeds v3_1's verified rules) as the baseline.
#   Writes a fresh v3_2 universe of artifacts.

# Round 2 — evaluate v3_2 vs all priors
PROMPTS_VERSION=v3_2 uv run convfinqa-eval
#   Comparison table now shows v1, v2, v3_1, v3_2.

# Continue iterating: --prompts-version v3_2 --variant v3_3, etc.
```

Equivalent using flags instead of env vars:

```bash
uv run python scripts/diagnose_failures.py --prompts-version v2  --variant v3_1
PROMPTS_VERSION=v3_1 uv run convfinqa-eval

uv run python scripts/diagnose_failures.py --prompts-version v3_1 --variant v3_2
PROMPTS_VERSION=v3_2 uv run convfinqa-eval
```

**Why this works** — the version name is the single source of truth across the app:

| Tool | Reads version from |
|---|---|
| `diagnose_failures.py` (input baseline) | `--prompts-version` flag → `settings.prompts_version` (`PROMPTS_VERSION` env) → `v2` fallback |
| `diagnose_failures.py` (output variant) | `--variant` flag → `settings.variant` (`VARIANT` env) → `v3_1` fallback |
| `convfinqa-eval` (focus + comparison) | `settings.prompts_version` (`PROMPTS_VERSION` env); always includes auto-discovered priors in the comparison table |
| `convfinqa-eval-api` (backend driver) | `PROMPTS_VERSION` env at process start |
| `convfinqa.prompts.load(name)` | Dynamic — any `prompts/<name>.py` module is loadable |
| `convfinqa.prompts.latest_all()` | Auto-discovers all `^v\d+(_\d+)?$` modules; powers the eval sweep |

Drop a new variant module into `prompts/` (typically by running the harness) and every script picks it up automatically. No registration, no aliases, no hardcoded lists.

### Post-loop stages (standalone, no harness invocation)

Both stages honour `--prompts-version` (input baseline) and `--variant` (output variant), so re-assembling a specific variant from hand-edited rules is just:

```bash
# Re-assemble prompts/v3_1.py from rules_<agent>_v3_1.jsonl
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --stage assemble

# Re-assemble prompts/v3_2.py from rules_<agent>_v3_2.jsonl (chained on v3_1)
uv run python scripts/diagnose_failures.py \
  --prompts-version v3_1 --variant v3_2 \
  --stage assemble

# Run the regression eval (stub — re-scores v3_1 vs v2; currently a placeholder)
uv run python scripts/diagnose_failures.py \
  --prompts-version v2 --variant v3_1 \
  --stage regression
```

### Pointing at a different input or output dir

```bash
uv run python scripts/diagnose_failures.py \
  --input evaluation/predictions/pydantic_predictions_v1.csv \
  --out-dir evaluation/v1_run \
  --prompts-version v1 --variant v1_opt \
  --limit 5
```

### Env-var overrides (preferred over CLI flags for persistent config)

```bash
RETRY_N=3                              uv run python scripts/diagnose_failures.py
LM_MAX_MODEL=deepseek-v4-flash         uv run python scripts/diagnose_failures.py --limit 1
RULES_DIR=evaluation/experimental      uv run python scripts/diagnose_failures.py --reset-rules
MAX_PRIOR_ATTEMPTS_IN_PAYLOAD=20       uv run python scripts/diagnose_failures.py
PROMPTS_VERSION=v3_1 VARIANT=v3_2      uv run python scripts/diagnose_failures.py
```

### All CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--input PATH` | `evaluation/predictions/pydantic_predictions_v2.csv` | Predictions CSV to read failing cases from. |
| `--out-dir PATH` | `evaluation` | Where to write `diagnostic_results_*` and `case_results_*`. |
| `--prompts-version` | `settings.prompts_version` (from `PROMPTS_VERSION` env) or `v2` | *Input* prompts version — the baseline being optimised. Loaded via `prompts.load(prompts_version)`. Use `v3_1`, `v3_2`, … to chain on top of a prior variant. Same name as `convfinqa-eval`'s `PROMPTS_VERSION`. |
| `--variant` | `v3_1` (`settings.variant`) | *Output* variant name. Controls the suffix on every artifact AND the generated prompts module name. Pass a new name each iteration (e.g. `--variant v3_2`). |
| `--limit N` | (none) | Truncate to the first N first-wrong cases. |
| `--diagnose-only` | off | Step 1 only — router classification, no propose, no verify, no rule writes. Mutually exclusive with `--propose-fix`. |
| `--propose-fix` | off | Step 1 + Step 2 — propose a fix per case, skip Step 3 (verify). No rule writes, no v3_1 regeneration. Group C verify columns render as `—`. Mutually exclusive with `--diagnose-only`. |
| `--no-diagnose-cache` | off | Ignore Step 1 (Diagnose) cache; re-call the router for every case. |
| `--no-propose-cache` | off | Ignore Step 2 (Propose) cache; re-call the specialist Propose LLM for every attempt. |
| `--no-verify-cache` | off | Ignore Step 3 (Verify) cache; re-run verify replays for every attempt. |
| `--stage {all,assemble,regression}` | `all` | Short-circuit to a single post-loop stage. |
| `--reset-rules` | off | Truncate `rules_<agent>_v3_1.jsonl` AND `rule_attempts_<agent>_v3_1.jsonl` for all four agents. |
| `--retry-n N` | `settings.retry_n` (1) | Total attempts cap per case (1..3). |
| `--force` | off | Reserved for resume semantics (currently no-op — `case_results` is overwritten). |
| `--skip-regression` | off | Skip the post-loop regression subprocess (local dev only). |
| `--verbose / -v` | off | DEBUG logging. |

### Outputs under `evaluation/diagnostics/`

`<variant>` below is the value of `--variant` (default `v3_1`). Each variant produces its own independent set of files.

- `diagnostic_results_<variant>.{csv,html}` (HTML uses the same dark theme + inspector-panel viewer as predictions HTML)
- `case_results_<variant>.jsonl` (also the 3-step cache file — see §Run modes)
- `rules_<agent>_<variant>.jsonl` × 4 (verified passes — source of truth for `prompts/<variant>.py`)
- `rule_attempts_<agent>_<variant>.jsonl` × 4 (pass + fail history — feeds the specialist on future runs)
- `unresolved_cases_<variant>.json`
- `src/convfinqa/prompts/<variant>.py` (generated; never hand-edited)

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

GEPA artifacts live in `runs/<gepa_name>/`; prediction outputs live in `evaluation/predictions/` and s7 diagnostics in `evaluation/diagnostics/`.

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
uv run ruff check --no-fix src scripts tests
uv run ruff format --check src scripts tests
uv run mypy
uv run pytest tests -q
uv run python -m convfinqa.tracking.gate      # offline eval-regression gate
cd frontend && npm run typecheck && npm run test:unit && npm run build
```

Every one of these runs in CI on every pull request, plus a Docker build and
`terraform fmt`/`validate`. Current state — all green:

| Gate | State |
|---|---|
| ruff check + format | clean |
| mypy (strict-ish, 71 files) | clean |
| pytest | **101 passed**, zero network calls, no API key required |
| frontend typecheck + vitest + build | clean, 11 unit tests |
| eval-regression gate | passes; champion `v2` at 77.14% against a 76.64% floor |

The **eval-regression gate** is the load-bearing one. Because prediction CSVs
are committed, it re-scores them deterministically with no API calls, and fails
the build if the champion drops below its registered floor, if any CSV's
`correct` column stops agreeing with re-scoring its own answers, or — for a
registered challenger — it prints the exact questions that flipped pass→fail.

Two notes on the mypy config, since both were real bugs:
`packages = ["src"]` pointed at a directory that is not a package, so every
`convfinqa.*` import resolved to `Any` and the strict settings applied to almost
nothing; and `Settings` boots with no key, so CI needs no placeholder secret —
the absence of a key is itself part of what the suite verifies.

## Notes for Contributors

- All imports go through `convfinqa.*`; no legacy root modules exist.
- Do not change `src/convfinqa/pipeline/wire_format.py` without rerunning the cached evaluation smoke.
- Do not change backend routes or SSE event shapes without updating frontend types/tests.
