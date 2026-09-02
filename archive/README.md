# archive/

Retired experiment by-products, moved out of the live tree on 2026-08-31 so the
directories the code reads (`runs/`, `evaluation/predictions/`,
`evaluation/diagnostics/`) hold only what something still depends on. Everything
here was moved with `git mv`, so history is intact and any file can be restored
with `git mv archive/<path> <path>`.

Nothing in `src/`, `tests/`, CI, or the Docker image reads from this directory.
`.dockerignore` excludes it.

## What moved, and what stayed

| Archived here | Why it is an experiment by-product | What stayed in place (still load-bearing) |
|---|---|---|
| `runs/gepa_smoke_20260429_204159/` — `gepa_logs/` (iteration dumps + `gepa_state.bin`), every `*_predictions*.csv`, `parity_report*.csv`, `accuracy_by_*.csv`, `model_accuracy_comparison.csv` | The smoke run was a ~30 min wiring check, never a transferable optimisation (`AGENTS.md`). Its scoring CSVs were superseded by `evaluation/predictions/`. | `runs/gepa_smoke_20260429_204159/optimized_runner.json` + `config.json` — the GEPA prompt, still re-scorable with `RUN_GEPA=1 GEPA_NAME=gepa_smoke_20260429_204159 uv run convfinqa-optimize`, and still backfilled by `tracking/backfill.py::backfill_gepa`. |
| `runs/gepa_real_20260502_005251/dspy_gepa_logs/` — 25 per-task iteration JSONs + `gepa_state.bin` | Optimiser internals. `RESUME_GEPA` on this run would need `gepa_state.bin` back, but the run is complete and is the production overlay, so resuming it is not a path anyone takes. | `dspy_optimized_runner.json` (the GEPA prompt `pipeline/prompts_loader.py` defaults to), `config.json`, `dspy_summary.json`, `dspy_gepa_stats.json`. |
| `evaluation/predictions/` — `dspy_predictions_v2{,_joined}.csv`, `api_predictions_v2.csv` (2 rows), `parity_report_v2.csv`, `model_accuracy_comparison_v2.csv` | The s3–s5 DSPy-vs-Pydantic-vs-API parity experiment. `serving/evaldata.py` tolerates their absence (`load_joined` returns `None`), and the CI gate only re-scores `pydantic_predictions_*`. | `pydantic_predictions_v{1,2,3_1}{,_joined}.csv` + `.html` — the REUSE_CACHE cache, the registry's evidence, the demo pack's source, and what `tracking/gate.py` re-scores on every PR. |
| `evaluation/diagnostics/*_v3_2.*` — `case_results`, `diagnostic_results.{csv,html}`, `unresolved_cases`, four empty `rule_attempts_*`, one empty `rules_preprocess` | The second s7 round ran diagnose/propose over 94 cases but promoted zero rules; `prompts/v3_2.py` was never assembled, so there is no v3_2 to evaluate. | The complete `_v3_1` universe — the `rules_<agent>_v3_1.jsonl` stores are the source `prompts/v3_1.py` is generated from. |

## Not archived, deliberately

- `scripts/` — every file is either a `[project.scripts]` console entry point in
  `pyproject.toml` (`convfinqa-eval`, `-eval-api`, `-optimize`, `-serve`), copied
  into the image by `Dockerfile`, and linted by CI — or a deploy/smoke script the
  AWS workflow runs. Retiring the GEPA / API-parity entry points is a code change
  (pyproject, tests, README), not a file move; see the s04 eval-loop plan.
- `ai_specs/` — the PRPs are design history, not experiment output.
- `evaluation/mlflow_snapshot.json` still lists the `gepa_smoke_…` and `s7-v3_2`
  runs. That is correct: they happened. Regenerate with `convfinqa-mlflow snapshot`
  only when the tracking store is next rebuilt.
