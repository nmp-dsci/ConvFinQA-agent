"""Centralized configuration for the ConvFinQA project.

All environment variables consumed by `pydantic_agent.py`, `api_eval.py`,
`dspy_agent.py`, and `app.py` are declared here on a single `Settings`
class — validated, typed, and overridable via `~/.env` or process env vars.

Usage
-----
    from config import settings
    if settings.reuse_cache:
        ...
    api_key = settings.deepseek_api_key.get_secret_value()

Boolean env vars accept the conventional set {1, true, yes, on} (case-insensitive)
as truthy, with the inverse parsed as falsy by pydantic-settings.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from ~/.env *before* Settings() is instantiated.
# Several upstream libraries (notably dspy.LM) read DEEPSEEK_API_KEY directly
# from os.environ at construction time, so the env must be loaded first.
load_dotenv(Path.home() / ".env")

# ---- Artifact directory layout (single source of truth) -------------------
# All cached evaluation artifacts live under evaluation/, split by kind so the
# directory stops sprawling as new versions accumulate:
#   predictions/ — prediction CSVs + HTML + joined CSVs. Served by the API and
#                  consumed by REUSE_CACHE; this is the on-disk reproducibility
#                  cache that v1/v2/... accuracy reproduces from.
#   diagnostics/ — s7 harness stores: rules_*, rule_attempts_*, case_results_*,
#                  diagnostic_results_*, unresolved_cases_*.
# Anchored to the repo root so paths are stable regardless of the process cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = REPO_ROOT  # backwards-compatible alias
EVAL_ROOT = REPO_ROOT / "evaluation"
PREDICTIONS_DIR = EVAL_ROOT / "predictions"
DIAGNOSTICS_DIR = EVAL_ROOT / "diagnostics"
# Dataset and optimizer artifacts are repo content, not cwd-relative content.
# Anchoring them here is what lets the API, the CLIs, and the container all
# resolve the same files regardless of which directory the process started in.
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
# MLflow file store (dev) and the committed export the demo reads instead.
MLRUNS_DIR = REPO_ROOT / "mlruns"
MLFLOW_SNAPSHOT_PATH = EVAL_ROOT / "mlflow_snapshot.json"
# Run-trace store: per-turn stage IO for every serving and eval turn.
TRACES_DIR = REPO_ROOT / ".traces"


class Settings(BaseSettings):
    """Single source of truth for environment-driven configuration.

    Field name → env var mapping is automatic via pydantic-settings (uppercase).
    Add a new env var by declaring a typed field here; remove an old one by
    deleting the field and grepping for stragglers.
    """

    model_config = SettingsConfigDict(
        env_file=Path.home() / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated env vars in the shell / .env
        case_sensitive=False,
    )

    # ---- API keys ---------------------------------------------------------
    # Optional at boot, required at call time. A fresh clone — and the demo
    # container, which has no keys by construction — must import the whole
    # package, run the test suite, and serve every non-LLM route without one.
    # `require_deepseek_api_key()` is the single place that demands it, and it
    # raises only when an LLM call is actually about to happen.
    deepseek_api_key: SecretStr | None = None
    logfire_token: SecretStr | None = None

    # ---- Prompts ----------------------------------------------------------
    # None → auto-detect highest version in prompts/. Otherwise pin (e.g. "v2").
    prompts_version: str | None = None
    # DSPy runner JSON overlay path; takes precedence over prompts_version.
    prompts_overlay_path: Path | None = None

    # ---- Evaluation -------------------------------------------------------
    # When True (default), skip already-scored conversations and merge cached + new.
    reuse_cache: bool = True
    # Async concurrency for api_eval.py and pydantic_agent.py.
    max_concurrency: int = 8

    # ---- DSPy / GEPA ------------------------------------------------------
    # Selects an input/output optimization run dir under runs/<gepa_name>/.
    # `None` means "unset" — consumers apply their own default:
    #   pydantic_agent.py defaults to DEFAULT_GEPA_NAME (the v2 run).
    #   dspy_agent.py interprets unset as "do a fresh optimization run".
    gepa_name: str | None = None
    # "smoke" (~30 min) or "real" (5–9 hr).
    gepa_mode: str = "smoke"
    # Set False to skip GEPA and only run baseline evaluation.
    run_gepa: bool = True
    # Resume target — path or "latest".
    resume_gepa: str | None = None
    # Output suffix label for dspy_predictions_<version>.csv after a GEPA run.
    version: str = "v_pending"
    # Repo-local DSPy LM cache directory.
    dspy_cachedir: Path = Field(default_factory=lambda: Path(".dspy_cache").resolve())

    # ---- Diagnosis harness (s7) -------------------------------------------
    # DeepSeek flagship model for the s7 diagnostic router + 4 specialist Fix
    # agents. Default = `deepseek-v4-pro` (1.6T/49B MoE) for highest-quality
    # reasoning during prompt optimisation. Override via `LM_MAX_MODEL=...`
    # to swap to `deepseek-v4-flash` (cheaper) or any DeepSeek model id.
    # Migrated from the legacy `deepseek-reasoner` alias which deprecates
    # 2026-07-24; the dspy/GEPA backend uses the same v4-pro identifier.
    lm_max_model: str = "deepseek-v4-pro"
    rules_dir: Path = Field(default_factory=lambda: DIAGNOSTICS_DIR)
    retry_n: int = 1
    max_prior_attempts_in_payload: int = 50
    # Output variant for the s7 harness. Controls the suffix used in every
    # artifact name (rules_<agent>_<variant>.jsonl, case_results_<variant>.jsonl,
    # diagnostic_results_<variant>.{csv,html}, etc.) AND the name of the
    # generated prompts module (src/convfinqa/prompts/<variant>.py). Operators
    # iterate by passing --variant v3_1, v3_2, ... or VARIANT=v3_1.
    # The corresponding base prompts version (input) is controlled separately
    # by --version / PROMPTS_VERSION — e.g. to chain v3_2 on top of v3_1:
    #   uv run python scripts/diagnose_failures.py --version v3_1 --variant v3_2
    variant: str = "v3_1"

    # ---- FastAPI / frontend ----------------------------------------------
    # Comma-separated CORS allow-list. Both localhost and 127.0.0.1 are needed
    # for the various ways the frontend dev server / preview can be reached.
    # In the container the SPA is same-origin, so this list only matters in dev.
    frontend_origins: str = (
        "http://localhost:5173,http://localhost:4173,"
        "http://127.0.0.1:5173,http://127.0.0.1:4173,http://127.0.0.1:8765"
    )

    # ---- Demo mode --------------------------------------------------------
    # The single flag that separates the public deployment from dev. When set,
    # every LLM call is refused at one choke point (`convfinqa.llm`) and chat
    # is served from the recorded demo pack instead. Read-only surfaces —
    # reports, splits, answers, traces, experiments — stay genuinely live.
    demo_mode: bool = False
    # Owner token for admin writes (promotion, research launches). Unset means
    # admin writes are refused outright rather than left open.
    owner_token: SecretStr | None = None

    # ---- Serving limits ---------------------------------------------------
    # Cheapest check first: a global in-flight cap, then a per-IP window.
    # In-memory is *correct* here — App Runner runs this at max-size 1.
    max_inflight_turns: int = 4
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    # Hard ceiling on a single LLM call, and how many times to retry it.
    llm_timeout_seconds: float = 120.0
    llm_max_attempts: int = 4

    # ---- Tracking ---------------------------------------------------------
    # `file:` store in dev; the demo reads the committed snapshot instead.
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "convfinqa"
    registered_model_name: str = "convfinqa-pipeline"
    # Persist per-stage IO for every serving turn. Off in tests.
    trace_capture_enabled: bool = True

    def require_deepseek_api_key(self) -> str:
        """Return the DeepSeek key, raising a clear error when it is absent.

        Called at LLM-construction time, never at import time — that ordering is
        what keeps a keyless clone (and the demo image) fully importable.
        """
        if self.deepseek_api_key is None:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. It is required for any call that "
                "reaches the model. Set it in ~/.env or the process environment; "
                "read-only routes, the test suite, and DEMO_MODE=1 need no key."
            )
        return self.deepseek_api_key.get_secret_value()


settings = Settings()  # singleton import-time instance
