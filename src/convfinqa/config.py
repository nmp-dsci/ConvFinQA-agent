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
    deepseek_api_key: SecretStr
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
    lm_max_model: str = "deepseek-reasoner"
    rules_dir: Path = Field(default_factory=lambda: Path("evaluation").resolve())
    retry_n: int = 1
    max_prior_attempts_in_payload: int = 50

    # ---- FastAPI / frontend ----------------------------------------------
    # Comma-separated CORS allow-list. Both localhost and 127.0.0.1 are needed
    # for the various ways the frontend dev server / preview can be reached.
    frontend_origins: str = (
        "http://localhost:5173,http://localhost:4173,"
        "http://127.0.0.1:5173,http://127.0.0.1:4173,http://127.0.0.1:8765"
    )


settings = Settings()  # singleton import-time instance
