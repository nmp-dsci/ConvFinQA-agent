"""Versioned pipeline system prompts.

Each version is a sibling Python module exporting four module-level constants:

    TRIAGE_PROMPT
    PREPROCESS_PROMPT
    RETRIEVER_PROMPT
    CALCULATOR_PROMPT

`load(version)` returns the four prompts as a dict keyed by short agent name
(`triage`, `preprocess`, `retriever`, `calculator`) — the shape that
`pydantic_agent.PROMPTS` expects.

Add a new version (e.g. `v3`) by dropping a `prompts/v3.py` file with the same
four constants, then setting `PROMPTS_VERSION=v3` in the env (or passing it
through as the default in `pydantic_agent.py`).
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

_AGENT_VARS: dict[str, str] = {
    "triage": "TRIAGE_PROMPT",
    "preprocess": "PREPROCESS_PROMPT",
    "retriever": "RETRIEVER_PROMPT",
    "calculator": "CALCULATOR_PROMPT",
}


def latest_all() -> list[str]:
    """Return all version strings found in this package, sorted ascending (e.g. ['v1', 'v2'])."""
    versions = [
        m.name
        for m in pkgutil.iter_modules([str(Path(__file__).parent)])
        if re.match(r"^v\d+$", m.name)
    ]
    if not versions:
        raise RuntimeError("No versioned prompt modules found in prompts/")
    return sorted(versions, key=lambda v: int(v[1:]))


def latest() -> str:
    """Return the highest version string found in this package (e.g. 'v2')."""
    return latest_all()[-1]


def load(version: str) -> dict[str, str]:
    """Return the four agent prompts for the requested version.

    Raises ImportError if the module is missing, AttributeError if it doesn't
    export all four required constants.
    """
    module = importlib.import_module(f"convfinqa.prompts.{version}")
    out: dict[str, str] = {}
    for short, var in _AGENT_VARS.items():
        if not hasattr(module, var):
            raise AttributeError(
                f"prompts.{version} is missing required constant '{var}'. "
                "Each version module must export TRIAGE_PROMPT, PREPROCESS_PROMPT, "
                "RETRIEVER_PROMPT, and CALCULATOR_PROMPT."
            )
        out[short] = getattr(module, var)
    return out
