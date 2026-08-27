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


_VERSION_RE = re.compile(r"^v(\d+)(?:_(\d+))?$")


def _version_key(name: str) -> tuple[int, int]:
    """Sort key for version module names: (major, minor). Missing minor → 0.

    `v1` → (1, 0); `v2` → (2, 0); `v3` → (3, 0);
    `v3_1` → (3, 1); `v3_2` → (3, 2).
    Names that don't match the regex sort to the end (10_000, 0).
    """
    m = _VERSION_RE.match(name)
    if not m:
        return (10_000, 0)
    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) is not None else 0
    return (major, minor)


def latest_all() -> list[str]:
    """Return all version strings found in this package, sorted ascending.

    Matches both plain versions (`v1`, `v2`) and variant versions (`v3_1`, `v3_2`).
    Tagged variants like `v3_2_alt` are NOT auto-included — they're still
    loadable via `load("v3_2_alt")` but won't appear in the eval comparison table.
    """
    versions = [
        m.name
        for m in pkgutil.iter_modules([str(Path(__file__).parent)])
        if _VERSION_RE.match(m.name)
    ]
    if not versions:
        raise RuntimeError("No versioned prompt modules found in prompts/")
    return sorted(versions, key=_version_key)


def latest() -> str:
    """Return the highest version string found in this package (e.g. 'v3_1')."""
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
