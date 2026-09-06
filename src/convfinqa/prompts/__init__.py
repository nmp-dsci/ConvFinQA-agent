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


# --- Single-session (Agent SDK) prompts ---------------------------------------
#
# The qa_agent runtime has one prompt, not four, so it gets its own lineage:
# modules named `sdk_v1`, `sdk_v2`, … exporting a single `SDK_PROMPT` constant.
# They are kept apart from the bundle versions deliberately — `load("sdk_v1")`
# would fail on the four missing constants, and `latest_all()` must not offer a
# single-session prompt as a pipeline bundle.

_SDK_VERSION_RE = re.compile(r"^sdk_v(\d+)$")
SDK_VAR = "SDK_PROMPT"


def is_sdk_version(version: str) -> bool:
    """Whether `version` names a single-session prompt rather than a bundle."""
    return bool(_SDK_VERSION_RE.match(version))


def sdk_versions() -> list[str]:
    """Every `sdk_vN` module in this package, sorted ascending by N."""
    found = [
        m.name
        for m in pkgutil.iter_modules([str(Path(__file__).parent)])
        if _SDK_VERSION_RE.match(m.name)
    ]
    return sorted(found, key=lambda name: int(name.removeprefix("sdk_v")))


def latest_sdk() -> str:
    """The highest `sdk_vN` present. Raises when none has been written yet."""
    versions = sdk_versions()
    if not versions:
        raise RuntimeError("No sdk_v* prompt modules found in prompts/")
    return versions[-1]


def load_sdk(version: str) -> str:
    """The single-session prompt of `version` (`sdk_vN`).

    Raises ValueError for a name outside the lineage — a bundle version passed
    here by mistake must not be answered with one of its four prompts.
    """
    if not is_sdk_version(version):
        raise ValueError(
            f"{version!r} is not a single-session prompt version (expected sdk_vN)"
        )
    module = importlib.import_module(f"convfinqa.prompts.{version}")
    text = getattr(module, SDK_VAR, None)
    if not isinstance(text, str):
        raise AttributeError(
            f"prompts.{version} must export a string constant {SDK_VAR!r}"
        )
    return text
