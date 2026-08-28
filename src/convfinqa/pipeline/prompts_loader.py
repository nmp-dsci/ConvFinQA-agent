"""Resolve pipeline prompts from versioned modules or GEPA-overlay JSON."""

from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any

from convfinqa.config import RUNS_DIR, settings

_DEFAULT_GEPA_NAME = "gepa_real_20260502_005251"
GEPA_NAME = settings.gepa_name or _DEFAULT_GEPA_NAME
RUN_DIR = RUNS_DIR / GEPA_NAME
PROMPTS_PATH = RUN_DIR / "dspy_optimized_runner.json"
if not PROMPTS_PATH.exists():
    PROMPTS_PATH = RUN_DIR / "optimized_runner.json"

PROMPTS_VERSION = settings.prompts_version


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `overlay` onto a copy of `base`. `overlay` wins per leaf."""
    out = _copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = _copy.deepcopy(v)
    return out


def _load_optimized_prompts(path: Path) -> dict[str, str]:
    """Load per-stage instructions plus field guidance from a DSPy runner JSON."""
    raw = json.loads(path.read_text())
    overlay_path = settings.prompts_overlay_path
    if overlay_path and overlay_path.exists():
        overlay = json.loads(overlay_path.read_text())
        if overlay:
            raw = _deep_merge(raw, overlay)

    mapping = {
        "triage": "triage.predict",
        "preprocess": "preprocess.predict",
        "retriever": "retriever.predict",
        "calculator": "calculator.react",
    }
    prompts: dict[str, str] = {}
    for short, key in mapping.items():
        signature = raw[key]["signature"]
        instructions = signature["instructions"].rstrip()
        fields = signature.get("fields", [])

        field_lines: list[str] = []
        for field in fields:
            prefix = str(field.get("prefix", "")).strip().rstrip(":")
            desc = str(field.get("description", "")).strip()
            if not prefix or not desc or (desc.startswith("${") and desc.endswith("}")):
                continue
            field_lines.append(f"- {prefix}: {desc}")

        if field_lines:
            instructions = f"{instructions}\n\nField guidance:\n" + "\n".join(
                field_lines
            )
        prompts[short] = instructions
    return prompts


def resolve_prompts() -> dict[str, str]:
    """Resolve pipeline prompts. Overlay JSON wins; otherwise load versioned module."""
    if settings.prompts_overlay_path:
        return _load_optimized_prompts(PROMPTS_PATH)
    import convfinqa.prompts as _prompts_pkg

    version = PROMPTS_VERSION or _prompts_pkg.latest()
    try:
        return _prompts_pkg.load(version)
    except (ImportError, AttributeError):
        fallback = _prompts_pkg.latest()
        print(f"[prompts] '{version}' not found — falling back to '{fallback}'")  # noqa: T201
        return _prompts_pkg.load(fallback)


_resolve_prompts = resolve_prompts
PROMPTS = resolve_prompts()
