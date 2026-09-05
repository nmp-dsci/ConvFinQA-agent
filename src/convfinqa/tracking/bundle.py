"""The bundle fingerprint: what "a model version" means in this system.

There is no model checkpoint here to version — every component is a hosted model
behind an API. What actually changes between one accuracy number and the next is
a *bundle*: which prompts, which optimizer overlay, which two models, which
dataset, which code. Version the bundle together or version nothing, because a
prompt improvement measured against a different dataset snapshot is not a
measurement.

The same fingerprint is stamped on every MLflow run, every prediction CSV, and
every serving session, so any answer the app ever gave is attributable to the
exact build that produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

from convfinqa.config import REPO_ROOT, settings
from convfinqa.data.loader import DATASET_PATH


@cache
def code_sha() -> str:
    """Short git SHA of the working tree, or `unknown` outside a checkout.

    The container has no `.git`, which is why this degrades rather than raises —
    the deployed image records its SHA through a build arg instead.
    """
    env_sha = _env_sha()
    if env_sha:
        return env_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _env_sha() -> str:
    import os

    return os.environ.get("CONVFINQA_CODE_SHA", "").strip()


@cache
def dataset_hash() -> str:
    """First 12 hex chars of the dataset's SHA-256, or `missing`.

    Read in chunks: the file is ~21 MB and this runs at import in the API.
    """
    if not DATASET_PATH.exists():
        return "missing"
    digest = hashlib.sha256()
    with DATASET_PATH.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def prompts_version() -> str:
    """The prompt bundle in force: the pin, else the champion, else the newest.

    The middle step is the one that matters. Falling straight through to
    `prompts.latest()` meant `/healthz` reported `champion: v2` beside
    `prompts_version: v3_1` — two fields describing the same deployment and
    disagreeing, because "newest prompt file on disk" and "the bundle that was
    promoted" are different questions. `v3_1` exists as a file precisely because
    it was *tried and not promoted*; describing the landing page with it named a
    bundle nothing was serving.

    The explicit pin still wins: `PROMPTS_VERSION` is how an operator overrides
    the registry on purpose, and an override that the registry could veto would
    not be an override.
    """
    import convfinqa.prompts as prompts_pkg

    if settings.prompts_version:
        return settings.prompts_version

    # Imported lazily: `registry` imports this module, so a top-level import
    # here would be a cycle.
    from convfinqa.tracking import registry

    champion = registry.champion()
    if champion and champion in prompts_pkg.latest_all():
        return champion
    return prompts_pkg.latest()


def gepa_overlay() -> str | None:
    """The GEPA run whose optimized prompts are overlaid, if any."""
    if not settings.prompts_overlay_path:
        return None
    from convfinqa.pipeline.prompts_loader import GEPA_NAME

    return GEPA_NAME


def bundle_fingerprint(
    *,
    version: str | None = None,
    overlay: str | None = None,
) -> dict[str, Any]:
    """Return the full bundle spec for the current configuration.

    `version` and `overlay` override the resolved values so an eval of a
    non-active version stamps the version it actually ran, not the ambient one.
    """
    resolved = version or prompts_version()
    spec = {
        "prompts_version": resolved,
        "gepa_overlay": overlay if overlay is not None else gepa_overlay(),
        "lm_mini": "deepseek-v4-flash",
        "lm_max": settings.lm_max_model,
        "dataset_hash": dataset_hash(),
        "code_sha": code_sha(),
    }
    spec.update(_composition_fields(resolved))
    return spec


def _composition_fields(version: str) -> dict[str, str]:
    """Per-agent prompt versions (M2.5), read-only — `t1.p1.r2.c1` and v_* keys.

    Resolve, never register: fingerprinting happens in serving processes that
    must not mutate the committed registry. Runners call
    `prompt_ledger.ensure()` before fingerprinting so seqs already exist.
    Degrades to nothing for a version whose module cannot be loaded.
    """
    try:
        import convfinqa.prompts as prompts_pkg
        from convfinqa.tracking import prompt_ledger

        if prompts_pkg.is_sdk_version(version):
            # A single-session prompt has one lineage, not four: `s2@abcd1234`.
            entry = prompt_ledger.resolve_sdk(version)
            return {
                "composition": prompt_ledger.sdk_composition_string(entry),
                "v_sdk": f"{entry['seq']}@{entry['hash']}",
            }
        comp = prompt_ledger.resolve(version)
    except Exception:  # noqa: BLE001 — identity extras must never break a fingerprint
        return {}
    return {
        "composition": prompt_ledger.composition_string(comp),
        **{f"v_{a}": f"{v['seq']}@{v['hash']}" for a, v in comp.items()},
    }


def bundle_id(fingerprint: dict[str, Any] | None = None) -> str:
    """A stable short id for a bundle: same spec in, same id out.

    Used as the join key between an MLflow run, a prediction CSV, and a trace, so
    two identical configurations on different machines land in the same bucket.
    """
    spec = fingerprint or bundle_fingerprint()
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def write_bundle_file(path: Path, fingerprint: dict[str, Any] | None = None) -> Path:
    """Write the bundle spec next to an artifact so the artifact is self-describing."""
    spec = fingerprint or bundle_fingerprint()
    payload = {"bundle_id": bundle_id(spec), **spec}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
