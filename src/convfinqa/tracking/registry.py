"""Bundle registry: versions, champion/challenger aliases, promotion history.

The contract this file enforces, end to end:

1. **Every bundle is registered** — hand-edited prompts, an s7 auto-research
   round, a DSPy/GEPA run, no difference. Its full spec is captured at
   registration and is never deleted, so a failed challenger keeps its spec and
   its evidence exactly as long as a champion does.
2. **It is evaluated on the held-out set** the optimizer never saw.
3. **The comparator decides promotion** — first version is champion by default;
   after that, a net-positive paired comparison on the shared question set
   (more questions fixed than broken; individual pass->fail flips no longer
   veto on their own — see `comparator`). The M2 targeted-challenger path can
   also promote via `registry.promote(force=True, reason=...)` when a target
   agent's first-fault count drops and overall paired accuracy does not
   regress, with the comparison attached rather than silently applied.
4. **Promotion is an append-only event.** The alias moves, and a record is
   appended with timestamp, verdict, and the runs it was based on. Nothing is
   ever overwritten, so the history is the history.

Storage is a JSON document beside the eval artifacts rather than MLflow's
registry tables, for one reason: the demo container has no tracking store, and
the registry view has to render there. The MLflow registry is updated too when
one is reachable, but this file is the source of truth the app reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from convfinqa.config import EVAL_ROOT, settings
from convfinqa.tracking.bundle import bundle_fingerprint, bundle_id
from convfinqa.tracking.comparator import ComparisonResult, accuracy, load_predictions

REGISTRY_PATH = EVAL_ROOT / "registry.json"

CHAMPION = "champion"
CHALLENGER = "challenger"
# The single-session runtime's own champion alias. It is a separate alias and
# not a second value of `champion` because the two runtimes are not
# interchangeable: serving reads `champion` to pick a four-agent bundle, and an
# sdk_vN name there would be a version nothing can serve.
SDK_CHAMPION = "sdk_champion"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RegistryDoc:
    """The whole registry: versions, aliases, promotion history, agent lineages."""

    versions: list[dict[str, Any]]
    aliases: dict[str, str]
    history: list[dict[str, Any]]
    # Per-agent prompt lineages (M2.5): agent -> ordered entries
    # {seq, hash, first_seen_in, parent, source, registered_at, run_id}.
    agent_prompts: dict[str, list[dict[str, Any]]] | None = None
    # The single-session prompt lineage: one ordered list of the same entry
    # shape, seq ``s1, s2, …``, keyed on the whole prompt's hash.
    sdk_prompts: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "model": settings.registered_model_name,
            "versions": self.versions,
            "aliases": self.aliases,
            "history": self.history,
            "agent_prompts": self.agent_prompts or {},
            "sdk_prompts": self.sdk_prompts or [],
        }


def load(path: Path | None = None) -> RegistryDoc:
    """Read the registry, returning an empty one when it does not exist yet."""
    target = path or REGISTRY_PATH
    if not target.exists():
        return RegistryDoc(versions=[], aliases={}, history=[])
    raw = json.loads(target.read_text())
    return RegistryDoc(
        versions=list(raw.get("versions", [])),
        aliases=dict(raw.get("aliases", {})),
        history=list(raw.get("history", [])),
        agent_prompts=dict(raw.get("agent_prompts", {})),
        sdk_prompts=list(raw.get("sdk_prompts", [])),
    )


def save(doc: RegistryDoc, path: Path | None = None) -> Path:
    """Write the registry back to disk."""
    target = path or REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc.as_dict(), indent=2, sort_keys=False) + "\n")
    return target


def find_version(doc: RegistryDoc, version: str) -> dict[str, Any] | None:
    """The registered entry for `version`, or None."""
    for entry in doc.versions:
        if entry.get("version") == version:
            return entry
    return None


def register(
    version: str,
    *,
    source: str = "manual",
    run_id: str = "",
    overlay: str | None = None,
    metrics: dict[str, float] | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Register (or refresh) a bundle version. Never deletes prior entries.

    `source` records *how* the bundle came to exist — `manual`, `gepa`, or `s7` —
    which is the field that makes "auto-research produced this one" visible in
    the registry view rather than folklore.
    """
    doc = load(path)
    fingerprint = bundle_fingerprint(version=version, overlay=overlay)
    entry = find_version(doc, version)
    if entry is None:
        entry = {
            "version": version,
            "registered_at": _now(),
            "source": source,
            "bundle_id": bundle_id(fingerprint),
            "bundle": fingerprint,
            "runs": [],
            "metrics": {},
            "notes": notes,
        }
        doc.versions.append(entry)
    else:
        # Refresh the spec but keep `registered_at` — the first sighting is the
        # fact worth preserving.
        entry["bundle"] = fingerprint
        entry["bundle_id"] = bundle_id(fingerprint)
        if notes:
            entry["notes"] = notes
        entry["source"] = source or entry.get("source", "manual")
    if run_id and run_id not in entry["runs"]:
        entry["runs"].append(run_id)
    if metrics:
        entry["metrics"] = {**entry.get("metrics", {}), **metrics}
    if extra:
        entry.update(extra)
    save(doc, path)
    return entry


def record_evaluation(
    version: str,
    *,
    run_id: str = "",
    source: str = "manual",
    path: Path | None = None,
) -> dict[str, Any]:
    """Register a version and attach the accuracy of its committed predictions."""
    try:
        df = load_predictions(version)
        metrics = {"accuracy": round(accuracy(df), 6), "n_questions": float(len(df))}
    except (FileNotFoundError, ValueError):
        metrics = {}
    return register(version, source=source, run_id=run_id, metrics=metrics, path=path)


def champion(path: Path | None = None) -> str | None:
    """The version currently aliased `champion`, if any."""
    return load(path).aliases.get(CHAMPION)


def is_sdk_version(version: str) -> bool:
    """Whether `version` belongs to the single-session (`sdk_vN`) lineage."""
    import convfinqa.prompts as prompts_pkg

    return prompts_pkg.is_sdk_version(version)


def set_alias(alias: str, version: str, path: Path | None = None) -> RegistryDoc:
    """Point `alias` at `version` without recording a promotion event.

    Used for `challenger`, which moves freely. Champion moves go through
    `promote`, which requires a verdict.

    The pipeline aliases and the single-session alias are kept apart in both
    directions: an `sdk_vN` version can only take an `sdk_`-prefixed alias, and
    a bundle version can never take one. Serving reads `champion` to build four
    agents, so an sdk version there would be a champion nothing can serve — and
    an sdk run must never be able to move the pipeline's alias by accident.
    """
    doc = load(path)
    if find_version(doc, version) is None:
        raise ValueError(f"Cannot alias unregistered version {version!r}")
    if is_sdk_version(version) != alias.startswith("sdk_"):
        raise ValueError(
            f"alias {alias!r} and version {version!r} belong to different "
            "runtimes: sdk_vN versions take sdk_-prefixed aliases only"
        )
    doc.aliases[alias] = version
    save(doc, path)
    return doc


@dataclass
class PromotionOutcome:
    """Result of a promotion attempt: whether it happened, and why."""

    promoted: bool
    version: str
    previous_champion: str | None
    reason: str
    comparison: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the API."""
        return {
            "promoted": self.promoted,
            "version": self.version,
            "previous_champion": self.previous_champion,
            "reason": self.reason,
            "comparison": self.comparison,
        }


def promote(
    version: str,
    *,
    comparison: ComparisonResult | None = None,
    actor: str = "owner",
    force: bool = False,
    reason: str | None = None,
    path: Path | None = None,
) -> PromotionOutcome:
    """Promote `version` to champion if the comparator allows it.

    The first registered version becomes champion by default — there is nothing
    to compare it against, and a system with no champion cannot serve. Every
    promotion after that needs a passing comparison.

    `force` exists for the deliberate override (a rollback to a known-good
    version, say) and is recorded as such in the history, never silently.
    """
    doc = load(path)
    if find_version(doc, version) is None:
        raise ValueError(f"Cannot promote unregistered version {version!r}")
    if is_sdk_version(version):
        raise ValueError(
            f"{version!r} is a single-session prompt; `promote` moves the "
            f"pipeline's champion, which cannot be an sdk version. Use "
            f"set_alias({SDK_CHAMPION!r}, ...) for the sdk lineage."
        )

    previous = doc.aliases.get(CHAMPION)

    if previous is None:
        reason = reason or "first registered version becomes champion by default"
    elif previous == version:
        return PromotionOutcome(
            promoted=False,
            version=version,
            previous_champion=previous,
            reason=f"{version} is already the champion",
            comparison=comparison.as_dict() if comparison else None,
        )
    elif force:
        reason = reason or "forced promotion (comparator bypassed deliberately)"
    else:
        if comparison is None:
            from convfinqa.tracking.comparator import compare

            comparison = compare(previous, version)
        if not comparison.promotable:
            return PromotionOutcome(
                promoted=False,
                version=version,
                previous_champion=previous,
                reason=f"comparator refused: {comparison.reason()}",
                comparison=comparison.as_dict(),
            )
        reason = f"comparator passed: {comparison.reason()}"

    doc.aliases[CHAMPION] = version
    if doc.aliases.get(CHALLENGER) == version:
        doc.aliases.pop(CHALLENGER, None)
    doc.history.append(
        {
            "at": _now(),
            "event": "promote",
            "version": version,
            "previous_champion": previous,
            "actor": actor,
            "forced": force,
            "reason": reason,
            "comparison": comparison.as_dict() if comparison else None,
        }
    )
    save(doc, path)
    _mirror_to_mlflow(version)
    return PromotionOutcome(
        promoted=True,
        version=version,
        previous_champion=previous,
        reason=reason,
        comparison=comparison.as_dict() if comparison else None,
    )


def sdk_champion(path: Path | None = None) -> str | None:
    """The version currently aliased `sdk_champion`, if any."""
    return load(path).aliases.get(SDK_CHAMPION)


def promote_sdk(
    version: str,
    *,
    comparison: ComparisonResult | None = None,
    reason: str | None = None,
    evidence_split: str = "test",
    actor: str = "evalloop-cycle-sdk",
    path: Path | None = None,
) -> PromotionOutcome:
    """Move ONLY the `sdk_champion` alias to `version`, and record why.

    The single-session lineage has its own champion and its own promotion
    path, kept apart from `promote` so that neither can move the other's
    alias: serving reads `champion` to build four agents, and an `sdk_vN` there
    would be a champion nothing can serve. Three refusals, all raised rather
    than returned, because each is a caller bug and not a verdict:

    - a version outside the `sdk_vN` lineage;
    - evidence from any split but the gate split (`evidence_split != "test"`)
      — train runs optimise, test runs promote, same as the pipeline arm;
    - a comparison that fails the campaign rule (`promotable_significant`),
      when one is given. The first sdk version becomes `sdk_champion` by
      default, as the first bundle becomes `champion`.

    The history event is ``promote_sdk`` with ``alias: sdk_champion`` so a
    reader of `registry.json` can never mistake it for a pipeline promotion.
    """
    doc = load(path)
    if not is_sdk_version(version):
        raise ValueError(
            f"{version!r} is not a single-session prompt version; promote_sdk "
            f"moves {SDK_CHAMPION!r} only and takes sdk_vN versions only"
        )
    if find_version(doc, version) is None:
        raise ValueError(f"Cannot promote unregistered version {version!r}")
    if evidence_split != "test":
        raise ValueError(
            "promotion evidence must come from the unseen test split — this "
            f"comparison ran on {evidence_split!r}. Train runs optimise; test "
            "runs promote."
        )
    previous = doc.aliases.get(SDK_CHAMPION)
    if previous == version:
        return PromotionOutcome(
            promoted=False,
            version=version,
            previous_champion=previous,
            reason=f"{version} is already the {SDK_CHAMPION}",
            comparison=comparison.as_dict() if comparison else None,
        )
    if previous is None and comparison is None:
        reason = reason or f"first sdk version becomes {SDK_CHAMPION} by default"
    elif comparison is not None and not comparison.promotable_significant:
        return PromotionOutcome(
            promoted=False,
            version=version,
            previous_champion=previous,
            reason=f"campaign rule refused: {comparison.reason()}",
            comparison=comparison.as_dict(),
        )
    elif comparison is None:
        raise ValueError(
            f"{SDK_CHAMPION} is {previous!r}; moving it needs a passing gate comparison"
        )
    else:
        reason = reason or f"campaign rule passed: {comparison.reason()}"

    doc.aliases[SDK_CHAMPION] = version
    doc.history.append(
        {
            "at": _now(),
            "event": "promote_sdk",
            "alias": SDK_CHAMPION,
            "version": version,
            "previous_champion": previous,
            "actor": actor,
            "forced": False,
            "reason": reason,
            "evidence_split": evidence_split,
            "comparison": comparison.as_dict() if comparison else None,
        }
    )
    save(doc, path)
    return PromotionOutcome(
        promoted=True,
        version=version,
        previous_champion=previous,
        reason=reason or "",
        comparison=comparison.as_dict() if comparison else None,
    )


def _mirror_to_mlflow(version: str) -> None:
    """Best-effort mirror of the alias into the MLflow model registry.

    Best-effort by design: the committed registry.json is the source of truth the
    app reads, and a missing tracking store must not block a promotion.
    """
    try:
        from mlflow.tracking import MlflowClient

        from convfinqa.tracking.mlflow_log import tracking_uri

        client = MlflowClient(tracking_uri=tracking_uri())
        name = settings.registered_model_name
        try:
            client.get_registered_model(name)
        except Exception:  # noqa: BLE001
            client.create_registered_model(name)
        versions = client.search_model_versions(f"name='{name}'")
        for candidate in versions:
            if candidate.tags.get("bundle_version") == version:
                client.set_registered_model_alias(name, CHAMPION, candidate.version)
                return
    except Exception:  # noqa: BLE001
        return


def summary(path: Path | None = None) -> dict[str, Any]:
    """The registry as the admin API serves it, newest promotion first."""
    doc = load(path)
    return {
        "model": settings.registered_model_name,
        "aliases": doc.aliases,
        "versions": sorted(
            doc.versions, key=lambda v: str(v.get("registered_at", "")), reverse=True
        ),
        "history": list(reversed(doc.history)),
    }
