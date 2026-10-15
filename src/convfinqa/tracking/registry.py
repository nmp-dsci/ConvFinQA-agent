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

# The universal champion (2026-10-15, owner's decision): one champion for the
# whole project, and it is an *agent* — a prompt version plus the runtime, model
# and effort it runs at, held in `champion_agent` beside the alias. sdk_v1 won it
# by beating the pipeline champion v8 on the gate split (+8.88pp, p=0.0003), so
# serving and every "the champion" on the page mean this one.
CHAMPION = "champion"
# The four-agent pipeline lineage keeps its own best under its own alias. It is
# not served; it is what pipeline campaigns gate against and what `promote`
# moves, so a pipeline improvement can never dethrone the champion by beating
# only the pipeline's previous best.
PIPELINE_CHAMPION = "pipeline_champion"
CHALLENGER = "challenger"
# Retired. Until 2026-10-15 the single-session runtime had its own alias beside
# a pipeline `champion`; the universal champion replaced both. Still read from a
# registry that predates the migration, never written.
SDK_CHAMPION = "sdk_champion"

# The confidence judge's alias (s12). Serving reads it to decide which judge
# prompt sits between an agent_sdk answer and the user; it is calibrated to
# the `sdk_champion` it was scored against, which `promote_judge` records.
JUDGE_CHAMPION = "judge_champion"


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
    # The confidence judge's prompt lineage (s12): same entry shape, seq
    # ``j1, j2, …``. A judge prompt answers nothing, so it lives in neither of
    # the answering lineages.
    judge_prompts: list[dict[str, Any]] | None = None
    # The universal champion as an agent: {version, runtime, model, effort}.
    # The alias names the prompt; this names how it runs. None before the
    # 2026-10-15 migration.
    champion_agent: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "model": settings.registered_model_name,
            "versions": self.versions,
            "aliases": self.aliases,
            "history": self.history,
            "agent_prompts": self.agent_prompts or {},
            "sdk_prompts": self.sdk_prompts or [],
            "judge_prompts": self.judge_prompts or [],
            **({"champion_agent": self.champion_agent} if self.champion_agent else {}),
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
        judge_prompts=list(raw.get("judge_prompts", [])),
        champion_agent=dict(raw["champion_agent"])
        if raw.get("champion_agent")
        else None,
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
    """The universal champion's prompt version, if any."""
    return load(path).aliases.get(CHAMPION)


def pipeline_champion(path: Path | None = None) -> str | None:
    """The four-agent pipeline lineage's best bundle, if any.

    Falls back to `champion` for a registry that predates the universal
    champion, when `champion` still named a pipeline bundle.
    """
    return _pipeline_champion_of(load(path))


def _pipeline_champion_of(doc: RegistryDoc) -> str | None:
    """`pipeline_champion` for an already-loaded registry."""
    named = doc.aliases.get(PIPELINE_CHAMPION)
    if named:
        return named
    legacy = doc.aliases.get(CHAMPION)
    if legacy and not is_sdk_version(legacy) and not is_judge_version(legacy):
        return legacy
    return None


def _sdk_champion_of(doc: RegistryDoc) -> str | None:
    """The champion's sdk prompt version for an already-loaded registry."""
    named = doc.aliases.get(CHAMPION)
    if named and is_sdk_version(named):
        return named
    return doc.aliases.get(SDK_CHAMPION)


def champion_agent(path: Path | None = None) -> dict[str, Any] | None:
    """The universal champion as an agent: {version, runtime, model, effort}.

    None for a registry that predates the record. Readers that need a model or
    an effort fall back to `llm`'s constants then, as they did before.
    """
    doc = load(path)
    if not doc.champion_agent:
        return None
    return dict(doc.champion_agent)


def is_sdk_version(version: str) -> bool:
    """Whether `version` belongs to the single-session (`sdk_vN`) lineage."""
    import convfinqa.prompts as prompts_pkg

    return prompts_pkg.is_sdk_version(version)


def is_judge_version(version: str) -> bool:
    """Whether `version` belongs to the confidence-judge (`judge_jN`) lineage."""
    import convfinqa.prompts as prompts_pkg

    return prompts_pkg.is_judge_version(version)


def _alias_family(name: str) -> str:
    """Which lineage an alias or version name belongs to: pipeline, sdk or judge."""
    if name.startswith("judge_"):
        return "judge"
    if name.startswith("sdk_"):
        return "sdk"
    return "pipeline"


def _alias_accepts(alias: str, version: str) -> bool:
    """Whether `alias` may name `version`.

    The universal champion may be either answering runtime — its record says
    which — but never a judge. Every other alias stays within its lineage.
    """
    if alias == CHAMPION:
        return _alias_family(version) != "judge"
    return _alias_family(alias) == _alias_family(version)


def set_alias(alias: str, version: str, path: Path | None = None) -> RegistryDoc:
    """Point `alias` at `version` without recording a promotion event.

    Used for `challenger`, which moves freely. Champion moves go through
    `promote`, which requires a verdict.

    Lineage aliases stay within their lineage: `pipeline_champion` and
    `challenger` take bundles only, `judge_`-prefixed aliases take judge
    versions only. The universal `champion` takes either answering runtime and
    never a judge; its runtime, model and effort live in `champion_agent`, which
    only the promotion paths write.
    """
    doc = load(path)
    if find_version(doc, version) is None:
        raise ValueError(f"Cannot alias unregistered version {version!r}")
    if not _alias_accepts(alias, version):
        raise ValueError(
            f"alias {alias!r} and version {version!r} belong to different "
            "runtimes or lineages: the champion takes an answering agent (vN or "
            "sdk_vN), pipeline aliases take bundles, judge_ aliases take judge_jN "
            "versions"
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
    """Promote bundle `version` to `pipeline_champion` if the comparator allows it.

    This is the pipeline lineage's promotion. It moves `pipeline_champion`, not
    the universal `champion`: a bundle that beats the previous bundle has not
    beaten the champion, which is an agent_sdk agent since 2026-10-15. The
    first registered bundle becomes the pipeline champion by default. Every
    promotion after that needs a passing comparison.

    `force` exists for the deliberate override (a rollback to a known-good
    version, say) and is recorded as such in the history, never silently.
    """
    doc = load(path)
    if find_version(doc, version) is None:
        raise ValueError(f"Cannot promote unregistered version {version!r}")
    if is_sdk_version(version):
        raise ValueError(
            f"{version!r} is a single-session prompt; `promote` moves "
            f"{PIPELINE_CHAMPION!r}, which cannot be an sdk version. Use "
            "promote_sdk(...) to move the champion's prompt."
        )
    if is_judge_version(version):
        raise ValueError(
            f"{version!r} is a confidence-judge prompt; `promote` moves "
            f"{PIPELINE_CHAMPION!r}, which cannot be a judge version. Use "
            f"promote_judge(...) for the judge lineage."
        )

    previous = _pipeline_champion_of(doc)

    if previous is None:
        reason = reason or "first registered version becomes champion by default"
    elif previous == version:
        return PromotionOutcome(
            promoted=False,
            version=version,
            previous_champion=previous,
            reason=f"{version} is already the {PIPELINE_CHAMPION}",
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

    doc.aliases[PIPELINE_CHAMPION] = version
    if doc.aliases.get(CHAMPION) == previous and not doc.champion_agent:
        # A registry that predates the universal champion: `champion` was the
        # pipeline's alias, so it moves with it until the migration runs.
        doc.aliases[CHAMPION] = version
    if doc.aliases.get(CHALLENGER) == version:
        doc.aliases.pop(CHALLENGER, None)
    doc.history.append(
        {
            "at": _now(),
            "event": "promote",
            "alias": PIPELINE_CHAMPION,
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
    """The champion's single-session prompt version, if the champion is one.

    Since the universal champion this is `champion` whenever it names an
    `sdk_vN`; a registry that predates it still answers from the retired
    `sdk_champion` alias.
    """
    return _sdk_champion_of(load(path))


def champion_evidence_of(evidence_csv: Path | str) -> dict[str, Any]:
    """The champion's floor, measured: the test pass it stands on and its accuracy there.

    The CI gate floor-checks the champion against this exact run. It is a
    property of the champion *agent*, not of its prompt version: sdk_v1 scores
    90.5% on Sonnet 5 and 87.4% on Haiku 5.5, so a floor kept on the version
    entry — empty for every sdk version, which let the gate pass at "registered
    0.00%" — would be wrong as soon as `promote_agent` moved the model.
    """
    from convfinqa.tracking.comparator import load_evalloop_csv

    frame = load_evalloop_csv(evidence_csv, split="test")
    return {
        "evidence_run": Path(str(evidence_csv)).name.removesuffix(".csv"),
        "accuracy": round(float(frame["correct"].mean()), 6),
    }


def _ran_as(run_name: str, agent: dict[str, Any]) -> list[str]:
    """Why `run_name` is not a pass of `agent` — empty when it is.

    The name carries the prompt version, the model slug (absent on a run from
    before slugs, which ran the default model) and the effort token.
    """
    from convfinqa.evalloop.runner import prior_sdk_effort, prior_sdk_model
    from convfinqa.llm import LM_SDK_MODEL, sdk_model_slug

    problems: list[str] = []
    version = str(agent.get("version"))
    if f"-{version}·" not in run_name:
        problems.append(f"it is not a {version} run")
    slug = prior_sdk_model(run_name)
    model = str(agent.get("model"))
    if (slug or sdk_model_slug(LM_SDK_MODEL)) != sdk_model_slug(model):
        problems.append(f"it ran {slug or LM_SDK_MODEL}, not {model}")
    effort = prior_sdk_effort(run_name)
    if effort is not None and effort != agent.get("effort"):
        problems.append(f"it ran at effort {effort}, not {agent.get('effort')}")
    return problems


def record_champion_evidence(
    evidence_csv: Path | str,
    *,
    actor: str = "owner",
    reason: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Put the champion's floor on its record, from the test pass that measured it.

    Refused (raised) when there is no champion record, when the CSV is not a
    complete test pass, or when its name says it ran a different version,
    model or effort from the champion. Appends a ``champion_evidence`` event.
    """
    doc = load(path)
    if not doc.champion_agent:
        raise ValueError(
            "the registry has no champion_agent record; run "
            "migrate_to_universal_champion first"
        )
    evidence = champion_evidence_of(evidence_csv)
    problems = _ran_as(evidence["evidence_run"], doc.champion_agent)
    if problems:
        raise ValueError(
            f"{evidence['evidence_run']} is not the champion's evidence: "
            + "; ".join(problems)
        )
    before = dict(doc.champion_agent)
    doc.champion_agent = {**before, **evidence}
    doc.history.append(
        {
            "at": _now(),
            "event": "champion_evidence",
            "alias": CHAMPION,
            "version": before.get("version"),
            "actor": actor,
            "reason": reason
            or "record the floor the CI gate checks the champion against",
            "champion_agent_before": before,
            "champion_agent_after": doc.champion_agent,
        }
    )
    save(doc, path)
    return dict(doc.champion_agent)


def promote_sdk(
    version: str,
    *,
    comparison: ComparisonResult | None = None,
    reason: str | None = None,
    evidence_split: str = "test",
    actor: str = "evalloop-cycle-sdk",
    evidence_csv: Path | str | None = None,
    path: Path | None = None,
) -> PromotionOutcome:
    """Move the champion's prompt to sdk `version`, and record why.

    `evidence_csv` is the candidate's gate pass: on a registry with a champion
    record it becomes the new champion's floor (`champion_evidence_of`). With
    none, the old floor is dropped rather than kept — it measured a different
    prompt — and the CI gate fails until one is recorded.

    On a registry with a `champion_agent` record this moves the universal
    `champion` and the record's version; the record's model and effort stay,
    because a prompt challenger is gated on the champion's model and effort. On
    a registry that predates the record it moves the retired `sdk_champion`
    alias, as it always did. Either way `promote` cannot reach it, and it
    cannot reach `pipeline_champion`. Three refusals, all raised rather than
    returned, because each is a caller bug and not a verdict:

    - a version outside the `sdk_vN` lineage;
    - evidence from any split but the gate split (`evidence_split != "test"`)
      — train runs optimise, test runs promote, same as the pipeline arm;
    - a comparison that fails the campaign rule (`promotable_significant`),
      when one is given. The first sdk version becomes `sdk_champion` by
      default, as the first bundle becomes `champion`.

    The history event is ``promote_sdk`` and names the alias it moved, so a
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
    alias = CHAMPION if doc.champion_agent else SDK_CHAMPION
    previous = _sdk_champion_of(doc)
    if previous == version:
        return PromotionOutcome(
            promoted=False,
            version=version,
            previous_champion=previous,
            reason=f"{version} is already the {alias}",
            comparison=comparison.as_dict() if comparison else None,
        )
    if previous is None and comparison is None:
        reason = reason or f"first sdk version becomes {alias} by default"
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
            f"{alias} is {previous!r}; moving it needs a passing gate comparison"
        )
    else:
        reason = reason or f"campaign rule passed: {comparison.reason()}"

    doc.aliases[alias] = version
    if doc.champion_agent:
        kept = {
            k: v
            for k, v in doc.champion_agent.items()
            if k not in ("accuracy", "evidence_run")
        }
        evidence = champion_evidence_of(evidence_csv) if evidence_csv else {}
        problems = (
            _ran_as(evidence["evidence_run"], {**kept, "version": version})
            if evidence
            else []
        )
        if problems:
            raise ValueError(
                f"{evidence['evidence_run']} is not {version}'s gate pass: "
                + "; ".join(problems)
            )
        doc.champion_agent = {**kept, "version": version, **evidence}
    doc.history.append(
        {
            "at": _now(),
            "event": "promote_sdk",
            "alias": alias,
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


#: The verdicts that may move the champion's model or effort. "beats" is the
#: house gate (net positive, one-sided cluster p < 0.05); "matches" is the
#: non-inferiority rule for a cheaper agent (s19).
AGENT_PROMOTING_OUTCOMES = ("beats", "matches")


def promote_agent(
    *,
    model: str,
    effort: str,
    verdict: dict[str, Any],
    evidence_split: str = "test",
    actor: str = "owner",
    reason: str | None = None,
    path: Path | None = None,
) -> PromotionOutcome:
    """Move the champion's model and effort, keeping its prompt, and record why.

    The champion is an agent; a challenger that shares its prompt and changes
    how it runs (s19: Sonnet 5 → Haiku 5.5) is promoted here. `verdict` is the
    model-swap verdict (`story.model_swap_verdict`): it must name an outcome in
    `AGENT_PROMOTING_OUTCOMES` and carry the `candidate_model`, `candidate_effort`
    and `version` it was measured for, which must match what is promoted, and test evidence is required, as for every
    other promotion. Refusals are raised, because each is a caller bug: no
    champion record, train evidence, an unknown effort, or a verdict that did
    not pass. The history event is ``promote_agent`` and carries the record
    before and after.
    """
    from convfinqa.llm import SDK_EFFORTS

    doc = load(path)
    if not doc.champion_agent:
        raise ValueError(
            "the registry has no champion_agent record; run "
            "migrate_to_universal_champion first"
        )
    if evidence_split != "test":
        raise ValueError(
            "promotion evidence must come from the unseen test split — this "
            f"verdict ran on {evidence_split!r}. Train runs optimise; test runs promote."
        )
    if effort not in SDK_EFFORTS:
        raise ValueError(f"unknown effort {effort!r}; expected one of {SDK_EFFORTS}")
    outcome = str(verdict.get("outcome") or "")
    before = dict(doc.champion_agent)
    if outcome not in AGENT_PROMOTING_OUTCOMES:
        return PromotionOutcome(
            promoted=False,
            version=str(before.get("version")),
            previous_champion=str(before.get("version")),
            reason=f"model-swap rule refused: {outcome or 'no outcome'} — "
            f"{verdict.get('reason', '')}",
            comparison=verdict,
        )
    mismatch = [
        f"{field} {verdict.get(key)!r} != {want!r}"
        for key, field, want in (
            ("candidate_model", "model", model),
            ("candidate_effort", "effort", effort),
            ("version", "version", before.get("version")),
        )
        if verdict.get(key) != want
    ]
    if not verdict.get("candidate_run"):
        mismatch.append("candidate_run missing: no test pass to floor-check")
    if mismatch:
        return PromotionOutcome(
            promoted=False,
            version=str(before.get("version")),
            previous_champion=str(before.get("version")),
            reason="verdict was not measured for this configuration: "
            + "; ".join(mismatch),
            comparison=verdict,
        )
    after = {**before, "model": model, "effort": effort}
    if (after["model"], after["effort"]) == (before.get("model"), before.get("effort")):
        return PromotionOutcome(
            promoted=False,
            version=str(before.get("version")),
            previous_champion=str(before.get("version")),
            reason="the champion already runs on this model and effort",
            comparison=verdict,
        )
    reason = (
        reason or f"model-swap rule passed ({outcome}): {verdict.get('reason', '')}"
    )
    after.update(champion_evidence_of(str(verdict["candidate_run"])))
    doc.champion_agent = after
    doc.history.append(
        {
            "at": _now(),
            "event": "promote_agent",
            "alias": CHAMPION,
            "version": after.get("version"),
            "previous_champion": before.get("version"),
            "champion_agent_before": before,
            "champion_agent_after": after,
            "actor": actor,
            "forced": False,
            "reason": reason,
            "evidence_split": evidence_split,
            "verdict": verdict,
        }
    )
    save(doc, path)
    return PromotionOutcome(
        promoted=True,
        version=str(after.get("version")),
        previous_champion=str(before.get("version")),
        reason=reason,
        comparison=verdict,
    )


def migrate_to_universal_champion(
    *,
    model: str,
    effort: str,
    reason: str,
    actor: str = "owner",
    path: Path | None = None,
) -> RegistryDoc:
    """Make the sdk champion the one champion, as an agent. Idempotent.

    Before: `champion` names the pipeline's bundle, `sdk_champion` the
    single-session prompt. After: `champion` names the sdk prompt,
    `champion_agent` records how it runs, `pipeline_champion` keeps the bundle,
    and the retired `sdk_champion` alias is gone. One ``universal_champion``
    history event records the move; the history itself is never rewritten.
    """
    from convfinqa.llm import SDK_EFFORTS

    if effort not in SDK_EFFORTS:
        raise ValueError(f"unknown effort {effort!r}; expected one of {SDK_EFFORTS}")
    doc = load(path)
    if doc.champion_agent:
        return doc
    sdk = _sdk_champion_of(doc)
    if not sdk:
        raise ValueError("no sdk champion to make universal")
    pipeline = _pipeline_champion_of(doc)
    if pipeline:
        doc.aliases[PIPELINE_CHAMPION] = pipeline
    doc.aliases[CHAMPION] = sdk
    doc.aliases.pop(SDK_CHAMPION, None)
    doc.champion_agent = {
        "version": sdk,
        "runtime": "agent_sdk",
        "model": model,
        "effort": effort,
    }
    doc.history.append(
        {
            "at": _now(),
            "event": "universal_champion",
            "alias": CHAMPION,
            "version": sdk,
            "previous_champion": pipeline,
            "champion_agent_after": dict(doc.champion_agent),
            "actor": actor,
            "forced": False,
            "reason": reason,
        }
    )
    save(doc, path)
    return doc


def judge_champion(path: Path | None = None) -> str | None:
    """The version currently aliased `judge_champion`, if any."""
    return load(path).aliases.get(JUDGE_CHAMPION)


def promote_judge(
    version: str,
    *,
    verdict: dict[str, Any] | None = None,
    runtime_version: str = "",
    reason: str | None = None,
    evidence_split: str = "calibrate",
    actor: str = "evalloop-judge",
    path: Path | None = None,
) -> PromotionOutcome:
    """Move ONLY the `judge_champion` alias to `version`, and record why.

    The judge's promotion rule is decided by `evalloop.judge.gate_judges`; this
    records its verdict. Refuses a non-judge version, refuses gate-split
    evidence (the gate is the judge's *test*, scored once by the frozen
    champion — a judge chosen on it would have been tuned on its own exam),
    and, when a verdict is given, refuses one that did not pass. The first
    judge becomes champion by default, as the first bundle does. The history
    event is ``promote_judge`` and carries ``runtime_version`` — the
    `sdk_champion` this judge was calibrated against — because a judge is only
    calibrated for one answering model.
    """
    doc = load(path)
    if not is_judge_version(version):
        raise ValueError(
            f"{version!r} is not a confidence-judge prompt version; promote_judge "
            f"moves {JUDGE_CHAMPION!r} only and takes judge_jN versions only"
        )
    if find_version(doc, version) is None:
        raise ValueError(f"Cannot promote unregistered version {version!r}")
    if evidence_split == "test":
        raise ValueError(
            "a judge is never promoted on the gate split — that split is the "
            "judge's own test, scored once by the frozen champion"
        )
    previous = doc.aliases.get(JUDGE_CHAMPION)
    if previous == version:
        return PromotionOutcome(
            promoted=False,
            version=version,
            previous_champion=previous,
            reason=f"{version} is already the {JUDGE_CHAMPION}",
            comparison=verdict,
        )
    if previous is None and verdict is None:
        reason = reason or f"first judge becomes {JUDGE_CHAMPION} by default"
    elif verdict is not None and not verdict.get("promotable"):
        return PromotionOutcome(
            promoted=False,
            version=version,
            previous_champion=previous,
            reason=f"judge rule refused: {verdict.get('reason', '')}",
            comparison=verdict,
        )
    elif verdict is None:
        raise ValueError(
            f"{JUDGE_CHAMPION} is {previous!r}; moving it needs a passing judge gate"
        )
    else:
        reason = reason or f"judge rule passed: {verdict.get('reason', '')}"

    doc.aliases[JUDGE_CHAMPION] = version
    doc.history.append(
        {
            "at": _now(),
            "event": "promote_judge",
            "alias": JUDGE_CHAMPION,
            "version": version,
            "previous_champion": previous,
            "runtime_version": runtime_version,
            "actor": actor,
            "forced": False,
            "reason": reason,
            "evidence_split": evidence_split,
            "comparison": verdict,
        }
    )
    save(doc, path)
    return PromotionOutcome(
        promoted=True,
        version=version,
        previous_champion=previous,
        reason=reason or "",
        comparison=verdict,
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
