"""Per-agent prompt lineages (M2.5): each subagent's prompt versions on its own.

The four prompts are independent — the teacher changes one at a time — so each
agent gets its own lineage: ``triage t1, t2, …``, ``retriever r1, r2, …``. An
entry is identified by the prompt's content hash (the truth) and carries a
human seq label. A bundle module (``v4``) then resolves to a *composition* —
``t1.p1.r2.c1`` — which is what makes "v4 is v3_1 with only the retriever
changed" a fact the registry can state and MLflow can filter on.

The ledger lives in ``registry.json → agent_prompts``. Reads are free and
side-effect-free (`resolve`); writes happen only in explicit paths — the
backfill CLI, the teacher's propose step, and the start of an eval-loop run
(`ensure`) — never as a byproduct of fingerprinting in a serving process.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("convfinqa.tracking")

AGENTS = ("triage", "preprocess", "retriever", "calculator")
_INITIAL = {"triage": "t", "preprocess": "p", "retriever": "r", "calculator": "c"}


def prompt_hash(text: str) -> str:
    """Content identity of one agent prompt: first 8 hex chars of SHA-256."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lineage(doc: Any, agent: str) -> list[dict[str, Any]]:
    if doc.agent_prompts is None:
        doc.agent_prompts = {}
    lineage: list[dict[str, Any]] = doc.agent_prompts.setdefault(agent, [])
    return lineage


def resolve(version: str) -> dict[str, dict[str, str]]:
    """Read-only composition of a bundle version: agent → {seq, hash}.

    A hash the ledger has not seen yet renders as ``t?`` — the caller that
    *runs* things should have called `ensure` first; a caller that merely
    reports (fingerprints, healthz) must not mutate the committed registry.
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.tracking import registry

    prompts = prompts_pkg.load(version)
    doc = registry.load()
    out: dict[str, dict[str, str]] = {}
    for agent in AGENTS:
        h = prompt_hash(prompts[agent])
        entry = next(
            (e for e in (doc.agent_prompts or {}).get(agent, []) if e["hash"] == h),
            None,
        )
        seq = entry["seq"] if entry else f"{_INITIAL[agent]}?"
        out[agent] = {"seq": seq, "hash": h}
    return out


def ensure(
    version: str, *, source: str = "manual", run_id: str = ""
) -> dict[str, dict[str, str]]:
    """Register any unseen prompt hashes of `version`; return its composition.

    New entries get the next seq in that agent's lineage, with the previous
    latest entry as parent. Idempotent: a known hash is returned as-is.
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.tracking import registry

    prompts = prompts_pkg.load(version)
    doc = registry.load()
    changed = False
    out: dict[str, dict[str, str]] = {}
    for agent in AGENTS:
        h = prompt_hash(prompts[agent])
        lineage = _lineage(doc, agent)
        entry = next((e for e in lineage if e["hash"] == h), None)
        if entry is None:
            seq = f"{_INITIAL[agent]}{len(lineage) + 1}"
            entry = {
                "seq": seq,
                "hash": h,
                "first_seen_in": version,
                "parent": lineage[-1]["seq"] if lineage else None,
                "source": source,
                "registered_at": _now(),
                "run_id": run_id,
            }
            lineage.append(entry)
            changed = True
        out[agent] = {"seq": entry["seq"], "hash": h}
    if changed:
        registry.save(doc)
    return out


def composition_string(comp: dict[str, dict[str, str]]) -> str:
    """`t1.p1.r2.c1` — the human-scannable form, in fixed agent order."""
    return ".".join(comp[a]["seq"] for a in AGENTS)


def changed_agents(base_version: str, new_version: str) -> list[str]:
    """Which agents' prompts differ between two bundle versions."""
    import convfinqa.prompts as prompts_pkg

    base = prompts_pkg.load(base_version)
    new = prompts_pkg.load(new_version)
    return [a for a in AGENTS if prompt_hash(base[a]) != prompt_hash(new[a])]


def backfill(versions: list[str] | None = None) -> dict[str, str]:
    """Seed the lineages from the committed bundle modules, oldest first."""
    import convfinqa.prompts as prompts_pkg

    out: dict[str, str] = {}
    for v in versions or prompts_pkg.latest_all():
        out[v] = composition_string(ensure(v, source="backfill"))
    return out


# ── Phase D: mirror each agent's lineage into MLflow's prompt registry ───


def mirror_to_mlflow(version: str) -> dict[str, str]:
    """Register `version`'s four prompts in MLflow's prompt registry.

    One registered prompt per agent (``convfinqa.triage`` …), a new version
    only when the text changed, and a ``champion`` alias per agent when
    `version` is the bundle champion. The JSON ledger stays the source of
    truth; this is the browsing surface (MLflow UI → Prompts).
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.tracking import mlflow_log, registry

    prompts = prompts_pkg.load(version)
    comp = resolve(version)
    is_champion = registry.champion() == version
    out: dict[str, str] = {}
    try:
        mlflow_log._mlflow()
        import mlflow.genai as genai
    except Exception:  # noqa: BLE001 — mirroring is never load-bearing
        log.warning("mlflow prompt registry unavailable; skipped", exc_info=True)
        return out
    for agent in AGENTS:
        name = f"convfinqa.{agent}"
        seq, h = comp[agent]["seq"], comp[agent]["hash"]
        try:
            try:
                latest = genai.load_prompt(f"prompts:/{name}@latest")
            except Exception:  # noqa: BLE001 — first registration
                latest = None
            if latest is not None and latest.template == prompts[agent]:
                pv = latest
            else:
                pv = genai.register_prompt(
                    name=name,
                    template=prompts[agent],
                    commit_message=f"{seq}@{h} (bundle {version})",
                    tags={"seq": seq, "hash": h, "bundle": version},
                )
            if is_champion:
                genai.set_prompt_alias(name, alias="champion", version=pv.version)
            out[agent] = f"{name} v{pv.version} ({seq}@{h})"
        except Exception:  # noqa: BLE001
            log.warning("prompt mirror failed for %s", name, exc_info=True)
    return out
