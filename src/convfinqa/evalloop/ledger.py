"""The prompt writer's memory: what was tried on each agent, and how it went.

The M2 teacher had memory of a kind — the last five runs' *diagnoses* — but it
was memory without feedback. It could see what had been proposed and never
whether any of it worked, so a rejected idea could be proposed again, sharpened,
and rejected again, for as many cycles as the campaign lasted.

This joins the two halves that were never joined: every ``kind=propose`` run
carries the prompt it wrote and the agent it targeted; every ``kind=gate`` run
carries the verdict for a candidate version. Keyed on that version, one query
produces, per target agent, the attempt history a writer needs before it writes
again — *this is what I changed, this is what happened*.

Everything here is read-only and best-effort: a tracking store that is down
degrades the writer to the memoryless behaviour it had before, never blocks it.
"""

from __future__ import annotations

import json
from typing import Any

OPTIMIZATION_EXPERIMENT = "convfinqa-optimization"


def _client() -> Any:
    from mlflow.tracking import MlflowClient

    from convfinqa.tracking import mlflow_log

    mlflow_log._mlflow()
    return MlflowClient(tracking_uri=mlflow_log.tracking_uri())


def _runs(client: Any, experiment: str, kind: str, limit: int = 200) -> list[Any]:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    return list(
        client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.kind = '{kind}'",
            order_by=["attributes.start_time DESC"],
            max_results=limit,
        )
    )


def _artifact_json(client: Any, run_id: str, name: str) -> Any:
    try:
        from pathlib import Path

        return json.loads(Path(client.download_artifacts(run_id, name)).read_text())
    except Exception:  # noqa: BLE001 — an unreadable artifact is missing memory, not an error
        return None


def attempts(
    *,
    target_agent: str | None = None,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Past prompt rewrites with their gate outcomes, newest first.

    `target_agent` narrows to one agent's lineage, which is what the writer
    wants: the history that bears on the prompt in front of it.
    """
    try:
        client = _client()
        proposals = _runs(client, experiment, "propose")
        verdicts = _runs(client, experiment, "gate")
    except Exception:  # noqa: BLE001
        return []

    by_version: dict[str, dict[str, Any]] = {}
    for run in verdicts:
        version = run.data.params.get("candidate_version", "")
        if version and version not in by_version:
            by_version[version] = _artifact_json(
                client, run.info.run_id, "verdict.json"
            ) or {
                "promoted": run.data.tags.get("promoted") == "true",
                "reason": run.data.tags.get("reason", ""),
                **{
                    k: v
                    for k, v in run.data.metrics.items()
                    if k in {"accuracy_delta", "cluster_p_one_sided", "n_compared"}
                },
            }

    out: list[dict[str, Any]] = []
    for run in proposals:
        agent = run.data.params.get("target_agent", "")
        if target_agent and agent != target_agent:
            continue
        version = run.data.params.get("new_version", "")
        proposal = _artifact_json(client, run.info.run_id, "proposal.json") or {}
        outcome = by_version.get(version)
        out.append(
            {
                "version": version,
                "base_version": run.data.params.get("prompts_version", ""),
                "target_agent": agent,
                "at": run.info.start_time,
                "rationale": proposal.get("rationale", ""),
                "summary_of_changes": proposal.get("summary_of_changes", ""),
                "prompt": proposal.get("prompt", ""),
                "outcome": "promoted"
                if (outcome or {}).get("promoted")
                else ("rejected" if outcome else "not yet gated"),
                "verdict": (outcome or {}).get("reason", ""),
                "accuracy_delta": (outcome or {}).get("accuracy_delta"),
                "cluster_p_one_sided": (outcome or {}).get("cluster_p_one_sided"),
            }
        )
        if len(out) >= limit:
            break
    return out


def ledger_text(target_agent: str, limit: int = 12) -> str:
    """The attempt history as prose for the writer's prompt.

    Deliberately blunt about outcomes: a rejected attempt is labelled REJECTED
    with its delta and p, because the point of showing it is to stop the writer
    re-proposing it. An empty history says so explicitly rather than being
    omitted, so the writer can tell "nothing tried yet" from "history withheld".
    """
    rows = attempts(target_agent=target_agent, limit=limit)
    if not rows:
        return (
            f"\n\n## Prior attempts on {target_agent}\n"
            "None. This is the first rewrite of this agent's prompt in the "
            "recorded history.\n"
        )
    lines = [f"\n\n## Prior attempts on {target_agent} (newest first)"]
    for r in rows:
        head = f"- {r['version']} — {r['outcome'].upper()}"
        if r.get("accuracy_delta") is not None:
            head += f" (Δ {float(r['accuracy_delta']) * 100:+.2f}pp"
            if r.get("cluster_p_one_sided") is not None:
                head += f", p={float(r['cluster_p_one_sided']):.3f}"
            head += ")"
        lines.append(head)
        if r.get("summary_of_changes"):
            lines.append(f"  changed: {r['summary_of_changes']}")
        if r.get("rationale"):
            lines.append(f"  reasoning: {r['rationale'][:400]}")
    lines.append(
        "\nDo not re-propose a change that was already REJECTED unless you can "
        "say what is different this time."
    )
    return "\n".join(lines)
