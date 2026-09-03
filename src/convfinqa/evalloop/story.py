"""The public write-up, built from what actually happened rather than asserted.

`story.json` is assembled entirely from the tracking store and the committed
registry: campaigns from the ``kind=gate`` runs, per-agent panels from the
``kind=evalloop`` runs those verdicts were computed on, prompt diffs and
rationales from the ``kind=propose`` runs, and the champion lineage from
``registry.json``. Nothing in it is typed by hand, which is the property that
makes it worth publishing — a page that could disagree with the record is a
claim, not evidence.

The HTML page is generated from that JSON, so republishing after another
campaign is one command, and a CI check can fail when the committed page no
longer matches the registry it claims to describe.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from convfinqa.config import EVAL_ROOT, REPO_ROOT

STORY_PATH = EVAL_ROOT / "story.json"
DOCS_DIR = REPO_ROOT / "docs" / "optimization"
AGENTS = ("triage", "preprocess", "retriever", "calculator")
PANEL_METRICS = {
    "triage": "acc_triage_turn_type",
    "preprocess": "acc_preprocess_skeleton",
    "retriever": "retriever_operand_recall",
    "calculator": "acc_calculator_exec",
}


def _client() -> Any:
    from mlflow.tracking import MlflowClient

    from convfinqa.tracking import mlflow_log

    mlflow_log._mlflow()
    return MlflowClient(tracking_uri=mlflow_log.tracking_uri())


def _search(client: Any, experiment: str, filter_string: str) -> list[Any]:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    return list(
        client.search_runs(
            [exp.experiment_id],
            filter_string=filter_string,
            order_by=["attributes.start_time ASC"],
            max_results=500,
        )
    )


def _panel(metrics: dict[str, float]) -> dict[str, float | None]:
    return {a: metrics.get(m) for a, m in PANEL_METRICS.items()}


def collect(campaigns: list[str] | None = None) -> dict[str, Any]:
    """Everything the page shows, read out of the record."""
    from convfinqa.config import settings
    from convfinqa.evalloop.splits import load_manifest, manifest_path
    from convfinqa.tracking import registry

    client = _client()
    gates = _search(client, "convfinqa-optimization", "tags.kind = 'gate'")
    proposals = _search(client, "convfinqa-optimization", "tags.kind = 'propose'")
    evals = _search(client, settings.mlflow_experiment, "tags.kind = 'evalloop'")

    # Gate runs by candidate version, eval runs by (version, split).
    by_version_proposal = {r.data.params.get("new_version", ""): r for r in proposals}
    gate_runs_by_version: dict[str, dict[str, float]] = {}
    for r in evals:
        if r.data.params.get("split") != "test":
            continue
        gate_runs_by_version[r.data.params.get("prompts_version", "")] = dict(
            r.data.metrics
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in gates:
        name = run.data.params.get("campaign") or run.data.tags.get("campaign") or "—"
        if campaigns and name not in campaigns:
            continue
        candidate = run.data.params.get("candidate_version", "")
        baseline = run.data.params.get("baseline_version", "")
        proposal = by_version_proposal.get(candidate)
        detail: dict[str, Any] = {}
        diff = ""
        if proposal is not None:
            detail = _artifact(client, proposal.info.run_id, "proposal.json") or {}
            diff = (
                _artifact(client, proposal.info.run_id, "prompt_diff.json") or {}
            ).get("diff", "")
        metrics = run.data.metrics
        grouped.setdefault(name, []).append(
            {
                "label": run.data.params.get("experiment_label", ""),
                "at": run.info.start_time,
                "target_agent": run.data.params.get("target_agent", ""),
                "baseline_version": baseline,
                "candidate_version": candidate,
                "promoted": run.data.tags.get("promoted") == "true",
                "accuracy_delta": metrics.get("accuracy_delta"),
                "cluster_p_one_sided": metrics.get("cluster_p_one_sided"),
                "delta_ci": [metrics.get("delta_ci_lo"), metrics.get("delta_ci_hi")],
                "n_compared": metrics.get("n_compared"),
                "fixed": metrics.get("fail_to_pass"),
                "broken": metrics.get("pass_to_fail"),
                "target_metric_delta": metrics.get("target_metric_delta"),
                "rationale": detail.get("rationale", ""),
                "summary_of_changes": detail.get("summary_of_changes", ""),
                "prompt_chars": {
                    "before": (
                        proposal.data.metrics.get("prompt_chars_before")
                        if proposal
                        else None
                    ),
                    "after": (
                        proposal.data.metrics.get("prompt_chars_after")
                        if proposal
                        else None
                    ),
                },
                "diff": diff,
                "panel_baseline": _panel(gate_runs_by_version.get(baseline, {})),
                "panel_candidate": _panel(gate_runs_by_version.get(candidate, {})),
                "accuracy_baseline": gate_runs_by_version.get(baseline, {}).get(
                    "accuracy"
                ),
                "accuracy_candidate": gate_runs_by_version.get(candidate, {}).get(
                    "accuracy"
                ),
            }
        )

    doc = registry.load()
    lineage = [
        {
            "at": event.get("at"),
            "version": event.get("version"),
            "previous": event.get("previous_champion"),
            "actor": event.get("actor"),
            "reason": event.get("reason", ""),
        }
        for event in doc.history
        if event.get("event") == "promote"
    ]

    # Which manifest to describe comes from the *runs*, not from the ambient
    # EVAL_MANIFEST. Reading the environment would let the page describe a
    # different split from the one the experiments were actually gated on —
    # silently, and with no way to tell from the page itself.
    manifest_names = [
        r.data.params.get("manifest")
        for r in evals
        if r.data.params.get("split") == "test" and r.data.params.get("manifest")
    ]
    try:
        chosen = manifest_names[-1] if manifest_names else None
        manifest = load_manifest(manifest_path(chosen) if chosen else None)
        split_info = {
            "name": manifest["name"],
            "gate_reports": manifest["stats"]["test"]["n_reports"],
            "gate_questions": manifest["stats"]["test"]["n_questions"],
            "train_reports": manifest["stats"]["train"]["n_reports"],
        }
    except Exception:  # noqa: BLE001 — the page still builds without a manifest
        split_info = {}

    champion = doc.aliases.get("champion")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion": champion,
        # The champion's own gate accuracy, so the headline has a number even
        # before any experiment has moved it — which is the state a campaign
        # spends most of its life in.
        "champion_accuracy": gate_runs_by_version.get(champion or "", {}).get(
            "accuracy"
        ),
        "champion_panel": _panel(gate_runs_by_version.get(champion or "", {})),
        "split": split_info,
        "alpha": 0.05,
        "rule": (
            "net positive on the shared gate questions AND one-sided "
            "cluster-corrected McNemar p < 0.05"
        ),
        "campaigns": [
            {"name": name, "experiments": rows}
            for name, rows in sorted(grouped.items())
        ],
        "lineage": lineage,
        "champion_track": _champion_track(grouped, gate_runs_by_version),
    }


def _champion_track(
    grouped: dict[str, list[dict[str, Any]]],
    gate_metrics: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Accuracy and the per-agent panel at each point the champion moved.

    Only promoted experiments appear: the champion track is the line the page
    is about, and a rejection does not move it. Rejections are still in
    `campaigns` — they are most of the story, and hiding them would make the
    loop look better than it is.
    """
    track: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rows in grouped.values():
        for row in rows:
            if not row["promoted"]:
                continue
            for version in (row["baseline_version"], row["candidate_version"]):
                if version in seen:
                    continue
                seen.add(version)
                metrics = gate_metrics.get(version, {})
                track.append(
                    {
                        "version": version,
                        "at": row["at"],
                        "accuracy": metrics.get("accuracy"),
                        "panel": _panel(metrics),
                        "moved_by": row["label"]
                        if version == row["candidate_version"]
                        else None,
                        "target_agent": row["target_agent"]
                        if version == row["candidate_version"]
                        else None,
                    }
                )
    return track


def _artifact(client: Any, run_id: str, name: str) -> Any:
    try:
        return json.loads(Path(client.download_artifacts(run_id, name)).read_text())
    except Exception:  # noqa: BLE001
        return None


def build(
    *, campaigns: list[str] | None = None, out_dir: str | None = None
) -> dict[str, Any]:
    """Write `evaluation/story.json` and the published page; return a summary."""
    data = collect(campaigns)
    STORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORY_PATH.write_text(json.dumps(data, indent=1, default=str) + "\n")

    target = Path(out_dir) if out_dir else DOCS_DIR
    target.mkdir(parents=True, exist_ok=True)
    page = target / "index.html"
    page.write_text(render(data))
    (target / "story.json").write_text(json.dumps(data, indent=1, default=str) + "\n")
    return {
        "story_json": str(STORY_PATH),
        "page": str(page),
        "n_campaigns": len(data["campaigns"]),
        "n_experiments": sum(len(c["experiments"]) for c in data["campaigns"]),
        "champion": data["champion"],
    }


def render(data: dict[str, Any]) -> str:
    """The published page. Import kept local so `collect` never needs it."""
    from convfinqa.evalloop.story_page import render_page

    return render_page(data)
