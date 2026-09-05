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
from statistics import NormalDist
from typing import Any

from convfinqa.config import EVAL_ROOT, REPO_ROOT

STORY_PATH = EVAL_ROOT / "story.json"
DOCS_DIR = REPO_ROOT / "docs" / "optimization"
SDK_PAGE = "agent-sdk.html"
AGENTS = ("triage", "preprocess", "retriever", "calculator")
PANEL_METRICS = {
    "triage": "acc_triage_turn_type",
    "preprocess": "acc_preprocess_plan",
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
    sdk_grouped: dict[str, list[dict[str, Any]]] = {}
    for run in gates:
        name = run.data.params.get("campaign") or run.data.tags.get("campaign") or "—"
        if campaigns and name not in campaigns:
            continue
        is_sdk = (
            run.data.tags.get("runtime") or run.data.params.get("runtime")
        ) == "agent_sdk"
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
        target_class = run.data.params.get("target_class", "")
        (sdk_grouped if is_sdk else grouped).setdefault(name, []).append(
            {
                "label": run.data.params.get("experiment_label", ""),
                "at": run.info.start_time,
                "runtime": "agent_sdk" if is_sdk else "pipeline",
                # For the SDK arm the target is a failure class, and it sits in
                # `target_agent` too so every reader of an experiment row
                # (page, app, chart) shows *something* without branching.
                "target_agent": (
                    target_class if is_sdk else run.data.params.get("target_agent", "")
                ),
                "target_class": target_class if is_sdk else "",
                "edits": _sdk_edits(run.data.params.get("rewrite_id", ""))
                if is_sdk
                else [],
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
    sdk_champion = doc.aliases.get("sdk_champion")
    eval_records = [
        {
            "run_name": r.info.run_name,
            "start_time": r.info.start_time,
            "params": dict(r.data.params),
            "metrics": dict(r.data.metrics),
        }
        for r in evals
    ]
    try:
        from convfinqa.evalloop import ledgers

        sdk_gate_rows = ledgers.load("gates", runtime="agent_sdk").to_dict(
            orient="records"
        )
    except Exception:  # noqa: BLE001 — no ledger means no SDK gate yet
        sdk_gate_rows = []
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion": champion,
        "sdk_champion": sdk_champion,
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
        "sdk_campaigns": [
            {"name": name, "runtime": "agent_sdk", "experiments": rows}
            for name, rows in sorted(sdk_grouped.items())
        ],
        "runtime_comparison": runtime_comparison(
            eval_records,
            sdk_gate_rows,
            champion=champion,
            sdk_champion=sdk_champion,
            by_turn_type=_cross_runtime_turn_type_split(
                eval_records, champion=champion, sdk_champion=sdk_champion
            ),
        ),
    }


#: What a runtime's arm of the comparison carries. Every value is None until a
#: run of that arm exists, so the page can say "not yet run" rather than 0.
_ARM_KEYS = (
    "version",
    "run_name",
    "accuracy",
    "by_turn_type",
    "panel",
    "cost",
    "wall",
    # Execution accuracy is the headline; program accuracy is the check on it —
    # both arms answer far more turns than they reproduce gold programs for, and
    # a reader comparing the headline against the paper's human figure needs the
    # second number in the same place. Stories built before this key carry it as
    # absent, and the serving route derives it from the committed CSV.
    "program_accuracy",
)


def _arm(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return dict.fromkeys(_ARM_KEYS)
    metrics = record.get("metrics") or {}
    params = record.get("params") or {}
    cost = None
    for key in ("sdk_cost_usd", "cost_usd", "total_cost_usd"):
        if metrics.get(key) is not None:
            cost = metrics[key]
            break
    return {
        "version": params.get("prompts_version") or params.get("version"),
        "run_name": record.get("run_name"),
        "accuracy": metrics.get("accuracy"),
        "by_turn_type": {
            # The aggregate hides where the difference lives: number turns are a
            # lookup both arms have saturated, program turns are the reasoning.
            "number": metrics.get("accuracy_gold_turn_type_Number"),
            "program": metrics.get("accuracy_gold_turn_type_Program"),
        },
        "panel": _panel(metrics),
        "cost": cost,
        "wall": metrics.get("wall_seconds"),
        "program_accuracy": metrics.get("program_accuracy"),
    }


TURN_TYPES = ("Number", "Program")


def turn_type_gate(
    baseline_csv: Path | str, candidate_csv: Path | str
) -> dict[str, Any] | None:
    """Paired per-turn-type verdict from two committed prediction CSVs.

    Split out from `runtime_comparison` so that stays pure: this reads files. The
    aggregate verdict answers "is the candidate better"; this answers "at what",
    and the two arms differ enough per slice that reporting only the aggregate
    would misdescribe the result — a +8.88pp headline that is +13.03pp on program
    turns and exactly 0.00pp on number turns.

    Returns None when either CSV is missing, so a story built without them keeps
    every field absent rather than zero.
    """
    import pandas as pd

    from convfinqa.tracking.comparator import (
        durkalski_z,
        mcnemar_exact_p_one_sided,
    )

    base, cand = Path(baseline_csv), Path(candidate_csv)
    if not base.exists() or not cand.exists():
        return None
    key = ["report_id", "turn_index"]
    b = pd.read_csv(base)[key + ["gold_turn_type", "correct"]]
    c = pd.read_csv(cand)[key + ["correct"]]
    merged = b.merge(c, on=key, suffixes=("_base", "_cand"))
    if merged.empty:
        return None
    out: dict[str, Any] = {}
    for turn_type in TURN_TYPES:
        rows = merged[merged["gold_turn_type"] == turn_type]
        if rows.empty:
            continue
        fixed = int(((~rows["correct_base"]) & rows["correct_cand"]).sum())
        broken = int((rows["correct_base"] & (~rows["correct_cand"])).sum())
        per_cluster: list[tuple[int, int]] = []
        for _, group in rows.groupby("report_id"):
            f = int(((~group["correct_base"]) & group["correct_cand"]).sum())
            b_ = int((group["correct_base"] & (~group["correct_cand"])).sum())
            if f or b_:
                per_cluster.append((b_, f))
        base_acc = float(rows["correct_base"].mean())
        cand_acc = float(rows["correct_cand"].mean())
        z = durkalski_z(per_cluster) if per_cluster else 0.0
        out[turn_type.lower()] = {
            "n": int(len(rows)),
            "baseline_accuracy": round(base_acc, 6),
            "candidate_accuracy": round(cand_acc, 6),
            "delta_pp": round((cand_acc - base_acc) * 100, 4),
            "fixed": fixed,
            "broken": broken,
            "n_flip_clusters": len(per_cluster),
            "cluster_z": round(z, 4),
            "cluster_p_one_sided": round(float(NormalDist().cdf(-z)), 6),
            "mcnemar_p_one_sided": round(mcnemar_exact_p_one_sided(broken, fixed), 8),
        }
    return out or None


def _latest(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(records, key=lambda r: r.get("start_time") or 0) if records else None


def _cross_runtime_turn_type_split(
    eval_records: list[dict[str, Any]],
    *,
    champion: str | None,
    sdk_champion: str | None,
) -> dict[str, Any] | None:
    """The paired per-turn-type verdict, from the two arms' committed CSVs.

    The CSVs are named after their runs and committed, so this reproduces on any
    clone with no tracking server and no API calls. None when either is absent.
    """
    from convfinqa.evalloop.runner import PREDICTIONS_DIR

    comparison = runtime_comparison(
        eval_records, [], champion=champion, sdk_champion=sdk_champion
    )
    names = [comparison[arm].get("run_name") for arm in ("pipeline", "agent_sdk")]
    if not all(names):
        return None
    baseline, candidate = (PREDICTIONS_DIR / f"{name}.csv" for name in names)
    return turn_type_gate(baseline, candidate)


def runtime_comparison(
    eval_records: list[dict[str, Any]],
    sdk_gate_rows: list[dict[str, Any]],
    *,
    champion: str | None,
    sdk_champion: str | None,
    by_turn_type: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The two arms on the gate split, side by side, and the gate between them.

    `eval_records` are ``kind=evalloop`` runs as plain dicts (``run_name``,
    ``start_time``, ``params``, ``metrics``); `sdk_gate_rows` are gates-ledger
    rows with ``runtime == "agent_sdk"``. The pipeline arm is the champion's
    latest test100 run, the SDK arm the latest ``sdk-evalloop-test100-*`` run
    (its `sdk_champion` if one exists, else whichever ran last), and the gate
    is the latest SDK gate row whose baseline is the pipeline champion — the
    cross-runtime verdict. Every field is None until the corresponding run
    exists; the page must not read absence as zero.
    """
    tests = [
        r
        for r in eval_records
        if (r.get("params") or {}).get("split") == "test"
        and "test100" in str(r.get("run_name") or "")
    ]
    pipeline_runs = [
        r
        for r in tests
        if (r.get("params") or {}).get("runtime", "pipeline") != "agent_sdk"
        and (r.get("params") or {}).get("prompts_version") == champion
    ]
    sdk_runs = [
        r
        for r in tests
        if (r.get("params") or {}).get("runtime") == "agent_sdk"
        and str(r.get("run_name") or "").startswith("sdk-evalloop-test100-")
    ]
    if sdk_champion:
        preferred = [
            r
            for r in sdk_runs
            if (r.get("params") or {}).get("prompts_version") == sdk_champion
        ]
        sdk_runs = preferred or sdk_runs
    gate_rows = [
        g
        for g in sdk_gate_rows
        if str(g.get("baseline_version") or "") == (champion or "")
        and str(g.get("split") or "") == "test"
    ]
    gate_row = max(gate_rows, key=lambda g: str(g.get("gated_at") or ""), default=None)
    gate: dict[str, Any] = {
        "delta_pp": None,
        "p_value": None,
        "ci": [None, None],
        "fixed": None,
        "broken": None,
        "candidate_version": None,
        "promoted": None,
        "gate_id": None,
        "by_turn_type": by_turn_type,
    }
    if gate_row is not None:
        gate.update(
            {
                "delta_pp": _num(gate_row.get("delta_pp")),
                "p_value": _num(gate_row.get("p_value")),
                "ci": [_num(gate_row.get("ci_low")), _num(gate_row.get("ci_high"))],
                "fixed": _int(gate_row.get("fixed")),
                "broken": _int(gate_row.get("broken")),
                "candidate_version": gate_row.get("candidate_version"),
                "promoted": bool(gate_row.get("promoted")),
                "gate_id": gate_row.get("gate_id"),
            }
        )
    return {
        "pipeline": _arm(_latest(pipeline_runs)),
        "agent_sdk": _arm(_latest(sdk_runs)),
        "gate": gate,
    }


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # NaN from a pandas cell reads as absent


def _int(value: Any) -> int | None:
    number = _num(value)
    return None if number is None else int(number)


def _sdk_edits(rewrite_id: str) -> list[dict[str, Any]]:
    """The per-class edits of one SDK rewrite, off the rewrites ledger."""
    if not rewrite_id:
        return []
    try:
        from convfinqa.evalloop import ledgers

        rows = ledgers.load("rewrites", runtime="agent_sdk")
    except Exception:  # noqa: BLE001
        return []
    hits = rows[rows["rewrite_id"] == rewrite_id]
    return [
        {
            "edit_id": str(r.edit_id),
            "failure_class": str(r.failure_class or ""),
            "target": str(r.target or ""),
            "change_kind": str(r.change_kind or ""),
            "rationale": str(r.rationale or ""),
            "n_diagnoses": _int(r.n_diagnoses),
        }
        for r in hits.itertuples()
    ]


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
    sdk_page = target / SDK_PAGE
    sdk_page.write_text(render_sdk(data))
    (target / "story.json").write_text(json.dumps(data, indent=1, default=str) + "\n")
    return {
        "story_json": str(STORY_PATH),
        "page": str(page),
        "sdk_page": str(sdk_page),
        "n_campaigns": len(data["campaigns"]),
        "n_experiments": sum(len(c["experiments"]) for c in data["campaigns"]),
        "champion": data["champion"],
    }


def render(data: dict[str, Any]) -> str:
    """The published page. Import kept local so `collect` never needs it."""
    from convfinqa.evalloop.story_page import render_page

    return render_page(data)


def render_sdk(data: dict[str, Any]) -> str:
    """The Agent SDK experiment page, beside the campaign write-up."""
    from convfinqa.evalloop.story_page import render_sdk_page

    return render_sdk_page(data)
