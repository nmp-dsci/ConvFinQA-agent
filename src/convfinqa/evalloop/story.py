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

    # Every eval run the record holds: the tracking store's, plus the committed
    # snapshot's for runs the store does not have. Since the move to the central
    # server (2026-09-21) the pre-migration runs — the sdk_v1 champion's own
    # evidence among them — exist only in the snapshot and the read-only
    # archive, and a challenger run on the central server has to pair with them.
    eval_records = _with_snapshot_runs(
        [
            {
                "run_name": r.info.run_name,
                "run_id": r.info.run_id,
                "start_time": r.info.start_time,
                "params": dict(r.data.params),
                "metrics": dict(r.data.metrics),
            }
            for r in evals
        ]
    )

    # Gate runs by candidate version, eval runs by (version, split). Oldest
    # first, so a version's latest test run is the one that stays.
    by_version_proposal = {r.data.params.get("new_version", ""): r for r in proposals}
    gate_runs_by_version: dict[str, dict[str, float]] = {}
    for rec in sorted(eval_records, key=lambda r: r.get("start_time") or 0):
        params = rec.get("params") or {}
        if params.get("split") != "test":
            continue
        gate_runs_by_version[str(params.get("prompts_version", ""))] = dict(
            rec.get("metrics") or {}
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
        (r.get("params") or {}).get("manifest")
        for r in sorted(eval_records, key=lambda r: r.get("start_time") or 0)
        if (r.get("params") or {}).get("split") == "test"
        and (r.get("params") or {}).get("manifest")
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

    # Since 2026-10-15 the champion is one agent (`champion_agent`), and the
    # pipeline lineage's best is `pipeline_champion`. Everything that is about
    # the pipeline's own campaigns — the champion track, the lineage, the
    # pipeline arm of the runtime comparison — reads the latter.
    champion = doc.aliases.get("champion")
    sdk_champion = registry.sdk_champion()
    pipeline_champion = registry.pipeline_champion()
    champion_agent = registry.champion_agent()
    try:
        from convfinqa.evalloop import ledgers

        sdk_gate_rows = ledgers.load("gates", runtime="agent_sdk").to_dict(
            orient="records"
        )
    except Exception:  # noqa: BLE001 — no ledger means no SDK gate yet
        sdk_gate_rows = []
    model_comparison = sdk_model_comparison(eval_records, sdk_champion=sdk_champion)
    # The champion's own gate accuracy is its *agent's*: the reference arm of
    # the model comparison is the run on the champion's model and effort.
    # Keying on the version alone would pick whichever model last scored it.
    reference_arm = next(
        (
            a
            for a in model_comparison.get("models", [])
            if a.get("model") == model_comparison.get("reference_model")
            and a.get("effort") == model_comparison.get("reference_effort")
        ),
        None,
    )
    if champion and champion == sdk_champion and reference_arm is not None:
        champion_accuracy = reference_arm.get("accuracy")
        champion_panel = reference_arm.get("panel") or {}
    else:
        champion_accuracy = gate_runs_by_version.get(champion or "", {}).get("accuracy")
        champion_panel = _panel(gate_runs_by_version.get(champion or "", {}))
    campaigns_out = [
        {"name": name, "experiments": rows} for name, rows in sorted(grouped.items())
    ]
    sdk_campaigns_out = [
        {"name": name, "runtime": "agent_sdk", "experiments": rows}
        for name, rows in sorted(sdk_grouped.items())
    ]
    track = _champion_track(grouped, gate_runs_by_version)
    # The campaign record (gate and propose runs, their diffs and rationales)
    # predates the central server and was never migrated, so a store with no
    # gate runs at all is a store that never held them — not a record that
    # emptied. Carry the committed record forward rather than publish a page
    # that says no campaign ever ran, and say where it came from.
    campaign_record_from = None
    if not grouped and not sdk_grouped and not campaigns:
        prior = _committed_story()
        if prior.get("campaigns") or prior.get("sdk_campaigns"):
            campaigns_out = list(prior.get("campaigns") or [])
            sdk_campaigns_out = list(prior.get("sdk_campaigns") or [])
            track = list(prior.get("champion_track") or [])
            campaign_record_from = prior.get("campaign_record_from") or prior.get(
                "generated_at"
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Set when the campaign sections were carried from an earlier build.
        "campaign_record_from": campaign_record_from,
        "champion": champion,
        "champion_agent": champion_agent,
        "sdk_champion": sdk_champion,
        "pipeline_champion": pipeline_champion,
        # The champion's own gate accuracy, so the headline has a number even
        # before any experiment has moved it — which is the state a campaign
        # spends most of its life in.
        "champion_accuracy": champion_accuracy,
        "champion_panel": champion_panel,
        # The pipeline lineage's best, for the pipeline campaigns' own pages.
        "pipeline_champion_accuracy": gate_runs_by_version.get(
            pipeline_champion or "", {}
        ).get("accuracy"),
        "pipeline_champion_panel": _panel(
            gate_runs_by_version.get(pipeline_champion or "", {})
        ),
        "split": split_info,
        "alpha": 0.05,
        "rule": (
            "net positive on the shared gate questions AND one-sided "
            "cluster-corrected McNemar p < 0.05"
        ),
        "campaigns": campaigns_out,
        "lineage": lineage,
        "champion_track": track,
        "sdk_campaigns": sdk_campaigns_out,
        "runtime_comparison": runtime_comparison(
            eval_records,
            sdk_gate_rows,
            champion=pipeline_champion,
            sdk_champion=sdk_champion,
            by_turn_type=_cross_runtime_turn_type_split(
                eval_records,
                sdk_gate_rows,
                champion=pipeline_champion,
                sdk_champion=sdk_champion,
            ),
        ),
        "sdk_model_comparison": model_comparison,
        # The confidence judge (s12): every judge version's selective metrics
        # on each split it was scored on, from the committed scores CSVs only.
        "judge": _judge_summary(),
    }


def _with_snapshot_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`records` plus the committed snapshot's eval runs the store does not hold.

    The tracking store wins on a run both hold. The snapshot is the committed
    export (`convfinqa-mlflow snapshot`), so this reads no store and spends
    nothing; it is how a story built on the central server still sees the runs
    recorded before the move.
    """
    from convfinqa.tracking.snapshot import read_snapshot

    have = {str(r.get("run_id")) for r in records if r.get("run_id")}
    extra = [
        {
            "run_name": run.get("run_name"),
            "run_id": run.get("run_id"),
            "start_time": run.get("start_time"),
            "params": dict(run.get("params") or {}),
            "metrics": dict(run.get("metrics") or {}),
        }
        for run in read_snapshot().get("runs", [])
        if run.get("kind") == "evalloop" and str(run.get("run_id")) not in have
    ]
    return [*records, *extra]


def _committed_story() -> dict[str, Any]:
    """The committed story.json, or {} when there is none."""
    try:
        data: dict[str, Any] = json.loads(STORY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data


def _judge_summary() -> dict[str, Any]:
    """`evalloop.judge.summary()`, degraded to an empty record when it cannot run."""
    try:
        from convfinqa.evalloop import judge

        return judge.summary()
    except Exception:  # noqa: BLE001 — the page still builds without a judge
        return {"dataset": None, "champion": None, "versions": [], "gates": []}


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
    # The sdk arm's model, from the run's `sdk_model` param. None on the
    # pipeline arm, whose four agents' model is fixed in llm.py and not a
    # property of a pass.
    "model",
    # The sdk arm's effort — part of what the agent is. From the `sdk_effort`
    # param, or dated against the effort pin for a run older than the param.
    "effort",
)


def _arm(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return dict.fromkeys(_ARM_KEYS)
    metrics = record.get("metrics") or {}
    params = record.get("params") or {}
    # An sdk run is priced by `sdk_session_cost_usd` alone. Its older
    # `sdk_cost_usd` summed the CLI's running totals (2.3x too high on sdk_v1's
    # test pass), so a run without the corrected figure shows no cost rather
    # than that one — `backfill-sdk-cost` derives it from the run's traces.
    keys = (
        ("sdk_session_cost_usd",)
        if params.get("runtime") == "agent_sdk"
        else ("cost_usd", "total_cost_usd")
    )
    cost = None
    for key in keys:
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
        "model": params.get("sdk_model") or None,
        "effort": _sdk_effort_of(record)
        if params.get("runtime") == "agent_sdk"
        else None,
    }


def _sdk_model_of(record: dict[str, Any], reference_model: str) -> str:
    """The model an sdk run ran on. Pre-slug runs logged the param too; a run
    with neither ran the default of its day, which is the reference."""
    return str((record.get("params") or {}).get("sdk_model") or reference_model)


def _sdk_effort_of(record: dict[str, Any]) -> str:
    """The effort an sdk run ran at (`llm.sdk_effort_of_run`)."""
    from convfinqa.llm import sdk_effort_of_run

    return sdk_effort_of_run(record.get("params") or {}, record.get("start_time"))


def _champion_record() -> dict[str, Any]:
    try:
        from convfinqa.tracking import registry

        return registry.champion_agent() or {}
    except Exception:  # noqa: BLE001
        return {}


def sdk_reference_model() -> str:
    """The model the sdk arms are *compared* on: the champion's.

    Read from the champion record, never from a process setting: SDK_MODEL
    pins what one process runs, and the page must not change what it compares
    against because of an environment variable. A model-swap pass is a column
    beside the champion's, never a silent replacement of it.
    """
    from convfinqa.llm import LM_SDK_MODEL

    return str(_champion_record().get("model") or LM_SDK_MODEL)


def sdk_reference_effort() -> str:
    """The effort the sdk arms are compared at: the champion's."""
    from convfinqa.llm import LM_SDK_EFFORT

    return str(_champion_record().get("effort") or LM_SDK_EFFORT)


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
    sdk_gate_rows: list[dict[str, Any]] | None = None,
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
        eval_records, sdk_gate_rows or [], champion=champion, sdk_champion=sdk_champion
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
    reference_model: str | None = None,
) -> dict[str, Any]:
    """The two arms on the gate split, side by side, and the gate between them.

    `eval_records` are ``kind=evalloop`` runs as plain dicts (``run_name``,
    ``start_time``, ``params``, ``metrics``); `sdk_gate_rows` are gates-ledger
    rows with ``runtime == "agent_sdk"``. `champion` is the *pipeline*
    champion here — the arm the cross-runtime gate measured against. The gate
    is the latest SDK gate row whose baseline is that champion, and the SDK
    arm is **the run that gate measured** (its ``candidate_eval_run_id``), so
    no later pass — another model, another effort, a promotion that moved the
    champion's model — can re-label the verdict. Without a gate row the arm
    falls back to the latest ``sdk-evalloop-test100-*`` run of `sdk_champion`
    on the reference model. Runs of the same prompt on other agents are
    reported by `sdk_model_comparison`, not here. Every field is None until
    the corresponding run exists; the page must not read absence as zero.
    """
    reference_model = reference_model or sdk_reference_model()
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
        and _sdk_model_of(r, reference_model) == reference_model
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
    gated_run_id = str((gate_row or {}).get("candidate_eval_run_id") or "")
    gated = [r for r in tests if gated_run_id and r.get("run_id") == gated_run_id]
    if gated:
        sdk_runs = gated
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


#: The non-inferiority margin a cheaper challenger agent must hold (s19, owner's
#: decision D1, 2026-10-15): the test split's 95% cluster-bootstrap lower bound
#: on Δ accuracy may not fall below −3pp. Fixed before any Haiku 5.5 number
#: existed; changing it after seeing a result is not a margin.
MODEL_SWAP_MARGIN_PP = 3.0
MODEL_SWAP_ALPHA = 0.05


def _agent_key(record: dict[str, Any], reference_model: str) -> tuple[str, str]:
    """(model, effort) of an sdk run: what the agent was, not just its prompt."""
    return _sdk_model_of(record, reference_model), _sdk_effort_of(record)


def sdk_model_comparison(
    eval_records: list[dict[str, Any]],
    *,
    sdk_champion: str | None,
    reference_model: str | None = None,
    reference_effort: str | None = None,
    read_csvs: bool = True,
    margin_pp: float = MODEL_SWAP_MARGIN_PP,
) -> dict[str, Any]:
    """One prompt, several agents: the sdk champion scored on each model and effort, paired.

    A model-swap pass keeps the prompt, the tools, the questions and the scoring
    and changes how the agent runs, so two such passes are a paired comparison
    of the agents alone. Agents are keyed on **(model, effort)**: two efforts
    of one model are two agents, and collapsing them would compare a run with
    itself. The *reference* agent is the champion's (its record's model and
    effort).

    ``models`` holds the latest ``sdk-evalloop-test100-`` run of `sdk_champion`
    per agent (the reference first, then by accuracy). ``pairs`` holds each
    other agent against the reference:

    - on **test**, the gate's cluster-corrected McNemar and cluster bootstrap CI
      from the two committed CSVs, with the one-sided p in the direction of the
      observed delta, plus ``p_worse``/``p_better`` (`_paired_from_csvs`);
    - on **train**, ``train`` pairs the challenger's early-stopped runs with the
      reference's on the same ``train_draw_seed``. Both stop at the first wrong
      answer, so the unit is the conversation — answered right to the end or
      not — under an exact McNemar (`_train_pair_from_csvs`);
    - a ``verdict`` (`model_swap_verdict`): ``beats``, ``matches`` or
      ``falls_short`` under the rule fixed in s19.

    Nothing here promotes and nothing is written to the gates ledger; the
    verdict is what `registry.promote_agent` is handed. ``models`` is empty
    until a run exists, and a pair whose CSV is missing or incomplete is
    reported with its statistics None rather than dropped.
    """
    reference_model = reference_model or sdk_reference_model()
    reference_effort = reference_effort or sdk_reference_effort()
    reference_key = (reference_model, reference_effort)

    def of_champion(r: dict[str, Any]) -> bool:
        params = r.get("params") or {}
        return params.get("runtime") == "agent_sdk" and (
            sdk_champion is None or params.get("prompts_version") == sdk_champion
        )

    runs = [
        r
        for r in eval_records
        if of_champion(r)
        and (r.get("params") or {}).get("split") == "test"
        and str(r.get("run_name") or "").startswith("sdk-evalloop-test100-")
    ]
    by_agent: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in runs:
        by_agent.setdefault(_agent_key(r, reference_model), []).append(r)
    arms: list[dict[str, Any]] = []
    for (model, _effort), group in by_agent.items():
        latest = _latest(group)
        arm = _arm(latest)
        arm["model"] = model
        arm["n_scored"] = _int((latest or {}).get("metrics", {}).get("n_scored"))
        arms.append(arm)
    arms.sort(
        key=lambda a: (
            (a["model"], a["effort"]) != reference_key,
            -(a.get("accuracy") or 0.0),
        )
    )
    reference = next(
        (a for a in arms if (a["model"], a["effort"]) == reference_key), None
    )
    train_runs = _train_runs_by_agent(
        [r for r in eval_records if of_champion(r)], reference_model
    )
    pairs: list[dict[str, Any]] = []
    for arm in arms:
        if reference is None or arm is reference:
            continue
        pair: dict[str, Any] = {
            "version": sdk_champion,
            "baseline_model": reference_model,
            "baseline_effort": reference_effort,
            "candidate_model": arm["model"],
            "candidate_effort": arm["effort"],
            "baseline_run": reference.get("run_name"),
            "candidate_run": arm.get("run_name"),
            "n_compared": None,
            "delta_pp": None,
            "p_value": None,
            "p_worse": None,
            "p_better": None,
            "ci": [None, None],
            "fixed": None,
            "broken": None,
            "significant": None,
            "by_turn_type": None,
            "ceiling_errors": None,
        }
        if read_csvs and reference.get("run_name") and arm.get("run_name"):
            pair.update(_paired_from_csvs(reference["run_name"], arm["run_name"]))
        pair["train"] = _train_pairs(
            train_runs.get(reference_key, {}),
            train_runs.get((arm["model"], arm["effort"]), {}),
            read_csvs=read_csvs,
        )
        pair["verdict"] = model_swap_verdict(pair, margin_pp=margin_pp)
        pairs.append(pair)
    return {
        "version": sdk_champion,
        "reference_model": reference_model,
        "reference_effort": reference_effort,
        "margin_pp": margin_pp,
        "models": arms,
        "pairs": pairs,
    }


def _train_runs_by_agent(
    records: list[dict[str, Any]], reference_model: str
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    """Early-stopped train draws per agent, keyed by seed: the latest run of each.

    Only runs with a ``train_draw_seed`` and ``stop_at_first_wrong`` qualify —
    that is how sdk_v1's own train evidence was recorded (campaign s01), and a
    challenger pairs with it by repeating both. A run on the manifest's train
    list, or one that ran conversations to the end, is a different question
    set or a different protocol, and pairs with nothing here.
    """
    out: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for r in records:
        params = r.get("params") or {}
        seed = params.get("train_draw_seed")
        if params.get("split") != "train" or not seed:
            continue
        if str(params.get("stop_at_first_wrong")).lower() != "true":
            continue
        out.setdefault(_agent_key(r, reference_model), {}).setdefault(
            str(seed), []
        ).append(r)
    return {
        key: {
            seed: latest for seed, group in seeds.items() if (latest := _latest(group))
        }
        for key, seeds in out.items()
    }


def _train_pairs(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    read_csvs: bool = True,
) -> dict[str, Any]:
    """Per-seed conversation-level pairs, and their pooled exact McNemar."""
    by_seed: list[dict[str, Any]] = []
    for seed in sorted(set(baseline) & set(candidate)):
        row: dict[str, Any] = {
            "seed": seed,
            "baseline_run": baseline[seed].get("run_name"),
            "candidate_run": candidate[seed].get("run_name"),
        }
        if read_csvs:
            row.update(
                _train_pair_from_csvs(
                    str(baseline[seed].get("run_name")),
                    str(candidate[seed].get("run_name")),
                )
            )
        by_seed.append(row)
    measured = [r for r in by_seed if r.get("n_conversations")]
    pooled: dict[str, Any] = {"n_draws": len(measured)}
    if measured:
        from convfinqa.tracking.comparator import mcnemar_exact_p_one_sided

        fixed = sum(int(r["fixed"]) for r in measured)
        broken = sum(int(r["broken"]) for r in measured)
        pooled.update(
            {
                "n_conversations": sum(int(r["n_conversations"]) for r in measured),
                "baseline_full": sum(int(r["baseline_full"]) for r in measured),
                "candidate_full": sum(int(r["candidate_full"]) for r in measured),
                "fixed": fixed,
                "broken": broken,
                "p_better": round(mcnemar_exact_p_one_sided(broken, fixed), 8),
                "p_worse": round(mcnemar_exact_p_one_sided(fixed, broken), 8),
            }
        )
    return {"by_seed": by_seed, "pooled": pooled}


def _train_pair_from_csvs(baseline_run: str, candidate_run: str) -> dict[str, Any]:
    """One train draw, paired per conversation: answered right to the end, or not.

    Both runs stop at the first wrong answer, so a later turn exists in one CSV
    only when that arm got every earlier turn right — pairing questions would
    compare each arm on a set chosen by its own mistakes. The conversation is
    the unit both runs share. Returns {} when a CSV is missing, its report set
    differs, or it carries unscored rows (an incomplete pass is not measured).
    """
    from convfinqa.evalloop.runner import PREDICTIONS_DIR
    from convfinqa.tracking.comparator import mcnemar_exact_p_one_sided

    base = PREDICTIONS_DIR / f"{baseline_run}.csv"
    cand = PREDICTIONS_DIR / f"{candidate_run}.csv"
    if not base.exists() or not cand.exists():
        return {}
    import pandas as pd

    def outcomes(path: Path) -> dict[str, tuple[bool, int]] | None:
        frame = pd.read_csv(path)
        if (
            "unscored" in frame
            and frame["unscored"].astype(str).str.lower().eq("true").any()
        ):
            return None
        out: dict[str, tuple[bool, int]] = {}
        for report_id, rows in frame.groupby("report_id", sort=False):
            ok = (
                rows.sort_values("turn_index")["correct"].astype(str).str.lower()
                == "true"
            )
            right_before_error = int(ok.cumprod().sum())
            out[str(report_id)] = (bool(ok.all()), right_before_error)
        return out

    b, c = outcomes(base), outcomes(cand)
    if b is None or c is None or set(b) != set(c):
        return {}
    fixed = sum(1 for k in b if c[k][0] and not b[k][0])
    broken = sum(1 for k in b if b[k][0] and not c[k][0])
    n = len(b)
    return {
        "n_conversations": n,
        "baseline_full": sum(1 for v in b.values() if v[0]),
        "candidate_full": sum(1 for v in c.values() if v[0]),
        "fixed": fixed,
        "broken": broken,
        "p_better": round(mcnemar_exact_p_one_sided(broken, fixed), 8),
        "p_worse": round(mcnemar_exact_p_one_sided(fixed, broken), 8),
        "baseline_mean_right_before_error": round(sum(v[1] for v in b.values()) / n, 4),
        "candidate_mean_right_before_error": round(
            sum(v[1] for v in c.values()) / n, 4
        ),
    }


def model_swap_verdict(
    pair: dict[str, Any], *, margin_pp: float = MODEL_SWAP_MARGIN_PP
) -> dict[str, Any]:
    """`beats`, `matches` or `falls_short` for a challenger agent (s19's rule).

    - **beats**: test Δ > 0 with one-sided cluster p < α (the house gate), and
      train does not contradict it;
    - **matches**: test's 95% CI lower bound ≥ −`margin_pp`; test not
      significantly worse; program turns on test not significantly worse; the
      pooled train conversations not significantly worse; no more token-ceiling
      errors than the champion's run;
    - **falls_short**: anything else.

    ``outcome`` is None when test was not measured (a missing or incomplete
    CSV): no verdict is better than one computed on half a pass. Every check
    is listed with its value, so the page can say which one decided.
    """
    identity = {
        "version": pair.get("version"),
        "candidate_model": pair.get("candidate_model"),
        "candidate_effort": pair.get("candidate_effort"),
        # The test pass the verdict measured: `promote_agent` floors the new
        # champion on it, so the CI gate checks the agent that was promoted.
        "candidate_run": pair.get("candidate_run"),
    }
    if pair.get("n_compared") is None or pair.get("delta_pp") is None:
        return {
            **identity,
            "outcome": None,
            "reason": "test not measured — no complete paired pass yet",
            "margin_pp": margin_pp,
            "checks": [],
        }
    alpha = MODEL_SWAP_ALPHA
    delta = float(pair["delta_pp"])
    ci_low = pair.get("ci", [None, None])[0]
    ci_low_pp = None if ci_low is None else round(float(ci_low) * 100, 4)
    p_better = _num(pair.get("p_better"))
    p_worse = _num(pair.get("p_worse"))
    program = ((pair.get("by_turn_type") or {}).get("program")) or {}
    program_p_better = _num(program.get("cluster_p_one_sided"))
    program_p_worse = (
        None if program_p_better is None else round(1.0 - program_p_better, 6)
    )
    pooled = (pair.get("train") or {}).get("pooled") or {}
    train_p_worse = _num(pooled.get("p_worse"))
    ceiling = pair.get("ceiling_errors") or {}

    def check(name: str, ok: bool | None, detail: str) -> dict[str, Any]:
        return {"name": name, "ok": ok, "detail": detail}

    train_ok = None if train_p_worse is None else train_p_worse >= alpha
    checks = [
        check(
            "test_beats",
            p_better is not None and delta > 0 and p_better < alpha,
            f"Δ {delta:+.2f}pp, one-sided cluster p {p_better}",
        ),
        check(
            "test_ci_within_margin",
            ci_low_pp is not None and ci_low_pp >= -margin_pp,
            f"95% CI lower bound {ci_low_pp}pp vs −{margin_pp}pp",
        ),
        check(
            "test_not_worse",
            p_worse is not None and p_worse >= alpha,
            f"one-sided p toward worse {p_worse}",
        ),
        check(
            "program_turns_not_worse",
            program_p_worse is None or program_p_worse >= alpha,
            f"program turns: p toward worse {program_p_worse}",
        ),
        check(
            "train_not_worse",
            train_ok,
            f"conversations right to the end {pooled.get('baseline_full')} → "
            f"{pooled.get('candidate_full')} of {pooled.get('n_conversations')}, "
            f"p toward worse {train_p_worse}",
        ),
        check(
            "no_extra_ceiling_errors",
            int(ceiling.get("candidate") or 0) <= int(ceiling.get("baseline") or 0),
            f"token-ceiling errors {ceiling.get('baseline')} → {ceiling.get('candidate')}",
        ),
    ]
    by = {c["name"]: c["ok"] for c in checks}
    # Train is a veto, never a substitute: no train evidence leaves the train
    # check unknown, which neither blocks nor counts as passing evidence.
    train_vetoes = by["train_not_worse"] is False
    if by["test_beats"] and not train_vetoes:
        outcome = "beats"
    elif (
        by["test_ci_within_margin"]
        and by["test_not_worse"]
        and by["program_turns_not_worse"]
        and by["no_extra_ceiling_errors"]
        and not train_vetoes
    ):
        outcome = "matches"
    else:
        outcome = "falls_short"
    failed = [
        c["name"] for c in checks if c["ok"] is False and c["name"] != "test_beats"
    ]
    reason = (
        f"{outcome}: Δ {delta:+.2f}pp, CI low {ci_low_pp}pp (margin −{margin_pp}pp)"
        + (
            f"; failed {', '.join(failed)}"
            if outcome == "falls_short" and failed
            else ""
        )
        + ("; no train evidence" if train_ok is None else "")
    )
    return {
        **identity,
        "outcome": outcome,
        "reason": reason,
        "margin_pp": margin_pp,
        "checks": checks,
    }


def _paired_from_csvs(baseline_run: str, candidate_run: str) -> dict[str, Any]:
    """The gate's own statistics between two committed run CSVs, or {} when
    either is missing or incomplete — absence stays absence."""
    from convfinqa.evalloop.gate import gate_runs
    from convfinqa.evalloop.runner import PREDICTIONS_DIR

    base = PREDICTIONS_DIR / f"{baseline_run}.csv"
    cand = PREDICTIONS_DIR / f"{candidate_run}.csv"
    if not base.exists() or not cand.exists():
        return {}
    try:
        _, stats = gate_runs(
            base, cand, baseline_version=baseline_run, candidate_version=candidate_run
        )
    except Exception:  # noqa: BLE001 — an incomplete pass is "not measured"
        return {}
    # The gate's p is one-sided *towards improvement*, because the gate only
    # ever promotes. A scoring comparison has no privileged direction: a model
    # that is worse should read as significantly worse, not as "no effect
    # towards better". So the p reported here is one-sided in the direction of
    # the observed delta — Φ(−z) when the candidate is ahead, Φ(z) when behind
    # — and `significant` is that p against the same α.
    delta = float(stats["accuracy_delta"])
    z = float(stats.get("cluster_z") or 0.0)
    p_towards_better = float(stats["cluster_p_one_sided"])
    p_directed = p_towards_better if delta >= 0 else 1.0 - p_towards_better
    return {
        "n_compared": _int(stats.get("n_compared")),
        "delta_pp": round(delta * 100, 4),
        "cluster_z": round(z, 4),
        "p_value": round(p_directed, 6),
        "p_better": round(p_towards_better, 6),
        "p_worse": round(1.0 - p_towards_better, 6),
        "ci": [_num(stats.get("delta_ci_lo")), _num(stats.get("delta_ci_hi"))],
        "fixed": _int(stats.get("fail_to_pass")),
        "broken": _int(stats.get("pass_to_fail")),
        "significant": p_directed < float(stats.get("alpha") or 0.05),
        "by_turn_type": turn_type_gate(base, cand),
        "ceiling_errors": {
            "baseline": _ceiling_errors(base),
            "candidate": _ceiling_errors(cand),
        },
    }


def _ceiling_errors(path: Path) -> int:
    """Rows a pass lost to the per-conversation token ceiling."""
    import pandas as pd

    frame = pd.read_csv(path)
    if "error" not in frame:
        return 0
    return int(
        frame["error"].astype(str).str.startswith("token budget exhausted").sum()
    )


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


def with_program_accuracy(
    comparison: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Add each arm's program accuracy, read from its committed predictions CSV.

    Execution accuracy alone overstates what either arm is doing: both answer
    far more turns correctly than they reproduce gold programs for, and the SDK
    arm's headline sits above the paper's human-expert figure, which is exactly
    the claim a reader should be able to check against the program number. The
    figure is not in `story.json` (older stories predate it), so it is derived
    here from the same committed CSV the run is named after — no tracking
    server, no API calls, reproducible on any clone. Shared by the serving
    route and the published page so the two cannot disagree.

    Absent or unreadable CSV leaves the key `None`, never 0.0: "we did not
    measure it" and "it scored nothing" are different claims.
    """
    if not comparison:
        return comparison
    import pandas as pd

    from convfinqa.evalloop.runner import PREDICTIONS_DIR
    from convfinqa.tracking.comparator import program_accuracy

    out = dict(comparison)
    for arm in [k for k in out if k != "gate"]:
        row = out.get(arm)
        if not isinstance(row, dict):
            continue
        row = dict(row)
        out[arm] = row
        if row.get("program_accuracy") is not None:
            continue
        row["program_accuracy"] = None
        name = row.get("run_name")
        if not name:
            continue
        path = PREDICTIONS_DIR / f"{name}.csv"
        if not path.exists():
            continue
        try:
            row["program_accuracy"] = program_accuracy(pd.read_csv(path))[
                "program_accuracy"
            ]
        except Exception:  # noqa: BLE001 - a bad CSV must not break rendering
            continue
    return out


def with_model_program_accuracy(
    comparison: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """`with_program_accuracy` for the rows of `sdk_model_comparison`."""
    if not comparison or not comparison.get("models"):
        return comparison
    arms = {f"m{i}": row for i, row in enumerate(comparison["models"])}
    filled = with_program_accuracy({**arms, "gate": None}) or {}
    return {
        **comparison,
        "models": [filled.get(k, row) for k, row in arms.items()],
    }


def render(data: dict[str, Any]) -> str:
    """The published page. Import kept local so `collect` never needs it."""
    from convfinqa.evalloop.story_page import render_page

    return render_page(data)


def render_sdk(data: dict[str, Any]) -> str:
    """The Agent SDK experiment page, beside the campaign write-up."""
    from convfinqa.evalloop.story_page import render_sdk_page

    if data.get("runtime_comparison"):
        data = {
            **data,
            "runtime_comparison": with_program_accuracy(data["runtime_comparison"]),
        }
    if data.get("sdk_model_comparison"):
        data = {
            **data,
            "sdk_model_comparison": with_model_program_accuracy(
                data["sdk_model_comparison"]
            ),
        }
    return render_sdk_page(data)
