"""The gate for the single-session arm: overall accuracy decides, the panel watches.

The pipeline's targeted gate (`teacher.gate_targeted`) judges one subagent's
own panel metric beside the overall rule, because one experiment there changes
one subagent. The SDK arm rewrites one prompt, possibly in several places, so
there is no single agent whose metric the change is *for* — the verdict is the
overall paired rule alone (`comparator.promotable_significant`: net positive
AND one-sided cluster-corrected McNemar p < 0.05, same as `gate.py`). The
per-stage panel is still computed for both arms and travels with the verdict,
because the qa_agent reports its stages and "which area regressed" is worth
reading even when it decides nothing.

`log_gate_verdict` is this arm's counterpart of `teacher.log_gate_verdict`. It
exists as a separate function for one reason: the gates ledger has a `runtime`
column and the pipeline logger writes ``multi_agent`` into it unconditionally.
A row is append-only, so writing the wrong runtime is not a cosmetic slip — it
would pool an SDK verdict into the pipeline's history forever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from convfinqa.evalloop import ledgers
from convfinqa.evalloop.teacher import OPTIMIZATION_EXPERIMENT

RUNTIME = "agent_sdk"


def gate_overall(
    baseline_csv: Path | str,
    candidate_csv: Path | str,
    *,
    baseline_version: str,
    candidate_version: str,
    target_class: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """Paired comparison of two SDK-arm runs; the same verdict shape as the pipeline.

    Returns ``(verdict, comparison)``. The verdict carries `target_agent` as an
    empty string and `target_class` as the failure class the rewrite was aimed
    at, so the ledger row builder and the story can read either arm's verdict
    without branching on which produced it.
    """
    from convfinqa.evalloop import stage_scores
    from convfinqa.evalloop.gate import gate_reason, gate_runs, load_run_csv

    result, stats = gate_runs(
        baseline_csv,
        candidate_csv,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
    )
    base_panel = stage_scores.run_metrics(load_run_csv(baseline_csv))
    cand_panel = stage_scores.run_metrics(load_run_csv(candidate_csv))
    verdict = {
        "runtime": RUNTIME,
        "target_agent": "",
        "target_class": target_class or "",
        "target_metric": "accuracy",
        "target_metric_before": stats["baseline_accuracy"],
        "target_metric_after": stats["candidate_accuracy"],
        "target_metric_delta": stats["accuracy_delta"],
        "target_moved": bool(stats["accuracy_delta"] > 0),
        "target_evidence": (
            f"overall accuracy {stats['baseline_accuracy']:.3f} → "
            f"{stats['candidate_accuracy']:.3f}"
        ),
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
        "overall_delta": stats["accuracy_delta"],
        "evidence_split": stats["evidence_split"],
        "promotable": stats["promotable"],
        "cluster_p_one_sided": stats["cluster_p_one_sided"],
        "agent_panel_baseline": base_panel,
        "agent_panel_candidate": cand_panel,
        "comparison": stats,
        "reason": gate_reason(stats),
        "baseline_csv": str(baseline_csv),
        "candidate_csv": str(candidate_csv),
    }
    return verdict, result


def log_gate_verdict(
    verdict: dict[str, Any],
    *,
    comparison: Any = None,
    campaign: str | None = None,
    label: str | None = None,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    rewrite_id: str | None = None,
    consecutive_rejections: int | None = None,
    champion_after: str | None = None,
) -> str:
    """Record one SDK-arm gate decision: an MLflow run and a gates-ledger row.

    Same artifacts, metrics and tags as the pipeline logger (`verdict.json`,
    `flips.json`, `ledger_rows.jsonl`), plus ``runtime=agent_sdk`` on the run
    and on the ledger row, and `target_class` where the pipeline has
    `target_agent`. `champion_after` is what `sdk_champion` will be once the
    caller acts on the verdict; it defaults to the current alias, which is
    right only when the row is written after promotion.
    """
    from convfinqa.evalloop.gate import load_run_csv
    from convfinqa.tracking import mlflow_log, registry

    stats = verdict["comparison"]
    target_class = str(verdict.get("target_class") or "")
    if champion_after is None:
        champion_after = registry.sdk_champion() or ""
    with mlflow_log.run(
        f"sdk-gate-{verdict['candidate_version']}-vs-{verdict['baseline_version']}",
        kind="gate",
        version=verdict["candidate_version"],
        params={
            "runtime": RUNTIME,
            "baseline_version": verdict["baseline_version"],
            "candidate_version": verdict["candidate_version"],
            "target_agent": "",
            "target_class": target_class,
            "evidence_split": verdict["evidence_split"],
            **({"campaign": campaign} if campaign else {}),
            **({"experiment_label": label} if label else {}),
        },
        tags={
            "loop": "evalloop",
            "stage": "gate",
            "runtime": RUNTIME,
            "promoted": "true" if verdict["promotable"] else "false",
            "target_class": target_class,
            **({"campaign": campaign} if campaign else {}),
        },
        experiment=experiment,
        omit_fingerprint=("lm_max", "lm_mini"),
    ) as rec:
        rec.dict_artifact(
            "verdict.json",
            {
                "promoted": bool(verdict["promotable"]),
                "reason": verdict["reason"],
                "runtime": RUNTIME,
                "target_class": target_class,
                **{
                    k: stats[k]
                    for k in (
                        "accuracy_delta",
                        "cluster_p_one_sided",
                        "n_compared",
                        "fail_to_pass",
                        "pass_to_fail",
                        "delta_ci_lo",
                        "delta_ci_hi",
                        "delta_p_positive",
                    )
                    if k in stats
                },
            },
        )
        flips = (
            {
                "broken": [f.as_dict() for f in comparison.regressions],
                "fixed": [f.as_dict() for f in comparison.improvements],
            }
            if comparison is not None
            else None
        )
        if flips is not None:
            rec.dict_artifact("flips.json", flips)
        rec.metrics(
            {
                "accuracy_delta": float(stats["accuracy_delta"]),
                "cluster_p_one_sided": float(stats["cluster_p_one_sided"]),
                "n_compared": float(stats["n_compared"]),
                "fail_to_pass": float(stats["fail_to_pass"]),
                "pass_to_fail": float(stats["pass_to_fail"]),
                "delta_ci_lo": float(stats["delta_ci_lo"]),
                "delta_ci_hi": float(stats["delta_ci_hi"]),
                "promoted": 1.0 if verdict["promotable"] else 0.0,
            }
        )
        base_csv = verdict.get("baseline_csv")
        cand_csv = verdict.get("candidate_csv")
        attribution: ledgers.AttributionOf | None = None
        if flips is not None and base_csv and cand_csv:
            try:
                attribution = ledgers.attribution_from_frames(
                    load_run_csv(base_csv), load_run_csv(cand_csv)
                )
            except Exception:  # noqa: BLE001 — bookkeeping must not sink a gate
                attribution = None
        row = ledgers.gate_row(
            stats,
            baseline_version=verdict["baseline_version"],
            candidate_version=verdict["candidate_version"],
            promoted=bool(verdict["promotable"]),
            reason=str(verdict["reason"]),
            flips=flips if attribution is not None else None,
            attribution_of_row=attribution,
            panel_baseline=verdict.get("agent_panel_baseline") or {},
            panel_candidate=verdict.get("agent_panel_candidate") or {},
            baseline_hash=sdk_prompt_hash(verdict["baseline_version"]),
            candidate_hash=sdk_prompt_hash(verdict["candidate_version"]),
            baseline_eval_run_id=ledgers.eval_run_ids(base_csv),
            candidate_eval_run_id=ledgers.eval_run_ids(cand_csv),
            gate_run_id=str(rec.run_id),
            rewrite_id=rewrite_id,
            runtime=RUNTIME,
            campaign=campaign,
            label=label,
            consecutive_rejections=consecutive_rejections,
            champion_after=champion_after,
        )
        ledgers.log_rows_to_run(rec, ledgers.append("gates", [row]), "gates")
        return str(rec.run_id)


def sdk_prompt_hash(version: str) -> str:
    """The whole-prompt hash of an `sdk_vN` version, or "" when unresolvable."""
    try:
        from convfinqa.tracking import prompt_ledger

        return str(prompt_ledger.resolve_sdk(version)["hash"])
    except Exception:  # noqa: BLE001 — an unwritten version is an empty cell
        return ""
