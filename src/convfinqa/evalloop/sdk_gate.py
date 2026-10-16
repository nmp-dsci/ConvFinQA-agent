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

`gate_challenger` (s21) is the verdict for a challenger *agent* — a prompt
written for another model than the champion's. Its baseline is the champion's
own committed test pass (`champion_agent.evidence_run`), never a re-run, and
its rule is s19's model-swap verdict (`story.model_swap_verdict`: beats, or
matches within the 3pp margin, with train as a veto), not the house gate: a
challenger on a cheaper model is asked to hold the line, not to beat it.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    promoted: bool | None = None,
    extra_artifacts: Mapping[str, Any] | None = None,
) -> str:
    """Record one SDK-arm gate decision: an MLflow run and a gates-ledger row.

    Same artifacts, metrics and tags as the pipeline logger (`verdict.json`,
    `flips.json`, `ledger_rows.jsonl`), plus ``runtime=agent_sdk`` on the run
    and on the ledger row, and `target_class` where the pipeline has
    `target_agent`. Call it **after** the promotion was tried: `promoted` is
    what the registry did, `gate_passed` (the verdict's, else `promotable`)
    what the rule said, and `champion_after` defaults to the champion as it
    now stands. With `promoted` omitted it is read off `champion_after`. Before
    2026-10-15 this row was written first and `promoted` meant "the rule
    passed", so a refused promotion left one on record that never happened.
    """
    from convfinqa.evalloop.gate import load_run_csv
    from convfinqa.tracking import mlflow_log, registry

    stats = verdict["comparison"]
    target_class = str(verdict.get("target_class") or "")
    gate_passed = bool(verdict.get("gate_passed", verdict["promotable"]))
    if champion_after is None:
        champion_after = registry.sdk_champion() or ""
    if promoted is None:
        promoted = champion_after == verdict["candidate_version"]
    rule = str(verdict.get("rule") or "house")
    outcome = str(verdict.get("outcome") or "")
    base_agent = tuple(verdict.get("baseline_agent") or ("", ""))
    cand_agent = tuple(verdict.get("candidate_agent") or ("", ""))
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
            "verdict_rule": rule,
            **({"verdict_outcome": outcome} if outcome else {}),
            **({"baseline_sdk_model": base_agent[0]} if base_agent[0] else {}),
            **({"baseline_sdk_effort": base_agent[1]} if base_agent[1] else {}),
            **({"candidate_sdk_model": cand_agent[0]} if cand_agent[0] else {}),
            **({"candidate_sdk_effort": cand_agent[1]} if cand_agent[1] else {}),
            **({"campaign": campaign} if campaign else {}),
            **({"experiment_label": label} if label else {}),
        },
        tags={
            "loop": "evalloop",
            "stage": "gate",
            "runtime": RUNTIME,
            "promoted": "true" if promoted else "false",
            "gate_passed": "true" if gate_passed else "false",
            "target_class": target_class,
            **({"campaign": campaign} if campaign else {}),
        },
        experiment=experiment,
        omit_fingerprint=("lm_max", "lm_mini"),
    ) as rec:
        rec.dict_artifact(
            "verdict.json",
            {
                "promoted": bool(promoted),
                "gate_passed": gate_passed,
                "rule": rule,
                "outcome": outcome,
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
                "promoted": 1.0 if promoted else 0.0,
                "gate_passed": 1.0 if gate_passed else 0.0,
            }
        )
        for name, payload in (extra_artifacts or {}).items():
            rec.dict_artifact(name, payload)
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
            promoted=bool(promoted),
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
            gate_passed=gate_passed,
            verdict_rule=rule,
            verdict_outcome=outcome,
            baseline_agent=(str(base_agent[0]), str(base_agent[1])),
            candidate_agent=(str(cand_agent[0]), str(cand_agent[1])),
        )
        ledgers.log_rows_to_run(rec, ledgers.append("gates", [row]), "gates")
        return str(rec.run_id)


def train_seed_of(csv_path: Path | str) -> str | None:
    """The ``train_draw_seed`` the pass that wrote `csv_path` drew, if recorded."""
    from convfinqa.evalloop.sdk_teacher import _eval_run_record

    record = _eval_run_record(ledgers.eval_run_ids(csv_path))
    seed = (record or {}).get("params", {}).get("train_draw_seed")
    return str(seed) if seed else None


def gate_challenger(
    candidate_csv: Path | str,
    *,
    candidate_version: str,
    sdk_model: str,
    sdk_effort: str,
    candidate_train: Mapping[str, Path | str] | None = None,
    target_class: str | None = None,
    base_agent_csv: Path | str | None = None,
    eval_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Any]:
    """A challenger agent against the champion's committed evidence; the s19 rule decides.

    The baseline arm is `champion_agent.evidence_run` — the exact test pass
    the champion's floor stands on — and the train arms are the champion's own
    early-stopped draws, paired with `candidate_train` (``{seed: csv}``) on the
    seeds both drew. The champion is never run. The house statistics (Δ, the
    one-sided cluster p, the CI, the flips) are still computed against the
    champion and fill the ledger row; what decides is `model_swap_verdict`:
    ``beats`` or ``matches`` pass, ``falls_short`` does not.

    `base_agent_csv` is the challenger's own previous prompt on the same agent
    (sdk_v1 on Haiku 5.5, for sdk_v3 on Haiku 5.5). It decides nothing; it
    answers "did the rewrite help this model", which the verdict against the
    champion cannot.
    """
    from convfinqa.evalloop import story
    from convfinqa.evalloop.gate import gate_runs
    from convfinqa.evalloop.runner import PREDICTIONS_DIR
    from convfinqa.tracking import registry

    champion = registry.champion_agent() or {}
    evidence_run = str(champion.get("evidence_run") or "")
    if not evidence_run:
        raise SystemExit(
            "the champion record names no evidence_run — record it with "
            "`convfinqa-mlflow champion-evidence <csv>`; a challenger is paired "
            "against the champion's committed pass, never a re-run"
        )
    champion_version = str(champion.get("version"))
    champion_agent = (str(champion.get("model")), str(champion.get("effort")))
    if (sdk_model, sdk_effort) == champion_agent:
        raise SystemExit(
            f"{sdk_model} · {sdk_effort} is the champion's own agent; a prompt "
            "challenger on it is judged by the house gate (`gate_overall`)"
        )
    base_csv = PREDICTIONS_DIR / f"{evidence_run}.csv"
    verdict, comparison = gate_overall(
        base_csv,
        candidate_csv,
        baseline_version=champion_version,
        candidate_version=candidate_version,
        target_class=target_class,
    )
    records = eval_records if eval_records is not None else _records()
    champion_train = story._train_runs_by_agent(
        [
            r
            for r in records
            if (r.get("params") or {}).get("runtime") == RUNTIME
            and (r.get("params") or {}).get("prompts_version") == champion_version
        ],
        champion_agent[0],
    ).get(champion_agent, {})
    pair = story.challenger_pair(
        version=candidate_version,
        baseline={
            "version": champion_version,
            "model": champion_agent[0],
            "effort": champion_agent[1],
            "run_name": evidence_run,
        },
        candidate={
            "model": sdk_model,
            "effort": sdk_effort,
            "run_name": Path(candidate_csv).stem,
        },
        baseline_train=champion_train,
        candidate_train={
            str(seed): {"run_name": Path(csv).stem}
            for seed, csv in (candidate_train or {}).items()
        },
    )
    swap = pair["verdict"]
    passed = swap.get("outcome") in registry.AGENT_PROMOTING_OUTCOMES
    house = str(verdict["reason"])
    verdict.update(
        {
            "rule": "model_swap",
            "outcome": swap.get("outcome") or "",
            "gate_passed": passed,
            # `promotable` is what every older reader of a verdict acts on, so
            # it carries the rule that decided here.
            "promotable": passed,
            "baseline_agent": champion_agent,
            "candidate_agent": (sdk_model, sdk_effort),
            "challenger_pair": pair,
            "challenger_verdict": swap,
            "house_reason": house,
            "reason": (
                f"challenger {candidate_version} · {sdk_model} · {sdk_effort} vs "
                f"champion {champion_version} · {champion_agent[0]} · "
                f"{champion_agent[1]}: {swap.get('reason')}"
            ),
        }
    )
    if base_agent_csv:
        _, base_stats = gate_runs(
            base_agent_csv,
            candidate_csv,
            baseline_version=Path(base_agent_csv).stem,
            candidate_version=candidate_version,
        )
        verdict["vs_base_agent"] = {
            "baseline_run": Path(base_agent_csv).stem,
            "baseline_accuracy": base_stats["baseline_accuracy"],
            "candidate_accuracy": base_stats["candidate_accuracy"],
            "delta_pp": round(float(base_stats["accuracy_delta"]) * 100, 4),
            "p_better": base_stats["cluster_p_one_sided"],
            "ci": [base_stats.get("delta_ci_lo"), base_stats.get("delta_ci_hi")],
            "fixed": base_stats["fail_to_pass"],
            "broken": base_stats["pass_to_fail"],
            "by_turn_type": story.turn_type_gate(base_agent_csv, candidate_csv),
        }
    return verdict, comparison


def _records() -> list[dict[str, Any]]:
    """Every eval run on record; the committed snapshot alone when no store answers."""
    from convfinqa.evalloop import story

    try:
        return story.all_eval_records()
    except Exception:  # noqa: BLE001 — the champion's draws are in the snapshot
        return story._with_snapshot_runs([])


def sdk_prompt_hash(version: str) -> str:
    """The whole-prompt hash of an `sdk_vN` version, or "" when unresolvable."""
    try:
        from convfinqa.tracking import prompt_ledger

        return str(prompt_ledger.resolve_sdk(version)["hash"])
    except Exception:  # noqa: BLE001 — an unwritten version is an empty cell
        return ""
