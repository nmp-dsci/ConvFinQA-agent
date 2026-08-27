"""GEPA training driver, metric helpers, and per-turn predictions CSV writer."""

# ruff: noqa: B905

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
import pandas as pd

from convfinqa.config import PREDICTIONS_DIR, settings
from convfinqa.evaluation import numeric_match


def conv_turn_accuracy(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """Fraction of turns in this conversation where pred matches gold."""
    golds = example.gold_answers
    preds = getattr(pred, "predictions", None) or []
    if not golds:
        return 0.0
    return sum(numeric_match(p, g) for p, g in zip(preds, golds)) / len(golds)


def conv_metric_with_feedback(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> dspy.Prediction:
    """GEPA-style metric: returns score plus per-turn feedback for reflection."""
    golds = example.gold_answers
    preds = getattr(prediction, "predictions", None) or []
    score = (
        sum(numeric_match(p, g) for p, g in zip(preds, golds)) / len(golds)
        if golds
        else 0.0
    )

    if not preds:
        feedback = (
            "The runner returned no predictions for this conversation, likely due to "
            "an LM/adapter parsing failure. The downstream agents could not produce "
            "structured outputs. Consider tightening output format instructions."
        )
        return dspy.Prediction(score=score, feedback=feedback)

    lines = [f"Conversation on report {example.report_id}:"]
    for i, (q, g) in enumerate(zip(example.questions, golds), start=1):
        p = preds[i - 1] if i <= len(preds) else "<missing>"
        ok = numeric_match(p, g)
        tag = "PASS" if ok else "FAIL"
        lines.append(f"  T{i} {tag}  Q: {q}")
        lines.append(f"        pred={p!r}  gold={g!r}")
    lines.append(
        "FAIL turns indicate either: wrong value retrieved, wrong DSL program, "
        "answer formatted with extraneous units (e.g. '$3.0 billion' instead of "
        "'3.0'), or unrounded float vs gold percent. Aim for plain numeric strings."
    )
    return dspy.Prediction(score=score, feedback="\n".join(lines))


def _eval_result_to_joined(eval_result: Any, *, model_label: str) -> pd.DataFrame:
    from convfinqa.data.loader import qa_data

    rows: list[dict[str, Any]] = []
    for ex, pred, _ in eval_result.results:
        preds = getattr(pred, "predictions", None) or []
        for i, (q, g) in enumerate(zip(ex.questions, ex.gold_answers)):
            p = preds[i] if i < len(preds) else None
            rows.append(
                {
                    "report_id": ex.report_id,
                    "turn_index": i,
                    "question": q,
                    "gold_answer": g,
                    "pred_answer": p,
                    "correct": numeric_match(p, g) if p is not None else False,
                    "model": model_label,
                }
            )
    preds = pd.DataFrame(rows)
    qa = qa_data.sort_values(["report_id", "q_order"]).copy()
    qa["turn_index"] = qa.groupby("report_id").cumcount()
    joined = preds.merge(
        qa[["report_id", "turn_index", "turn_type", "qa_split"]],
        on=["report_id", "turn_index"],
        how="inner",
    )
    joined["conv_type"] = joined["qa_split"].map({True: "Type II", False: "Type I"})
    return joined


def print_model_accuracy_table(
    joined_frames: list[pd.DataFrame],
    *,
    slice_col: str,
    title: str,
) -> None:
    """Print per-bucket accuracy across models from joined frames."""
    combined = pd.concat(joined_frames, ignore_index=True)
    rows: list[dict[str, Any]] = []

    overall: dict[str, Any] = {"bucket": "overall"}
    for model, frame in combined.groupby("model"):
        overall[f"{model}_acc"] = frame["correct"].mean()
    rows.append(overall)

    for bucket in sorted(combined[slice_col].dropna().unique()):
        row = {"bucket": bucket}
        for model, frame in combined.groupby("model"):
            cut = frame[frame[slice_col] == bucket]
            row[f"{model}_acc"] = cut["correct"].mean() if not cut.empty else None
        rows.append(row)

    out = pd.DataFrame(rows)
    printable = out.copy()
    for col in [c for c in printable.columns if c.endswith("_acc")]:
        printable[col] = printable[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    print(f"\n{title}:")  # noqa: T201
    print(printable.to_string(index=False))  # noqa: T201


def load_artifact_instructions(program_path: Path) -> dict[str, str]:
    """Load per-predictor instructions from a saved DSPy program artifact."""
    raw = json.loads(program_path.read_text())
    return {
        key: raw[key]["signature"]["instructions"].rstrip()
        for key in (
            "triage.predict",
            "preprocess.predict",
            "retriever.predict",
            "calculator.react",
        )
    }


def compare_runner_instructions(
    runner: Any,
    program_path: Path,
) -> dict[str, bool]:
    """Compare loaded predictor instructions against a saved program artifact."""
    expected = load_artifact_instructions(program_path)
    results: dict[str, bool] = {}
    print(f"\nInstruction comparison vs {program_path.name}:")  # noqa: T201
    for name, predictor in runner.named_predictors():
        if name not in expected:
            continue
        loaded = predictor.signature.instructions.rstrip()
        ok = loaded == expected[name]
        results[name] = ok
        status = "MATCH" if ok else "MISMATCH"
        print(f"  - {name:<10} {status}")  # noqa: T201
    return results


def write_predictions_csv(
    predictions_path: Path,
    eval_results: list[tuple[dspy.Example, dspy.Prediction, Any]],
) -> None:
    """Write per-turn predictions plus predicted turn labels for inspection."""
    with predictions_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "report_id",
            "turn_index",
            "question",
            "gold_answer",
            "pred_answer",
            "correct",
            "pred_turn_type",
            "pred_conv_type",
        ])
        for ex, pred, _ in eval_results:
            preds = getattr(pred, "predictions", None) or []
            responses = getattr(pred, "responses", None) or []
            for i, (q, g) in enumerate(zip(ex.questions, ex.gold_answers)):
                p = preds[i] if i < len(preds) else None
                response = responses[i] if i < len(responses) else None
                w.writerow([
                    ex.report_id,
                    i,
                    q,
                    g,
                    p,
                    numeric_match(p, g) if p is not None else False,
                    getattr(response, "turn_type", None),
                    getattr(response, "conv_type", None),
                ])


def main() -> None:
    """Baseline eval + optional GEPA training, mirroring dspy_agent.py:__main__."""
    # ruff: noqa: T201
    from convfinqa.backends.dspy import (
        ConversationRunner,
        conv_examples_test,
        conv_examples_train,
        lm_max,
    )
    from convfinqa.evaluation.joining import analyze_predictions

    test_set = conv_examples_test
    total_turns = sum(len(ex.questions) for ex in test_set)
    print(f"Test set: {len(test_set)} conversations, {total_turns} turns total")

    evaluator = dspy.Evaluate(
        devset=test_set,
        metric=conv_turn_accuracy,
        num_threads=8,
        display_progress=True,
    )
    eval_result = evaluator(ConversationRunner())

    print(f"\nOverall turn accuracy: {eval_result.score:.1f}%")
    print("\nPer-conversation:")
    n_errored = 0
    for ex, pred, s in eval_result.results:
        n_turns = len(ex.questions)
        preds = getattr(pred, "predictions", None)
        if preds is None:
            n_errored += 1
            print(f"  {ex.report_id:<45}  ERRORED ({n_turns} turns skipped)")
            continue
        n_pass = sum(numeric_match(p, g) for p, g in zip(preds, ex.gold_answers))
        print(f"  {ex.report_id:<45}  {n_pass}/{n_turns} turns  ({s:.0%})")
    if n_errored:
        print(
            f"\n{n_errored} conversation(s) errored (LM adapter failures); "
            "they count as 0 in the overall score."
        )

    if not settings.run_gepa:
        print("\n(Skipping GEPA. Set RUN_GEPA=1 to compile an optimized runner.)")
        return

    gepa_mode = settings.gepa_mode.lower()
    if gepa_mode not in {"smoke", "real"}:
        raise RuntimeError(f"GEPA_MODE must be 'smoke' or 'real', got {gepa_mode!r}")

    if gepa_mode == "smoke":
        n_val = 5
        gepa_kwargs: dict[str, Any] = {"max_metric_calls": 120}
    else:
        n_val = 12
        gepa_kwargs = {"auto": "light"}

    gepa_name = settings.gepa_name
    resume_target = settings.resume_gepa
    if gepa_name and resume_target:
        raise RuntimeError("GEPA_NAME and RESUME_GEPA are mutually exclusive")

    run_dir: Path | None = None
    existing_program: Path | None = None
    if not resume_target:
        if gepa_name:
            run_dir = Path("runs") / gepa_name
            existing_program = run_dir / "dspy_optimized_runner.json"
            if not existing_program.exists():
                existing_program = run_dir / "optimized_runner.json"
        else:
            candidate_dirs = sorted(
                Path("runs").glob(f"gepa_{gepa_mode}_*"), key=lambda p: p.name
            )
            for candidate in reversed(candidate_dirs):
                candidate_program = candidate / "dspy_optimized_runner.json"
                if not candidate_program.exists():
                    candidate_program = candidate / "optimized_runner.json"
                if candidate_program.exists():
                    run_dir = candidate
                    existing_program = candidate_program
                    break

    if existing_program and run_dir:
        print(f"\nFound {existing_program} — skipping GEPA, loading and evaluating.")
        optimized_runner = ConversationRunner()
        optimized_runner.load(str(existing_program))
        compare_runner_instructions(optimized_runner, existing_program)
        opt_eval_result = dspy.Evaluate(
            devset=test_set,
            metric=conv_turn_accuracy,
            num_threads=8,
            display_progress=True,
        )(optimized_runner)
        print(f"\nBaseline turn accuracy:  {eval_result.score:.1f}%")
        print(f"Optimized turn accuracy: {opt_eval_result.score:.1f}%")
        print(f"Δ = {opt_eval_result.score - eval_result.score:+.1f} pts")
        baseline_joined = _eval_result_to_joined(eval_result, model_label="baseline")
        optimized_joined = _eval_result_to_joined(opt_eval_result, model_label="optimized")
        print_model_accuracy_table(
            [baseline_joined, optimized_joined],
            slice_col="turn_type",
            title="Turn Type Accuracy by Model",
        )
        print_model_accuracy_table(
            [baseline_joined, optimized_joined],
            slice_col="conv_type",
            title="Conv Type Accuracy by Model",
        )

        version = settings.version
        eval_dir = PREDICTIONS_DIR
        eval_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = eval_dir / f"dspy_predictions_{version}.csv"
        write_predictions_csv(predictions_path, opt_eval_result.results)
        print(f"\nWrote {predictions_path}")
        analyze_predictions(predictions_path)
        return

    if gepa_name:
        raise RuntimeError(
            f"GEPA_NAME={gepa_name!r} was set, but no saved program exists. "
            "GEPA_NAME is only for load-and-evaluate of an existing run."
        )

    if resume_target == "latest":
        matches = sorted(Path("runs").glob(f"gepa_{gepa_mode}_*"), key=lambda p: p.name)
        if not matches:
            raise RuntimeError(
                f"RESUME_GEPA=latest with GEPA_MODE={gepa_mode} but no "
                f"runs/gepa_{gepa_mode}_* dirs exist"
            )
        run_dir = matches[-1]
        is_resume = True
    elif resume_target:
        run_dir = Path(resume_target)
        if not run_dir.exists() and not run_dir.is_absolute():
            candidate = Path("runs") / resume_target
            if candidate.exists():
                run_dir = candidate
        if not run_dir.exists():
            raise RuntimeError(
                f"RESUME_GEPA={resume_target} does not exist "
                f"(checked {Path(resume_target)} and {Path('runs') / resume_target})"
            )
        is_resume = True
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"gepa_{gepa_mode}_{run_ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        is_resume = False

    print("\n" + "=" * 60)
    print(f"GEPA mode: {gepa_mode.upper()}  ({'resuming' if is_resume else 'new'}: {run_dir})")
    print("=" * 60)

    gepa_trainset = conv_examples_train[n_val:]
    gepa_valset = conv_examples_train[:n_val]
    print(f"trainset: {len(gepa_trainset)} conv | valset: {len(gepa_valset)} conv")
    print(f"GEPA kwargs: {gepa_kwargs}")

    run_config = {
        "mode": gepa_mode,
        "gepa_kwargs": gepa_kwargs,
        "trainset_report_ids": [ex.report_id for ex in gepa_trainset],
        "valset_report_ids": [ex.report_id for ex in gepa_valset],
        "num_preds": len(ConversationRunner().predictors()),
    }
    config_path = run_dir / "config.json"
    if is_resume:
        if not config_path.exists():
            raise RuntimeError(
                f"{run_dir} has no config.json — was this dir created by an "
                "older agent.py? Resume is not safe without a recorded config."
            )
        saved_config = json.loads(config_path.read_text())
        mismatches = [k for k in run_config if saved_config.get(k) != run_config[k]]
        if mismatches:
            raise RuntimeError(
                f"Cannot resume {run_dir} — config differs from saved state on "
                f"{mismatches}. Resume requires identical mode/trainset/valset/num_preds."
            )
        print("Config matches saved state — resuming.")
    else:
        config_path.write_text(json.dumps(run_config, indent=2, default=str))

    optimizer = dspy.GEPA(
        metric=conv_metric_with_feedback,
        num_threads=8,
        track_stats=True,
        track_best_outputs=True,
        log_dir=str(run_dir / "dspy_gepa_logs"),
        reflection_minibatch_size=3,
        reflection_lm=lm_max,
        **gepa_kwargs,
    )

    optimized_runner = optimizer.compile(
        ConversationRunner(),
        trainset=gepa_trainset,
        valset=gepa_valset,
    )

    opt_eval_result = dspy.Evaluate(
        devset=test_set,
        metric=conv_turn_accuracy,
        num_threads=8,
        display_progress=True,
    )(optimized_runner)

    print(f"\nBaseline turn accuracy:  {eval_result.score:.1f}%")
    print(f"Optimized turn accuracy: {opt_eval_result.score:.1f}%")
    print(f"Δ = {opt_eval_result.score - eval_result.score:+.1f} pts")
    baseline_joined = _eval_result_to_joined(eval_result, model_label="baseline")
    optimized_joined = _eval_result_to_joined(opt_eval_result, model_label="optimized")
    print_model_accuracy_table(
        [baseline_joined, optimized_joined],
        slice_col="turn_type",
        title="Turn Type Accuracy by Model",
    )
    print_model_accuracy_table(
        [baseline_joined, optimized_joined],
        slice_col="conv_type",
        title="Conv Type Accuracy by Model",
    )

    program_path = run_dir / "dspy_optimized_runner.json"
    optimized_runner.save(str(program_path))
    compare_runner_instructions(optimized_runner, program_path)

    stats_path = run_dir / "dspy_gepa_stats.json"
    details = optimized_runner.detailed_results
    cand_instructions = [
        {name: pred.signature.instructions for name, pred in cand.named_predictors()}
        for cand in details.candidates
    ]
    stats = dict(
        candidates=cand_instructions,
        parents=details.parents,
        val_aggregate_scores=details.val_aggregate_scores,
        val_subscores=details.val_subscores,
        per_val_instance_best_candidates=[
            list(s) if hasattr(s, "__iter__") else s
            for s in details.per_val_instance_best_candidates
        ],
        discovery_eval_counts=details.discovery_eval_counts,
        total_metric_calls=details.total_metric_calls,
        num_full_val_evals=details.num_full_val_evals,
        log_dir=details.log_dir,
        seed=details.seed,
        best_idx=details.best_idx,
    )
    stats_path.write_text(json.dumps(stats, indent=2, default=str))

    summary_path = run_dir / "dspy_summary.json"
    baseline = ConversationRunner()
    instr_diff = {
        name: {
            "baseline": baseline_pred.signature.instructions,
            "optimized": opt_pred.signature.instructions,
        }
        for (name, opt_pred), (_, baseline_pred) in zip(
            optimized_runner.named_predictors(),
            baseline.named_predictors(),
            strict=True,
        )
    }
    summary = {
        "run_tag": run_dir.name,
        "mode": gepa_mode,
        "gepa_kwargs": gepa_kwargs,
        "resumed": is_resume,
        "trainset_size": len(gepa_trainset),
        "valset_size": len(gepa_valset),
        "testset_size": len(test_set),
        "baseline_test_score": eval_result.score,
        "optimized_test_score": opt_eval_result.score,
        "delta_pts": opt_eval_result.score - eval_result.score,
        "total_metric_calls": stats.get("total_metric_calls"),
        "num_full_val_evals": stats.get("num_full_val_evals"),
        "best_candidate_idx": stats.get("best_idx"),
        "predictor_instructions": instr_diff,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    version = settings.version
    eval_dir = PREDICTIONS_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = eval_dir / f"dspy_predictions_{version}.csv"
    write_predictions_csv(predictions_path, opt_eval_result.results)

    print(f"\nOptimization artifacts under {run_dir}/")
    print(f"  - dspy_optimized_runner.json ({program_path.stat().st_size:,} bytes)")
    print(f"  - dspy_gepa_stats.json      ({stats_path.stat().st_size:,} bytes)")
    print("  - dspy_summary.json         (human-readable diff + scores)")
    print("  - dspy_gepa_logs/         (GEPA's per-iteration logs)")
    print(f"Evaluation predictions: {predictions_path}")

    analyze_predictions(predictions_path)
