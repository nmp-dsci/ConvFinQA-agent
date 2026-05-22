"""Join predictions CSVs to qa_data slices; cross-run parity reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.data.loader import qa_data


def write_joined_predictions(csv_path: Path) -> Path:
    """Write a sibling `*_joined.csv` with q_order, turn_type, conv_type merged in."""
    preds = pd.read_csv(csv_path)
    qa = qa_data.sort_values(["report_id", "q_order"]).copy()
    qa["turn_index"] = qa.groupby("report_id").cumcount()
    joined = preds.merge(
        qa[["report_id", "turn_index", "q_order", "turn_type", "qa_split"]],
        on=["report_id", "turn_index"],
        how="inner",
    )
    joined["conv_type"] = joined["qa_split"].map({True: "Type II", False: "Type I"})
    out_path = csv_path.with_name(f"{csv_path.stem}_joined.csv")
    joined.to_csv(out_path, index=False)
    return out_path


_write_joined_predictions = write_joined_predictions


def join_predictions(
    predictions_path: Path,
    qa_df: pd.DataFrame,
    *,
    joined_name: str | None = None,
) -> pd.DataFrame:
    """Inner-join a predictions CSV with qa_data slices; optionally write a CSV."""
    preds = pd.read_csv(predictions_path)
    qa = qa_df.sort_values(["report_id", "q_order"]).copy()
    qa["turn_index"] = qa.groupby("report_id").cumcount()
    qa["conv_type"] = qa["qa_split"].map({True: "Type II", False: "Type I"})
    joined = preds.merge(
        qa[["report_id", "turn_index", "q_order", "turn_type", "conv_type"]],
        on=["report_id", "turn_index"],
        how="left",
    )
    if joined_name is not None:
        out_path = predictions_path.with_name(joined_name)
        joined.to_csv(out_path, index=False)
    return joined


_join_predictions = join_predictions


def analyze_predictions(predictions_path: Path) -> pd.DataFrame:
    """Inner-join a predictions CSV to qa_data and print accuracy by slice."""
    preds = pd.read_csv(predictions_path)
    qa = qa_data.sort_values(["report_id", "q_order"]).copy()
    qa["turn_index"] = qa.groupby("report_id").cumcount()
    joined = preds.merge(
        qa[["report_id", "turn_index", "q_order", "turn_type", "qa_split"]],
        on=["report_id", "turn_index"],
        how="inner",
    )
    joined["conv_type"] = joined["qa_split"].map({True: "Type II", False: "Type I"})

    overall = joined["correct"].mean()
    print(f"\nAccuracy breakdowns (n={len(joined)} turns, overall={overall:.1%})")  # noqa: T201
    for col in ("turn_type", "conv_type", "q_order"):
        cut = joined.groupby(col)["correct"].agg(["mean", "count"])
        cut["mean"] = cut["mean"].map(lambda v: f"{v:.1%}")
        print(f"\nBy {col}:")  # noqa: T201
        print(cut.to_string())  # noqa: T201

    for gold_col, pred_col in (
        ("turn_type", "pred_turn_type"),
        ("conv_type", "pred_conv_type"),
    ):
        if pred_col not in joined.columns:
            continue
        cut = joined.groupby([gold_col, pred_col])["correct"].agg(["mean", "count"])
        cut["mean"] = cut["mean"].map(lambda v: f"{v:.1%}")
        print(f"\nBy {gold_col} × {pred_col}:")  # noqa: T201
        print(cut.to_string())  # noqa: T201

    out_path = predictions_path.with_name(f"{predictions_path.stem}_joined.csv")
    joined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")  # noqa: T201
    return joined


def compare_prediction_runs(
    left_csv: Path,
    right_csv: Path,
    *,
    left_label: str,
    right_label: str,
    output_name: str,
) -> Path:
    """Side-by-side parity report between two prediction CSV artifacts."""
    left = pd.read_csv(left_csv).rename(
        columns={
            "pred_answer": f"pred_{left_label}",
            "correct": f"correct_{left_label}",
        }
    )
    right = pd.read_csv(right_csv).rename(
        columns={
            "pred_answer": f"pred_{right_label}",
            "correct": f"correct_{right_label}",
        }
    )
    merged = left.merge(
        right,
        on=["report_id", "turn_index", "question", "gold_answer"],
        how="outer",
        indicator=True,
    )
    drift = (merged["_merge"] != "both").sum()
    if drift:
        raise RuntimeError(
            f"Test-set drift: {drift} rows are not in both runs. "
            f"{left_csv.name} and {right_csv.name} must evaluate the same records."
        )
    merged["agree"] = merged[f"correct_{left_label}"] == merged[f"correct_{right_label}"]

    qa = qa_data.sort_values(["report_id", "q_order"]).copy()
    qa["turn_index"] = qa.groupby("report_id").cumcount()
    qa["conv_type"] = qa["qa_split"].map({True: "Type II", False: "Type I"})
    merged = merged.merge(
        qa[["report_id", "turn_index", "q_order", "turn_type", "conv_type"]],
        on=["report_id", "turn_index"],
        how="left",
    )

    out_path = right_csv.with_name(output_name)
    merged.to_csv(out_path, index=False)

    print(f"\n=== Parity ({left_csv.name} ↔ {right_csv.name}) ===")  # noqa: T201
    print(f"n turns: {len(merged)}")  # noqa: T201
    print(f"{left_label} acc:  {merged[f'correct_{left_label}'].mean():.1%}")  # noqa: T201
    print(f"{right_label} acc: {merged[f'correct_{right_label}'].mean():.1%}")  # noqa: T201
    print(  # noqa: T201
        "delta:    "
        f"{(merged[f'correct_{right_label}'].mean() - merged[f'correct_{left_label}'].mean()) * 100:+.1f} pp"
    )
    print(f"agreement: {merged['agree'].mean():.1%}")  # noqa: T201

    for col in ("turn_type", "conv_type", "q_order"):
        cut = merged.groupby(col).agg(
            left_acc=(f"correct_{left_label}", "mean"),
            right_acc=(f"correct_{right_label}", "mean"),
            n=(f"correct_{right_label}", "size"),
        )
        cut["delta_pp"] = (cut["right_acc"] - cut["left_acc"]) * 100
        cut["left_acc"] = cut["left_acc"].map(lambda v: f"{v:.1%}")
        cut["right_acc"] = cut["right_acc"].map(lambda v: f"{v:.1%}")
        cut["delta_pp"] = cut["delta_pp"].map(lambda v: f"{v:+.1f}")
        print(f"\nBy {col}:")  # noqa: T201
        print(cut.to_string())  # noqa: T201

    for col, title in (
        ("turn_type", "Turn Type Accuracy by Model"),
        ("conv_type", "Conv Type Accuracy by Model"),
    ):
        rows: list[dict[str, Any]] = [
            {
                "bucket": "overall",
                f"{left_label}_acc": merged[f"correct_{left_label}"].mean(),
                f"{right_label}_acc": merged[f"correct_{right_label}"].mean(),
            }
        ]
        for bucket in sorted(merged[col].dropna().unique()):
            cut = merged[merged[col] == bucket]
            rows.append(
                {
                    "bucket": bucket,
                    f"{left_label}_acc": cut[f"correct_{left_label}"].mean(),
                    f"{right_label}_acc": cut[f"correct_{right_label}"].mean(),
                }
            )
        table = pd.DataFrame(rows)
        printable = table.copy()
        for acc_col in [f"{left_label}_acc", f"{right_label}_acc"]:
            printable[acc_col] = printable[acc_col].map(lambda v: f"{v:.1%}")
        print(f"\n{title}:")  # noqa: T201
        print(printable.to_string(index=False))  # noqa: T201

    print(f"\nWrote {out_path}")  # noqa: T201
    return out_path


def compare_runs(dspy_csv: Path, pyd_csv: Path) -> Path:
    """Side-by-side parity report against the DSPy run on the same artifact."""
    return compare_prediction_runs(
        dspy_csv,
        pyd_csv,
        left_label="dspy",
        right_label="pyd",
        output_name="parity_report.csv",
    )
