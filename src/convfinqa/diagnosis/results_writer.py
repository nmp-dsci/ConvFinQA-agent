"""Write diagnostic_results_v3_opt.csv from CaseResults + input CSV (Group A)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.diagnosis.harness import case_results_to_rows
from convfinqa.diagnosis.models import CaseResult

# Output column order: Group A (preserved from predictions CSV) +
# Group B (router output, varies per attempt_id) + Group C (verify result).
GROUP_A_COLUMNS = [
    "report_id",
    "turn_index",
    "question",
    "gold_answer",
    "pred_answer",
    "correct",
    "pred_program",
    "gold_program",
    "pred_turn_type",
    "gold_turn_type",
    "pred_conv_type",
    "gold_conv_type",
    "pred_sub_questions",
    "history_text",
    "triage_io",
    "preprocess_io",
    "retriever_io",
    "calculator_io",
]
GROUP_B_COLUMNS = [
    "attempt_id",
    "failed_agent",
    "failure_mode",
    "failure_explanation",
    "supporting_evidence",
    "system_prompt_fix",
    "fix_type",
    "confidence",
]
GROUP_C_COLUMNS = [
    "harness_correct",
    "harness_first_failing_turn",
    "harness_turn_results",
    "harness_pred_answer",
    "harness_triage_io",
    "harness_preprocess_io",
    "harness_retriever_io",
    "harness_calculator_io",
    "verify_result",
    "failure_reason",
    "resolved",
]


def write_diagnostic_csv(
    results: list[CaseResult],
    full_df: pd.DataFrame,
    *,
    output_path: Path,
) -> Path:
    """Write the diagnostic_results CSV, joining Group A columns from full_df."""
    rows = case_results_to_rows(results)
    # Index input CSV by (report_id, turn_index) for fast lookup.
    full_df = full_df.copy()
    full_df["turn_index"] = full_df["turn_index"].astype(int)
    idx = full_df.set_index(["report_id", "turn_index"], drop=False)

    out_rows: list[dict[str, Any]] = []
    for r in rows:
        key = (r["report_id"], r["turn_index"])
        group_a: dict[str, Any] = {}
        if key in idx.index:
            src = idx.loc[key]
            if isinstance(src, pd.DataFrame):
                src = src.iloc[0]
            for col in GROUP_A_COLUMNS:
                group_a[col] = src.get(col, "")
        else:
            for col in GROUP_A_COLUMNS:
                group_a[col] = r.get(col, "")
        merged = {**group_a}
        for col in GROUP_B_COLUMNS:
            merged[col] = r.get(col, "")
        for col in GROUP_C_COLUMNS:
            merged[col] = r.get(col, "")
        out_rows.append(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = GROUP_A_COLUMNS + GROUP_B_COLUMNS + GROUP_C_COLUMNS
    pd.DataFrame(out_rows, columns=columns).to_csv(output_path, index=False)
    return output_path
