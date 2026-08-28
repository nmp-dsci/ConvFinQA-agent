"""Load first-wrong-per-conversation cases from a predictions CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.diagnosis.models import RouterPayload, StageIO
from convfinqa.prompts import load as load_prompts


def _parse_io(raw: Any) -> StageIO | None:
    """Tolerantly parse a JSON-encoded stage IO cell. Returns None on empty/invalid."""
    if raw is None:
        return None
    if isinstance(raw, float):
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return StageIO(
        input=data.get("input") or {},
        output=data.get("output") or {},
        trajectory=data.get("trajectory") or [],
    )


def load_first_wrong_cases(
    csv_path: Path,
    *,
    version: str = "v2",
    limit: int | None = None,
) -> tuple[list[RouterPayload], pd.DataFrame]:
    """Load first-wrong cases and return (RouterPayloads, full_df).

    Filters rows where correct == False and selects the minimum turn_index per
    report_id. Injects the four current prompts (default v2) into each payload.
    """
    full_df = pd.read_csv(csv_path).fillna("")
    full_df["correct_bool"] = (
        full_df["correct"].astype(str).str.lower().isin({"true", "1"})
    )
    wrong = full_df[~full_df["correct_bool"]]
    if wrong.empty:
        return [], full_df
    # min turn_index per report_id
    idx = wrong.groupby("report_id")["turn_index"].idxmin()
    first_wrong = wrong.loc[idx].sort_values(["report_id", "turn_index"])
    if limit is not None:
        first_wrong = first_wrong.head(limit)

    prompts = load_prompts(version)
    payloads: list[RouterPayload] = []
    for _, row in first_wrong.iterrows():
        payloads.append(
            RouterPayload(
                report_id=str(row["report_id"]),
                turn_index=int(row["turn_index"]),
                question=str(row["question"]),
                history_text=str(row["history_text"]),
                gold_answer=str(row["gold_answer"]),
                pred_answer=str(row["pred_answer"]),
                gold_program=str(row["gold_program"]),
                gold_turn_type=str(row["gold_turn_type"]).lower(),
                pred_turn_type=str(row["pred_turn_type"]).lower(),
                gold_conv_type=str(row["gold_conv_type"]),
                pred_conv_type=str(row["pred_conv_type"]),
                triage_io=_parse_io(row["triage_io"]),
                preprocess_io=_parse_io(row["preprocess_io"]),
                retriever_io=_parse_io(row["retriever_io"]),
                calculator_io=_parse_io(row["calculator_io"]),
                current_triage_prompt=prompts["triage"],
                current_preprocess_prompt=prompts["preprocess"],
                current_retriever_prompt=prompts["retriever"],
                current_calculator_prompt=prompts["calculator"],
            )
        )
    return payloads, full_df
