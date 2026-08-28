"""Dataset loading helpers for ConvFinQA."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd

from convfinqa.config import DATA_DIR
from convfinqa.data.schemas import ConvExample, Document

# Anchored to the repo root, not the cwd: uvicorn, pytest, the CLIs and the
# container all start from different directories and must find the same file.
DATASET_PATH = DATA_DIR / "convfinqa_dataset.json"


def load_raw_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    """Load the raw ConvFinQA dataset JSON."""
    with path.open() as f:
        return cast(dict[str, Any], json.load(f))


def training_data() -> pd.DataFrame:
    """Return the per-turn ConvFinQA dataframe used by evaluation."""
    data = load_raw_dataset()

    dfs: dict[str, pd.DataFrame] = {}
    features: dict[str, pd.DataFrame] = {}
    for key, records in data.items():
        dfs[key] = pd.concat(
            [
                pd.DataFrame(
                    {
                        **x["dialogue"],
                        "data_key": key,
                        "report_id": x["id"],
                        "q_order": range(len(x["dialogue"]["conv_questions"])),
                    }
                )
                for x in records
            ],
            ignore_index=True,
        )
        features[key] = pd.DataFrame(
            [{**x["features"], "report_id": x["id"], "data_key": key} for x in records]
        )

    features_df = pd.concat(features.values(), ignore_index=True)
    features_df["has_type2_question"].apply(
        lambda x: "Simple" if x is False else "Complex"
    )
    assert features_df["report_id"].nunique() == features_df.shape[0], (
        "id should be unique for each report"
    )

    question_df = pd.concat(dfs.values(), ignore_index=True)
    question_df["base"] = "Overall"
    question_df["turn_type"] = pd.to_numeric(
        question_df["turn_program"], errors="coerce"
    ).apply(lambda x: "Number" if pd.notnull(x) else "Program")
    question_df["turn_program_actions"] = question_df["turn_program"].str.split(
        r"(?<=\)),"
    )
    question_df["turn_program_actions_n"] = question_df["turn_program_actions"].apply(
        len
    )
    question_df["turn_program_calcs"] = question_df["turn_program_actions"].apply(
        lambda x: [m.group(1) if (m := re.match(r"\s*(\w+)\(", s)) else None for s in x]
    )
    question_df["question_id"] = (
        question_df["report_id"] + "_q" + question_df["q_order"].astype(str)
    )
    question_df = question_df.merge(
        features_df, on=["report_id", "data_key"], how="left"
    )

    assert question_df["question_id"].value_counts().max() == 1, (
        "question_id should be unique for each question"
    )
    assert question_df.isnull().sum(axis=0).sum() == 0, (
        "There should be no missing values after merge"
    )
    return question_df


def _sample_qa_data() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    qa = training_data().query('data_key=="train"')
    sampled_report_ids = (
        qa["report_id"].drop_duplicates().sample(n=200, random_state=42).tolist()
    )
    additional_test_ids = (
        qa.loc[~qa["report_id"].isin(sampled_report_ids), "report_id"]
        .drop_duplicates()
        .sample(n=60, random_state=42)
        .tolist()
    )
    all_report_ids = sampled_report_ids + additional_test_ids
    train_report_ids = (
        pd.Series(sampled_report_ids).sample(frac=0.6, random_state=42).tolist()
    )
    test_report_ids = [
        r for r in sampled_report_ids if r not in train_report_ids
    ] + additional_test_ids
    qa = qa[qa["report_id"].isin(all_report_ids)].reset_index(drop=True)
    return qa, sampled_report_ids, train_report_ids, test_report_ids


qa_data, sampled_report_ids, train_report_ids, test_report_ids = _sample_qa_data()

_RAW_DATA = load_raw_dataset()
_DOCS: dict[str, Document] = {
    rec["id"]: Document.model_validate(rec["doc"])
    for split_records in _RAW_DATA.values()
    for rec in split_records
}


def _build_conv_examples(report_ids: list[str], qa: pd.DataFrame) -> list[ConvExample]:
    examples: list[ConvExample] = []
    for rid in report_ids:
        group = qa[qa["report_id"] == rid].sort_values("q_order")
        examples.append(
            ConvExample(
                report_id=rid,
                questions=group["conv_questions"].tolist(),
                gold_answers=group["conv_answers"].tolist(),
                gold_programs=group["turn_program"].fillna("").tolist(),
                gold_turn_types=group["turn_type"].tolist(),
                gold_conv_types=group["qa_split"]
                .map({True: "Type II", False: "Type I"})
                .tolist(),
            )
        )
    return examples


def load_conv_examples_test() -> tuple[list[ConvExample], pd.DataFrame]:
    """Return the canonical cached-evaluation conversation sample."""
    return _build_conv_examples(sampled_report_ids, qa_data), qa_data
