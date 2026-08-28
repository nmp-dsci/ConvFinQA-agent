"""Eval runs, dataset splits, and the all-versions answers explorer.

Every route here reads committed files, so all of it stays fully live in the
demo. That is deliberate: the demo's honesty rests on the read-only half being
real, and these are the surfaces that show the held-out discipline and the
per-version results a visitor would otherwise have to take on trust.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from convfinqa.serving import evaldata
from convfinqa.serving.models import (
    AnswerRow,
    EvalSummary,
    ModelAccuracy,
    PredRow,
    SplitSummary,
    VersionAnswer,
)

router = APIRouter(prefix="/eval")

_SPLIT_DESCRIPTIONS = {
    "train": (
        "The 60% of sampled conversations the optimizer was allowed to see. GEPA "
        "and the s7 harness read these; no reported accuracy comes from them."
    ),
    "holdout": (
        "Held out from every optimizer. Every headline accuracy in this app is "
        "measured here, on questions no prompt was ever tuned against."
    ),
    "sampled": "The full 200-conversation sample the train/holdout split was drawn from.",
}


@router.get("/runs")
async def list_eval_runs() -> list[str]:
    """Prompt versions that have at least one committed joined CSV."""
    return evaldata.available_versions()


@router.get("/splits")
async def get_splits() -> list[SplitSummary]:
    """Dataset split membership, so the held-out claim is inspectable."""
    from convfinqa.data.loader import qa_data

    out: list[SplitSummary] = []
    for name in ("train", "holdout", "sampled"):
        report_ids = evaldata.splits()[name]
        n_questions = int(qa_data["report_id"].isin(report_ids).sum())
        out.append(
            SplitSummary(
                name=name,
                description=_SPLIT_DESCRIPTIONS[name],
                n_conversations=len(report_ids),
                n_questions=n_questions,
                report_ids=sorted(report_ids),
            )
        )
    return out


@router.get("/answers")
async def get_answers(
    report_id: str = "",
    limit: int = Query(default=200, ge=1, le=2000),
    only_disagreements: bool = False,
) -> list[AnswerRow]:
    """Every question with gold plus each version's answer, side by side.

    This is the surface where a reader can see *what changed* between versions
    rather than only that a percentage moved — including the turns v3_1 broke.
    """
    versions = evaldata.available_versions()
    if not versions:
        return []

    frames = {v: evaldata.load_joined(v) for v in versions}
    frames = {v: df for v, df in frames.items() if df is not None}
    if not frames:
        return []

    base = next(iter(frames.values()))
    if report_id:
        base = base[base["report_id"] == report_id]
    base = base.sort_values(["report_id", "turn_index"]).head(limit)

    golds = evaldata.gold_programs()
    rows: list[AnswerRow] = []
    for row in base.itertuples():
        key = (str(row.report_id), int(getattr(row, "q_order", row.turn_index)))
        answers: list[VersionAnswer] = []
        for version, df in frames.items():
            match = df[
                (df["report_id"] == row.report_id)
                & (df["turn_index"] == int(row.turn_index))
            ]
            if match.empty:
                continue
            entry = match.iloc[0]
            answers.append(
                VersionAnswer(
                    version=version,
                    pred_answer=str(entry.get("pred_answer", "")),
                    pred_program=str(entry.get("pred_program", "") or ""),
                    correct=bool(entry["correct"]),
                )
            )
        if only_disagreements and len({a.correct for a in answers}) <= 1:
            continue
        rows.append(
            AnswerRow(
                report_id=str(row.report_id),
                turn_index=int(row.turn_index),
                question=str(row.question),
                gold_answer=str(row.gold_answer),
                gold_program=golds.get(key, ""),
                gold_turn_type=str(getattr(row, "turn_type", "")),
                gold_conv_type=str(getattr(row, "conv_type", "")),
                versions=answers,
            )
        )
    return rows


@router.get("/runs/{run_name}/summary")
async def get_eval_summary(run_name: str) -> EvalSummary:
    """Overall and per-slice accuracy for one version, per backend."""
    available: dict[str, ModelAccuracy] = {}
    for model in evaldata.MODEL_CSV_PATTERN:
        df = evaldata.load_joined(run_name, model)
        if df is None:
            continue
        available[model] = ModelAccuracy.model_validate(
            {
                "overall": evaldata.slice_accuracy(df, "overall"),
                "by_turn_type": evaldata.slices_by(df, "turn_type"),
                "by_conv_type": evaldata.slices_by(df, "conv_type"),
                "by_q_order": evaldata.slices_by(df, "q_order"),
            }
        )
    if not available:
        raise HTTPException(
            status_code=404, detail=f"No predictions found for version {run_name}"
        )
    return EvalSummary(
        run_name=run_name, available_models=list(available), models=available
    )


@router.get("/runs/{run_name}/predictions")
async def get_eval_predictions(run_name: str, model: str = "pydantic") -> list[PredRow]:
    """Every scored row for one version."""
    if model not in evaldata.MODEL_CSV_PATTERN:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    df = evaldata.load_joined(run_name, model)
    if df is None:
        raise HTTPException(
            status_code=404, detail=f"No predictions for {run_name}/{model}"
        )
    golds = evaldata.gold_programs()
    rows: list[PredRow] = []
    for row in df.itertuples():
        q_order = int(getattr(row, "q_order", row.turn_index))
        rows.append(
            PredRow(
                report_id=str(row.report_id),
                turn_index=int(row.turn_index),
                question=str(row.question),
                gold_answer=str(row.gold_answer),
                gold_program=golds.get((str(row.report_id), q_order), ""),
                pred_answer=str(row.pred_answer),
                pred_program=str(getattr(row, "pred_program", "") or ""),
                correct=bool(row.correct),
                q_order=q_order,
                turn_type=str(getattr(row, "turn_type", "")),
                conv_type=str(getattr(row, "conv_type", "")),
            )
        )
    return rows
