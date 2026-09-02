"""Teacher-vs-human agreement (M2 trust): the labelling sheet and Cohen's κ.

The teacher's first-fault attributions drive target selection, so they must be
checked against a human's judgment before the loop is trusted at scale. The
plan's bar: κ ≥ 0.7 over ~30 hand-labelled cases.

Two halves: `make_sheet` samples diagnosed cases into a CSV with the teacher's
verdict hidden in its own column and empty ``human_agent`` / ``human_mode`` /
``notes`` columns for the reviewer; `cohens_kappa` scores the filled sheet.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.evalloop.teacher import AGENTS, DIAGNOSTICS_DIR


def make_sheet(
    diagnoses_paths: list[Path | str],
    *,
    out_path: Path | str | None = None,
    n: int = 30,
    seed: int = 2026,
) -> Path:
    """Sample diagnosed cases into a labelling CSV for a human reviewer."""
    cases: list[dict[str, Any]] = []
    for path in diagnoses_paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                cases.append(json.loads(line))
    if not cases:
        raise SystemExit("no diagnoses found in the given files")
    random.Random(seed).shuffle(cases)
    picked = cases[:n]
    rows = [
        {
            "report_id": c["report_id"],
            "turn_index": c["turn_index"],
            "version": c.get("version", ""),
            "teacher_agent": c["failed_agent"],
            "teacher_mode": c["failure_mode"],
            "teacher_why": c["what_went_wrong"],
            "human_agent": "",  # ← reviewer fills: triage|preprocess|retriever|calculator|gold
            "human_mode": "",
            "notes": "",
        }
        for c in picked
    ]
    out = Path(out_path or DIAGNOSTICS_DIR / f"labelling_sheet_{len(rows)}cases.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def cohens_kappa(labels: list[str], preds: list[str]) -> float:
    """Plain two-rater Cohen's κ over categorical labels."""
    assert len(labels) == len(preds) and labels
    n = len(labels)
    cats = sorted(set(labels) | set(preds))
    po = sum(1 for a, b in zip(labels, preds, strict=True) if a == b) / n
    pe = sum((labels.count(c) / n) * (preds.count(c) / n) for c in cats)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def score_sheet(sheet_path: Path | str) -> dict[str, Any]:
    """κ between the human's agent labels and the teacher's, from a filled sheet."""
    df = pd.read_csv(sheet_path).fillna("")
    filled = df[df["human_agent"].str.strip() != ""]
    if filled.empty:
        raise SystemExit(
            f"{sheet_path}: no rows have human_agent filled in — label first"
        )
    labels = [str(v).strip().lower() for v in filled["human_agent"]]
    preds = [str(v).strip().lower() for v in filled["teacher_agent"]]
    unknown = sorted(set(labels) - set(AGENTS) - {"gold"})
    kappa = cohens_kappa(labels, preds)
    return {
        "n_labelled": len(filled),
        "n_total": len(df),
        "agreement": round(
            sum(a == b for a, b in zip(labels, preds, strict=True)) / len(labels), 4
        ),
        "kappa": round(kappa, 4),
        "meets_bar": bool(kappa >= 0.7),
        "unknown_labels": unknown,
        "disagreements": [
            {
                "report_id": r.report_id,
                "turn_index": int(r.turn_index),
                "teacher": r.teacher_agent,
                "human": r.human_agent,
            }
            for r in filled.itertuples()
            if str(r.human_agent).strip().lower()
            != str(r.teacher_agent).strip().lower()
        ],
    }
