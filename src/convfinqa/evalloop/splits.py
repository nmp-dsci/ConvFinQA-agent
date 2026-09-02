"""Deterministic grouped splits for the eval loop, materialised as a manifest.

The committed manifest — not the seed — is the source of truth: it holds the
actual ``report_id`` lists, so recreating a split never depends on re-running
this code. The seed is provenance. (The old machinery is the cautionary tale:
two 60/40 splits both seeded 42 agree on only 78 of 120 conversations, because
the code around the seed changed.)

Allocation is by conversation, never by question — a question's siblings share
the report, the history and usually the error — and deterministically
stratified on ``has_type2_question``: each stratum is shuffled once with the
manifest seed and dealt round-robin across the three splits, against a
per-split budget proportional to the stratum's share of the pool — so every
split carries the pool's own Type II mix. The pool is the 2,777 train
conversations neither GEPA nor s7 ever touched.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from convfinqa.config import EVAL_ROOT

SPLITS_DIR = EVAL_ROOT / "splits"
DEFAULT_MANIFEST_PATH = SPLITS_DIR / "eval_loop_v1.json"
SPLIT_NAMES = ("train", "test", "holdout")


def _excluded_report_ids() -> set[str]:
    """The 260 conversations the old optimisers saw — never in any new split.

    ``loader.qa_data`` is already filtered to exactly that set: the 200 sampled
    conversations plus the 60 additional test ones.
    """
    from convfinqa.data.loader import qa_data

    return set(qa_data["report_id"])


def build_manifest(
    *, target_questions: int = 200, seed: int = 2026, name: str = "eval_loop_v1"
) -> dict[str, Any]:
    """Allocate the pool into train/test/holdout; return the manifest dict."""
    from convfinqa.data.loader import training_data
    from convfinqa.tracking.bundle import dataset_hash

    qa = training_data()
    excluded = _excluded_report_ids()
    per_report = (
        qa.groupby("report_id")
        .agg(
            n_questions=("question_id", "size"),
            type2=("has_type2_question", "first"),
        )
        .reset_index()
    )
    per_report = per_report[~per_report["report_id"].isin(excluded)]

    rng = random.Random(seed)
    splits: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
    n_by_report = dict(
        zip(per_report["report_id"], per_report["n_questions"], strict=True)
    )

    # Every split carries the pool's own stratum mix: each stratum gets a
    # per-split question budget proportional to its share of the pool, and is
    # dealt round-robin until each split's budget for it is met. Without the
    # budgets, whichever stratum is dealt first fills every split alone.
    total_q = int(per_report["n_questions"].sum())
    type2_q = int(per_report.loc[per_report["type2"], "n_questions"].sum())
    share2 = type2_q / total_q
    budgets = {True: target_questions * share2, False: target_questions * (1 - share2)}
    stratum_counts: dict[str, dict[bool, int]] = {
        s: {True: 0, False: 0} for s in SPLIT_NAMES
    }

    for stratum in (True, False):
        ids = sorted(per_report.loc[per_report["type2"] == bool(stratum), "report_id"])
        rng.shuffle(ids)
        deal = 0
        for rid in ids:
            open_splits = [
                s for s in SPLIT_NAMES if stratum_counts[s][stratum] < budgets[stratum]
            ]
            if not open_splits:
                break
            chosen = open_splits[deal % len(open_splits)]
            splits[chosen].append(rid)
            stratum_counts[chosen][stratum] += int(n_by_report[rid])
            deal += 1

    counts = {
        s: stratum_counts[s][True] + stratum_counts[s][False] for s in SPLIT_NAMES
    }

    all_ids = [rid for ids in splits.values() for rid in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split allocation produced overlapping report ids")
    if set(all_ids) & excluded:
        raise ValueError("split allocation leaked an excluded report id")
    short = [s for s in SPLIT_NAMES if counts[s] < target_questions]
    if short:
        raise ValueError(f"splits below the question target: {short}")

    type2_ids = set(per_report.loc[per_report["type2"], "report_id"])
    stats = {
        s: {
            "n_reports": len(ids),
            "n_questions": counts[s],
            "type2_share": round(sum(1 for r in ids if r in type2_ids) / len(ids), 4),
        }
        for s, ids in splits.items()
    }

    return {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset_hash": dataset_hash(),
        "excluded": {
            "n_report_ids": len(excluded),
            "reason": "seen by GEPA (optimizer_split) and s7 (v2 failures)",
        },
        "stratify": ["has_type2_question"],
        "target_questions": target_questions,
        "stats": stats,
        "splits": splits,
        "opened": [],
    }


def write_manifest(
    manifest: dict[str, Any], path: Path | None = None, *, force: bool = False
) -> Path:
    """Write the manifest; refuse to overwrite an existing one unless forced."""
    path = path or DEFAULT_MANIFEST_PATH
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists — the committed manifest is the source of "
            "truth for every run scored against it. Pass --force only to "
            "rebuild a manifest nothing has used yet."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the committed manifest."""
    path = path or DEFAULT_MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No split manifest at {path}. Run `convfinqa-evalloop make-splits`."
        )
    manifest: dict[str, Any] = json.loads(path.read_text())
    return manifest


def split_report_ids(
    split: str,
    *,
    n_reports: int | None = None,
    n_questions: int | None = None,
    path: Path | None = None,
) -> list[str]:
    """The report ids of one split, in manifest order, optionally truncated.

    `n_reports` truncates by conversation count. `n_questions` truncates by
    cumulative question count instead — a per-run budget, not a resize of the
    committed manifest — by walking manifest order and stopping once the
    budget is met. Pass at most one; both leave the manifest's own train/test/
    holdout pools (and any evidence already recorded against them) untouched.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}")
    if n_reports and n_questions:
        raise ValueError("pass at most one of n_reports, n_questions")
    ids = list(load_manifest(path)["splits"][split])
    if n_reports:
        return ids[:n_reports]
    if n_questions:
        from convfinqa.data.loader import training_data

        counts = training_data().groupby("report_id")["question_id"].size().to_dict()
        out: list[str] = []
        total = 0
        for report_id in ids:
            if total >= n_questions:
                break
            out.append(report_id)
            total += counts.get(report_id, 0)
        return out
    return ids
