"""Aim the SFT draw at the turns the student misses (s29, the owner's N2).

A random draw adds conversations in the pool's own mix. The aimed draw ranks
every candidate by how many of its turns the student is *expected* to miss,
read from gold alone, so nothing has to run first:

- each turn falls in a bucket by its gold program's last operation
  (``lookup`` for a number turn, else ``add``/``subtract``/``divide``/…);
- a bucket's miss rate is the student's error rate on it in a scored val pass,
  shrunk toward the pass's overall rate as if it held `SHRINK` more turns
  there, so a bucket val saw 15 times does not swing the ranking alone;
- a turn from question `LATE_FROM` + 1 onward has its rate multiplied by the
  pass's late-to-early miss ratio;
- a conversation's score is the sum over its turns, and the draw takes the
  top `n`, ties broken by report id.

The rates come from val only. Test 100 is the holdout: steering training data
by its failures would make the next test pass read high.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

SHRINK = 20
LATE_FROM = 4  # q_order 4 is the fifth question
_OPS = re.compile(r"(add|subtract|multiply|divide|exp|greater)\(")


def bucket(program: Any) -> str:
    """A gold turn's bucket: its program's last operation, or ``lookup``."""
    ops = _OPS.findall(str(program))
    return ops[-1] if ops else "lookup"


def gold_turns() -> Any:
    """Every pool question with its bucket and whether it comes late."""
    from convfinqa.data.loader import training_data

    qa = training_data()[["report_id", "q_order", "turn_program"]].copy()
    qa["bucket"] = qa["turn_program"].map(bucket)
    qa["late"] = qa["q_order"] >= LATE_FROM
    return qa


def miss_rates(val_csv: Path) -> dict[str, Any]:
    """The student's shrunk miss rate per bucket, and the late ratio, from a val pass."""
    import pandas as pd

    val = pd.read_csv(val_csv)
    if set(val["split"].dropna().astype(str)) != {"train"}:
        raise ValueError(f"{val_csv.name}: aim from a train-side (val) pass only")
    if "unscored" in val and val["unscored"].fillna(False).astype(bool).any():
        raise ValueError(f"{val_csv.name}: an incomplete pass cannot set the aim")
    val["correct"] = val["correct"].astype(bool)
    qa = gold_turns().set_index(["report_id", "q_order"])
    val = val.join(qa[["bucket", "late"]], on=["report_id", "turn_index"])
    if val["bucket"].isna().any():
        raise ValueError(f"{val_csv.name}: turns with no gold question in the pool")
    overall = float(val["correct"].mean())

    def miss(rows: Any) -> float:
        return 1.0 - (float(rows.sum()) + SHRINK * overall) / (len(rows) + SHRINK)

    by_bucket = {str(k): miss(g) for k, g in val.groupby("bucket")["correct"]}
    late = miss(val.loc[val["late"], "correct"]) / miss(
        val.loc[~val["late"], "correct"]
    )
    return {
        "source_csv": val_csv.name,
        "source_sha256": hashlib.sha256(val_csv.read_bytes()).hexdigest(),
        "overall_accuracy": round(overall, 4),
        "shrink": SHRINK,
        "late_from_q_order": LATE_FROM,
        "late_ratio": round(late, 4),
        "miss_by_bucket": {k: round(v, 4) for k, v in sorted(by_bucket.items())},
        # a bucket val never saw is missed at the overall rate
        "miss_unseen_bucket": round(1.0 - overall, 4),
    }


def scores(candidates: set[str], rates: dict[str, Any]) -> dict[str, float]:
    """Expected misses per candidate conversation under `rates`."""
    qa = gold_turns()
    qa = qa[qa["report_id"].isin(candidates)]
    miss = qa["bucket"].map(rates["miss_by_bucket"]).fillna(rates["miss_unseen_bucket"])
    miss = miss * qa["late"].map({True: rates["late_ratio"], False: 1.0})
    return {str(k): float(v) for k, v in miss.groupby(qa["report_id"]).sum().items()}


def top(candidates: set[str], rates: dict[str, Any], n: int) -> list[tuple[str, float]]:
    """The `n` highest-scoring candidates, highest first, ties by report id."""
    ranked = sorted(scores(candidates, rates).items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) < n:
        raise ValueError(f"only {len(ranked)} candidates for an aimed draw of {n}")
    return ranked[:n]
