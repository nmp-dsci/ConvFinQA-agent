"""What the sampled tries add up to: yield, flips, pairs, cost (s33 P3's readout).

Reads `tries.jsonl` and `sample_runs.jsonl` and nothing else, so the numbers on
the findings page rebuild from the committed record.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from convfinqa.dpo import pairs
from convfinqa.dpo.sample import RUNS_PATH, TRIES_PATH

# Modal's L4 list price, per second (sft/cloud/modal_app.py's GPU_USD_PER_S).
L4_USD_PER_S = 0.000222


def load(path: Path = TRIES_PATH) -> list[dict[str, Any]]:
    """Every line of a JSONL record."""
    from convfinqa.dpo.sample import open_text

    with open_text(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _kind(turn_type: Any) -> str:
    return "program" if str(turn_type).lower() == "program" else "number"


def summarise(
    records: list[dict[str, Any]], runs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Yield, flips, pairs, retries and cost over the sampled conversations."""
    by: dict[str, Counter[str]] = {
        "all": Counter(),
        "program": Counter(),
        "number": Counter(),
    }
    rights_hist: Counter[str] = Counter()
    stops: Counter[str] = Counter()
    harness: Counter[str] = Counter()
    by_k: dict[int, Counter[str]] = {}
    tokens: Counter[str] = Counter()
    wrong_answers_per_flip: list[int] = []
    sign_saved = 0
    n_questions_total = 0
    for rec in records:
        n_questions_total += int(rec["n_questions"])
        stops[
            "completed" if rec["stop"] == "completed" else rec["stop"].split(" at ")[0]
        ] += 1
        for q in rec["questions"]:
            kind = _kind(q.get("gold_turn_type"))
            tries = q["tries"]
            graded = [t for t in tries if pairs.status(t) != "dropped"]
            rights = [t for t in graded if pairs.status(t) == "right"]
            wrongs = [t for t in graded if pairs.status(t) == "wrong"]
            cl = [t for t in graded if pairs.clean(t)]
            for key in ("all", kind):
                c = by[key]
                c["questions"] += 1
                c["tries"] += len(tries)
                c["graded"] += len(graded)
                c["right"] += len(rights)
                c["dropped"] += len(tries) - len(graded)
                c["unclean"] += len(graded) - len(cl)
                c["pairs"] += len(q["pairs"])
                c["flip"] += int(bool(rights) and bool(wrongs))
                c["all_right"] += int(bool(graded) and not wrongs)
                c["all_wrong"] += int(bool(graded) and not rights)
                c["paired_q"] += int(bool(q["pairs"]))
            k = int(q["k"])
            bk = by_k.setdefault(k, Counter())
            bk["questions"] += 1
            bk["right"] += len(rights)
            bk["graded"] += len(graded)
            bk["flip"] += int(bool(rights) and bool(wrongs))
            rights_hist[f"{len(rights)}/{len(graded)}"] += 1
            if rights and wrongs:
                wrong_answers_per_flip.append(
                    len({str(t.get("answer")).strip() for t in wrongs})
                )
            for t in graded:
                if t.get("numeric_ok") and not t.get("sign_ok"):
                    sign_saved += 1
                if t.get("harness_retry"):
                    harness[str(t["harness_retry"])] += 1
                if int(t.get("attempts") or 1) > 1:
                    harness["schema_correction"] += 1
                for key, v in (t.get("usage") or {}).items():
                    if isinstance(v, (int, float)) and v is not None:
                        tokens[key] += int(v)
            for t in tries:
                if t.get("error"):
                    harness["dropped:" + str(t["error"]).split(":")[0][:40]] += 1

    def rates(c: Counter[str]) -> dict[str, Any]:
        q = max(1, c["questions"])
        return {
            **dict(c),
            "pass1": round(c["right"] / max(1, c["graded"]), 4),
            "flip_share": round(c["flip"] / q, 4),
            "all_right_share": round(c["all_right"] / q, 4),
            "all_wrong_share": round(c["all_wrong"] / q, 4),
            "pairs_per_question": round(c["pairs"] / q, 3),
        }

    runs = runs or []
    wall = sum(float(r.get("wall_s") or 0) for r in runs)
    n_conv = len(records)
    reached = by["all"]["questions"] / max(1, n_questions_total)
    return {
        "conversations": n_conv,
        "questions_in_conversations": n_questions_total,
        "questions_reached": by["all"]["questions"],
        "reached_share": round(reached, 4),
        "stops": dict(stops),
        "overall": rates(by["all"]),
        "program": rates(by["program"]),
        "number": rates(by["number"]),
        "by_position": {
            str(k): {
                **dict(c),
                "pass1": round(c["right"] / max(1, c["graded"]), 4),
                "flip_share": round(c["flip"] / max(1, c["questions"]), 4),
            }
            for k, c in sorted(by_k.items())
        },
        "rights_histogram": dict(sorted(rights_hist.items())),
        "distinct_wrong_answers_per_flip": dict(Counter(wrong_answers_per_flip)),
        "sign_flips_caught": sign_saved,
        "retries_and_drops": dict(harness),
        "tokens": dict(tokens),
        "splits": dict(Counter(str(r.get("split")) for r in records)),
        "sampling_wall_s": round(wall, 1),
        "l4_usd_at_one_container": round(wall * L4_USD_PER_S, 2),
        "runs": runs,
    }


def write(out: Path | None = None, tries_path: Path = TRIES_PATH) -> dict[str, Any]:
    """Write `<stem>` stats beside a tries file and return them."""
    records = load(tries_path)
    runs_path = (
        RUNS_PATH
        if tries_path == TRIES_PATH
        else tries_path.parent / f"{tries_path.stem}_runs.jsonl"
    )
    runs = load(runs_path) if runs_path.exists() else []
    summary = summarise(records, runs)
    name = "stats.json" if tries_path == TRIES_PATH else f"stats_{tries_path.stem}.json"
    path = out or tries_path.parent / name
    path.write_text(json.dumps(summary, indent=1) + "\n")
    return summary
