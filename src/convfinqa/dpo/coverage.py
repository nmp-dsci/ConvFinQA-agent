"""Coverage of the sampled pool (s34): every conversation and question, by turn
type, why a question gives no pair, and what each pair cap yields.

Reads a tries file and the manifest only, so the s34 page rebuilds from the
committed record:

    uv run python -m convfinqa.dpo.coverage evaluation/dpo/v1/tries8.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from convfinqa.dpo import pairs
from convfinqa.dpo.sample import load_records, questions_for

CAPS = (1, 2, 3, 4, 6, 8, 64)


def _kind(turn_type: Any) -> str:
    return "program" if str(turn_type).lower() == "program" else "number"


def _why_none(tries: list[dict[str, Any]]) -> str:
    """Why a reached question gives no pair."""
    graded = [t for t in tries if pairs.status(t) != "dropped"]
    rights = [t for t in graded if pairs.status(t) == "right"]
    wrongs = [t for t in graded if pairs.status(t) == "wrong"]
    if not graded:
        return "every try dropped"
    if not wrongs:
        return "all right"
    if not rights:
        return "all wrong"
    return "flip, but no clean right or no clean wrong"


def summarise(
    records: dict[str, dict[str, Any]], cap: int, strict: bool
) -> dict[str, Any]:
    """The coverage tables, with pairs counted under `cap` (and `strict`)."""
    from convfinqa.dpo import manifest

    m = manifest.load()
    split_of = {r: s for s, ids in m["splits"].items() for r in ids}
    gold_qs = questions_for(m["order"])

    q_by: dict[str, Counter[str]] = {"program": Counter(), "number": Counter()}
    why: dict[str, Counter[str]] = {"program": Counter(), "number": Counter()}
    by_pos: dict[int, Counter[str]] = {}
    conv_pairs: Counter[int] = Counter()
    conv: Counter[str] = Counter()
    stop_at: Counter[str] = Counter()
    rights_hist: Counter[str] = Counter()
    per_cap: dict[str, Counter[str]] = {
        f"{'strict' if s else 'plain'}:{c}": Counter()
        for s in (False, True)
        for c in CAPS
    }
    per_q_pairs: list[int] = []

    for rid in m["order"]:
        qs = gold_qs[rid]
        rec = records.get(rid)
        split = split_of[rid]
        conv["pool"] += 1
        if rec is None:
            conv["not sampled"] += 1
            for gq in qs:
                q_by[_kind(gq["turn_type"])]["not sampled"] += 1
            continue
        conv["sampled"] += 1
        conv[f"sampled:{split}"] += 1
        stop_at[
            "completed" if rec["stop"] == "completed" else rec["stop"].split(" at ")[1]
        ] += 1
        n_pairs_conv = 0
        reached = {q["k"]: q for q in rec["questions"]}
        for k, gq in enumerate(qs):
            kind = _kind(gq["turn_type"])
            c = q_by[kind]
            c["questions"] += 1
            q = reached.get(k)
            if q is None:
                c["not reached"] += 1
                why[kind]["not reached (an earlier question had no right try)"] += 1
                continue
            c["reached"] += 1
            tries = q["tries"]
            graded = [t for t in tries if pairs.status(t) != "dropped"]
            n_right = sum(pairs.status(t) == "right" for t in graded)
            rights_hist[f"{n_right}/{len(graded)}"] += 1
            bp = by_pos.setdefault(k, Counter())
            bp["reached"] += 1
            bp["right"] += n_right
            bp["graded"] += len(graded)
            for s in (False, True):
                for cap_ in CAPS:
                    got = pairs.select_capped(
                        tries,
                        random.Random(pairs.question_seed(q["question_id"])),
                        cap_,
                        gold=q["gold"],
                        strict=s,
                    )
                    key = f"{'strict' if s else 'plain'}:{cap_}"
                    per_cap[key][f"pairs:{kind}"] += len(got)
                    per_cap[key][f"pairs:{split}"] += len(got)
                    per_cap[key]["pairs"] += len(got)
                    per_cap[key]["questions"] += bool(got)
            got = pairs.select_capped(
                tries,
                random.Random(pairs.question_seed(q["question_id"])),
                cap,
                gold=q["gold"],
                strict=strict,
            )
            per_q_pairs.append(len(got))
            n_pairs_conv += len(got)
            bp["pairs"] += len(got)
            bp["paired"] += bool(got)
            if got:
                c["paired"] += 1
                c["pairs"] += len(got)
            else:
                c["no pair"] += 1
                why[kind][_why_none(tries)] += 1
            c["right"] += n_right
            c["graded"] += len(graded)
        conv_pairs[min(n_pairs_conv, 10)] += 1
        conv["with a pair"] += bool(n_pairs_conv)

    nonzero = sorted((p for p in per_q_pairs if p), reverse=True)
    top_decile = sum(nonzero[: max(1, len(nonzero) // 10)])
    return {
        "cap": cap,
        "strict": strict,
        "conversations": dict(conv),
        "stops": dict(stop_at),
        "questions": {k: dict(v) for k, v in q_by.items()},
        "no_pair_reasons": {k: dict(v.most_common()) for k, v in why.items()},
        "by_position": {str(k): dict(v) for k, v in sorted(by_pos.items())},
        "pairs_per_conversation": {str(k): v for k, v in sorted(conv_pairs.items())},
        "rights_histogram": dict(
            sorted(
                rights_hist.items(), key=lambda kv: tuple(map(int, kv[0].split("/")))
            )
        ),
        "caps": {k: dict(v) for k, v in per_cap.items()},
        "top_decile_share": round(top_decile / max(1, sum(nonzero)), 3),
    }


def main(argv: list[str] | None = None) -> int:
    """Print (and write beside the tries) the coverage at cap 4, strict."""
    args = argv if argv is not None else sys.argv[1:]
    path = Path(args[0])
    cap = int(args[1]) if len(args) > 1 else 4
    out = summarise(load_records(path), cap=cap, strict=True)
    dest = path.parent / f"coverage_{path.stem}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    sys.stdout.write(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
