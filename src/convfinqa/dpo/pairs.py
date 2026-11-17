"""Grade a group of tries at one question and pick its pairs (s33 §04).

A try is **right** when `numeric_match` accepts it *and* its sign agrees with
gold: rule 1 of `numeric_match` rounds both sides to an integer, so 0.5 matches
−0.5, and a wrong-sign "chosen" would teach exactly the wrong thing. The eval
metric is untouched; the sign check lives only here.

A try that errored with no answer is **dropped**, never rejected. A try that
needed a harness retry (H1–H3) or a schema correction is graded but not
**clean**, and round 1 pairs only clean tries: SFT collapses those detours, so
a chosen that kept one would differ from what the model actually wrote.

Pairs come from disjoint tries, at most `MAX_PAIRS` per question, rejected
tries with *different* wrong answers first, so two pairs cover two mistakes
rather than one twice.
"""

from __future__ import annotations

import math
import random
from typing import Any

MAX_PAIRS = 2


def sign_ok(pred: Any, gold: Any) -> bool:
    """False only when both sides are non-zero numbers of opposite sign."""
    from convfinqa.evaluation.metrics import _to_float

    try:
        pv, _ = _to_float(str(pred))
        gv, _ = _to_float(str(gold))
    except (ValueError, TypeError):
        return True
    if not (math.isfinite(pv) and math.isfinite(gv)) or pv == 0 or gv == 0:
        return True
    return (pv < 0) == (gv < 0)


def grade(answer: str, gold: Any) -> dict[str, bool]:
    """Numeric match, sign agreement, and both together (`right`)."""
    from convfinqa.evaluation.metrics import numeric_match

    numeric_ok = bool(numeric_match(answer, gold))
    signed = sign_ok(answer, gold)
    return {"numeric_ok": numeric_ok, "sign_ok": signed, "right": numeric_ok and signed}


def status(t: dict[str, Any]) -> str:
    """`dropped` (no answer), `right` or `wrong`."""
    if t.get("error"):
        return "dropped"
    return "right" if t.get("right") else "wrong"


def clean(t: dict[str, Any]) -> bool:
    """Answered on the first attempt with no harness retry."""
    return int(t.get("attempts") or 1) == 1 and not t.get("harness_retry")


def _norm(answer: Any) -> str:
    return str(answer).strip().lower()


def select(tries: list[dict[str, Any]], rng: random.Random) -> list[tuple[int, int]]:
    """Up to `MAX_PAIRS` (chosen, rejected) index pairs from disjoint clean tries."""
    rights = [t["i"] for t in tries if status(t) == "right" and clean(t)]
    wrongs = [t for t in tries if status(t) == "wrong" and clean(t)]
    if not rights or not wrongs:
        return []
    rng.shuffle(rights)
    by_answer: dict[str, list[int]] = {}
    for t in wrongs:
        by_answer.setdefault(_norm(t.get("answer")), []).append(t["i"])
    groups = list(by_answer.values())
    rng.shuffle(groups)
    for g in groups:
        rng.shuffle(g)
    rejected: list[int] = []
    while any(groups):  # round-robin: one of each distinct wrong answer first
        for g in groups:
            if g:
                rejected.append(g.pop())
    return list(zip(rights, rejected, strict=False))[:MAX_PAIRS]


def keep(
    tries: list[dict[str, Any]], pairs: list[tuple[int, int]], rng: random.Random
) -> int | None:
    """The right try the conversation continues on, or None to stop.

    Preference: clean and unpaired, then clean, then any right try (a harness
    retry is still the student's own history, as it would be at serving).
    """
    used = {i for pair in pairs for i in pair}
    rights = [t for t in tries if status(t) == "right"]
    for tier in (
        [t for t in rights if clean(t) and t["i"] not in used],
        [t for t in rights if clean(t)],
        rights,
    ):
        if tier:
            return int(rng.choice(tier)["i"])
    return None


def question_seed(question_id: str, salt: int = 0) -> int:
    """A stable per-question seed, so pairing and the kept try reproduce."""
    import hashlib

    digest = hashlib.sha256(f"{question_id}|{salt}".encode()).hexdigest()
    return int(digest[:12], 16)


# --- s34: pairs over every right × wrong combination, capped -----------------


def close(pred: Any, gold: Any, rel: float = 0.01) -> bool:
    """A stricter "right" for a chosen: within `rel` of gold, at some percent scale,
    or equal to gold at the precision gold is written to.

    `numeric_match` rounds both sides to an integer, so 0.0931 matches 0.0852;
    this is the labeller's check only, never the eval metric. Non-numbers pass.
    """
    from convfinqa.evaluation.metrics import _to_float

    try:
        pv, _ = _to_float(str(pred))
        gv, _ = _to_float(str(gold))
    except (ValueError, TypeError):
        return True
    if not (math.isfinite(pv) and math.isfinite(gv)):
        return True
    if gv == 0:
        return abs(pv) < 1e-9
    digits = str(gold).strip().rstrip("%").split(".")
    places = len(digits[1]) if len(digits) == 2 and digits[1].isdigit() else 0
    for scale in (1.0, 100.0, 0.01):
        p = pv * scale
        if abs(p - gv) <= rel * abs(gv) or round(p, places) == round(gv, places):
            return True
    return False


def select_capped(
    tries: list[dict[str, Any]],
    rng: random.Random,
    cap: int,
    *,
    gold: Any = None,
    strict: bool = False,
) -> list[tuple[int, int]]:
    """Up to `cap` (chosen, rejected) pairs from every clean right × clean wrong.

    Greedy, so the pairs spread before they repeat: each next pair takes a wrong
    answer used least so far, then the least-used rejected try, then the
    least-used chosen try (random among ties). Distinct mistakes come first, as
    in `select`, and no one try carries a question's weight. With `strict`, a
    chosen must also be `close` to gold.
    """
    rights = [
        t["i"]
        for t in tries
        if status(t) == "right"
        and clean(t)
        and (not strict or close(t.get("answer"), gold))
    ]
    wrongs = [t for t in tries if status(t) == "wrong" and clean(t)]
    if not rights or not wrongs or cap <= 0:
        return []
    answer_of = {t["i"]: _norm(t.get("answer")) for t in wrongs}
    combos = [(c, w["i"]) for c in rights for w in wrongs]
    rng.shuffle(combos)
    used_c: dict[int, int] = {}
    used_r: dict[int, int] = {}
    used_a: dict[str, int] = {}
    out: list[tuple[int, int]] = []
    while combos and len(out) < cap:
        best = min(
            combos,
            key=lambda p: (
                used_a.get(answer_of[p[1]], 0),
                used_r.get(p[1], 0),
                used_c.get(p[0], 0),
            ),
        )
        combos.remove(best)
        out.append(best)
        used_c[best[0]] = used_c.get(best[0], 0) + 1
        used_r[best[1]] = used_r.get(best[1], 0) + 1
        a = answer_of[best[1]]
        used_a[a] = used_a.get(a, 0) + 1
    return out


def selection(
    records: dict[str, dict[str, Any]],
    split_of: dict[str, str],
    cap: int = 4,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Every capped pair over a tries file, as indices into it (s34's chosen cap).

    One row per pair: which question, which try is chosen and which rejected,
    both answers and gold, the turn kind and the manifest split. Rendering the
    pair's text is `sft.rows`' job; this is the record of which tries were paired.
    """
    rows: list[dict[str, Any]] = []
    for rid, rec in records.items():
        for q in rec["questions"]:
            tries = q["tries"]
            by_i = {t["i"]: t for t in tries}
            got = select_capped(
                tries,
                random.Random(question_seed(q["question_id"])),
                cap,
                gold=q["gold"],
                strict=strict,
            )
            for c, r in got:
                rows.append(
                    {
                        "report_id": rid,
                        "question_id": q["question_id"],
                        "k": q["k"],
                        "split": split_of[rid],
                        "kind": "program"
                        if str(q.get("gold_turn_type")).lower() == "program"
                        else "number",
                        "gold": q["gold"],
                        "chosen_i": c,
                        "rejected_i": r,
                        "chosen_answer": by_i[c].get("answer"),
                        "rejected_answer": by_i[r].get("answer"),
                        "cap": cap,
                        "strict": strict,
                    }
                )
    return rows
