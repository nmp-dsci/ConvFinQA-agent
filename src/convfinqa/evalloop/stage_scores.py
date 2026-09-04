"""Per-agent scores derived from the dataset's own gold (M2.5 Phase B).

The gold *program* plus gold *answer* determine what three of the four stages
should have produced, so most of the per-agent panel needs no judge:

- **triage** — direct: ``gold_turn_type`` / ``gold_conv_type`` are columns.
- **preprocess** — structural: the op skeleton of its planned program
  (``pred_program`` is written by the preprocess stage) vs the gold program's.
  Equivalent-but-differently-shaped programs read as misses; that looseness is
  documented and the teacher adjudicates those cases.
- **retriever** — derived: the gold program's numeric operands, minus
  ``const_*`` tokens and minus operands that are earlier gold answers in the
  conversation (those come from history, not the document). Number turns score
  the retrieved value against the gold answer.
- **calculator** — the gold answer, conditioned on retrieval having succeeded,
  which is what separates "wrong operand" from "wrong computation".

Everything reads the run dataframe the eval loop already produces — zero API
calls — and lands both as per-row columns and as run-level metrics.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pandas as pd

from convfinqa.evaluation.metrics import numeric_match, parse_program

ROW_COLUMNS = [
    "triage_turn_type_ok",
    "preprocess_skeleton_ok",
    "retriever_operand_recall",
    "calculator_ok",
]

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_float(text: Any) -> float | None:
    m = _NUM_RE.search(str(text).replace("%", ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def _values_match(a: Any, b: Any) -> bool:
    """Tolerant numeric equality: numeric_match plus scale slack (%, thousands)."""
    if numeric_match(a, b):
        return True
    fa, fb = _to_float(a), _to_float(b)
    if fa is None or fb is None:
        return False
    for scale in (1.0, 100.0, 0.01, 1000.0, 0.001):
        if math.isclose(fa * scale, fb, rel_tol=1e-3, abs_tol=1e-3):
            return True
    return False


def _skeleton(program: Any) -> list[str] | None:
    ops = parse_program(program)
    return [op for op, _ in ops] if ops else None


def gold_document_operands(
    gold_program: Any, prior_gold_answers: list[Any]
) -> list[str]:
    """Operands the retriever was responsible for finding in the document.

    Drops ``const_*`` tokens, ``#N`` step references, and operands that match an
    earlier gold answer of the same conversation — those come from history.
    """
    ops = parse_program(gold_program)
    if not ops:
        return []
    # parse_program normalises `const_100` to `100`, so recover the constant
    # values from the raw text and drop them from the lookup set too.
    const_vals = set(re.findall(r"const_([\w.]+)", str(gold_program)))
    out: list[str] = []
    for _, args in ops:
        for arg in args:
            a = str(arg).strip()
            if a.startswith(("const_", "#")) or a in const_vals:
                continue
            if _to_float(a) is None:
                continue
            if any(_values_match(a, g) for g in prior_gold_answers):
                continue
            out.append(a)
    return out


def _match_multiset(needed: list[str], retrieved: list[str]) -> tuple[int, list[str]]:
    """Match each needed operand to at most one retrieved value.

    Gold reuses the same operand value more than once when a program is e.g.
    ``divide(1200, 1200)``. Checking `needed` for set membership in `retrieved`
    would call that fully covered from a single correct value, hiding a real
    retrieval miss on the second sub-question. Greedy one-to-one consumption
    means a value can cover only one needed operand.
    """
    remaining = list(retrieved)
    missing: list[str] = []
    hits = 0
    for g in needed:
        idx = next((i for i, v in enumerate(remaining) if _values_match(v, g)), None)
        if idx is None:
            missing.append(g)
        else:
            hits += 1
            remaining.pop(idx)
    return hits, missing


def _retrieved_values(row: dict[str, Any]) -> list[str]:
    """Every value the retriever returned for this turn, from its capture."""
    raw = row.get("retriever_io")
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        io = json.loads(raw)
    except json.JSONDecodeError:
        return []
    answers = ((io or {}).get("output") or {}).get("answers") or []
    return [str(a.get("answer", "")) for a in answers if isinstance(a, dict)]


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-agent columns; returns the same frame, mutated."""
    triage_ok: list[bool | None] = []
    skeleton_ok: list[bool | None] = []
    recall: list[float | None] = []
    calc_ok: list[bool | None] = []

    gold_by_conv: dict[str, dict[int, Any]] = {}
    for r in df.itertuples():
        gold_by_conv.setdefault(r.report_id, {})[int(r.turn_index)] = r.gold_answer

    for r in df.itertuples():
        gold_tt = str(getattr(r, "gold_turn_type", "") or "").lower()
        pred_tt = str(getattr(r, "pred_turn_type", "") or "").lower()
        triage_ok.append(pred_tt == gold_tt if gold_tt else None)

        gold_sk = _skeleton(getattr(r, "gold_program", ""))
        if gold_sk is None:
            skeleton_ok.append(None)  # number turn — no plan to score
            recall_gold: list[str] = []
        else:
            skeleton_ok.append(_skeleton(getattr(r, "pred_program", "")) == gold_sk)
            prior = [
                v for t, v in gold_by_conv[r.report_id].items() if t < int(r.turn_index)
            ]
            recall_gold = gold_document_operands(r.gold_program, prior)

        retrieved = _retrieved_values(r._asdict())
        if gold_sk is None:
            # number turn: did the retriever surface the gold value?
            recall.append(
                1.0 if any(_values_match(v, r.gold_answer) for v in retrieved) else 0.0
            )
        elif recall_gold:
            hit, _missing = _match_multiset(recall_gold, retrieved)
            recall.append(round(hit / len(recall_gold), 4))
        else:
            recall.append(None)  # every operand came from history/constants

        calc_ok.append(bool(r.correct) if gold_sk is not None else None)

    df["triage_turn_type_ok"] = pd.Series(triage_ok, index=df.index, dtype=object)
    df["preprocess_skeleton_ok"] = pd.Series(skeleton_ok, index=df.index, dtype=object)
    df["retriever_operand_recall"] = pd.Series(recall, index=df.index, dtype=object)
    df["calculator_ok"] = pd.Series(calc_ok, index=df.index, dtype=object)
    return df


def _mean(series: pd.Series) -> float | None:
    vals = [
        v
        for v in series
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not vals:
        return None
    return round(sum(float(v) for v in vals) / len(vals), 6)


def run_metrics(df: pd.DataFrame) -> dict[str, float]:
    """The per-agent metric panel for one run. Skips metrics with no support."""
    if "triage_turn_type_ok" not in df.columns:
        score_rows(df)
    out: dict[str, float] = {}
    panel = {
        "acc_triage_turn_type": _mean(df["triage_turn_type_ok"]),
        "acc_preprocess_skeleton": _mean(df["preprocess_skeleton_ok"]),
        "retriever_operand_recall": _mean(df["retriever_operand_recall"]),
        "acc_calculator_exec": _mean(df["calculator_ok"]),
    }
    full = df[df["retriever_operand_recall"] == 1.0]
    if len(full):
        panel["calculator_acc_given_full_recall"] = _mean(full["calculator_ok"])
    for k, v in panel.items():
        if v is not None:
            out[k] = v
    return out


# The deterministic metric each targeted challenger must move (Phase C).
TARGET_METRIC = {
    "triage": "acc_triage_turn_type",
    "preprocess": "acc_preprocess_skeleton",
    "retriever": "retriever_operand_recall",
    "calculator": "acc_calculator_exec",
}


# --- Gold-derived attribution -------------------------------------------------
#
# The same checks, read in pipeline order: the first one that fails is where the
# turn first diverged from what gold says should have happened. Deterministic
# and free, and it is what `pick_target` ranks on.
#
# Rewritten 2026-09-04 after hand-attributing seven failures and disagreeing
# with the shipped rule on five of them. Re-scoring all twelve committed runs
# moved 37.4% of 554 first-wrong cases; the retriever's share fell by 62%. Three
# things were wrong:
#
# 1. **Plan shape was judged before operand coverage.** A skeleton mismatch
#    short-circuited before the retriever check ran, so preprocess won ties it
#    should have lost. The order now follows the data dependency: `pred_program`
#    is symbolic, its placeholders bind to the retriever's answers, so the plan
#    genuinely cannot be evaluated until coverage is known.
# 2. **Skeleton equality is not a valid test for a symbolic plan.** It punished
#    correct plans shaped differently from gold and passed wrong plans that
#    happened to share a shape. `bind_and_execute` replaces it.
# 3. **Every case had to name an agent.** The fallback charged the calculator,
#    so a bad gold label or an unreadable record was billed to an innocent
#    agent. `NON_AGENT` verdicts exist so that stops.

AGENT_ORDER = ("triage", "preprocess", "retriever", "calculator")

#: Verdicts that deliberately name no subagent. They are excluded from
#: targeting's numerator *and* denominator — charging an agent for a gold label
#: we do not trust is how a campaign spends an experiment on nothing.
NON_AGENT = ("gold_suspect", "ambiguous", "unscorable")

_NOT_FOUND_RE = re.compile(
    r"not (?:found|reported|available|given|stated|disclosed|provided|present)"
    r"|unavailable|not in the|no (?:value|figure|data)|cannot be|unknown|n/a",
    re.IGNORECASE,
)

_docs_cache: dict[str, str] | None = None


def report_documents() -> dict[str, str]:
    """Every report's raw text, keyed by report id, read once and cached.

    Needed for exactly one check — is the operand gold cites anywhere in the
    document? — but that check is what separates a retrieval failure from a
    dataset error, and the loop has been charging the former for the latter.
    """
    global _docs_cache
    if _docs_cache is None:
        from convfinqa.data.loader import load_raw_dataset

        docs: dict[str, str] = {}
        for items in load_raw_dataset().values():
            if not isinstance(items, list):
                continue
            for record in items:
                rid = record.get("id")
                if rid and rid not in docs:
                    docs[rid] = json.dumps(record.get("doc", ""))
        _docs_cache = docs
    return _docs_cache


def retrieved_pairs(row: Any) -> list[tuple[str, str]]:
    """The retriever's (sub-question, answer) pairs for one turn."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    raw = get("retriever_io")
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        io = json.loads(raw)
    except json.JSONDecodeError:
        return []
    answers = ((io or {}).get("output") or {}).get("answers") or []
    return [
        (str(a.get("question", "")), str(a.get("answer", "")))
        for a in answers
        if isinstance(a, dict)
    ]


def planned_sub_questions(row: Any) -> list[str]:
    """What preprocess asked for, as planned — not as answered."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    raw = get("pred_sub_questions")
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(q) for q in loaded] if isinstance(loaded, list) else []


def retriever_declined(row: Any) -> bool:
    """Did the retriever fail to return a usable number for some sub-question?

    Either it said so in words, or it returned something with no number in it,
    or a planned sub-question got no answer row at all.
    """
    pairs = retrieved_pairs(row)
    if not pairs:
        return True
    for _question, answer in pairs:
        if _to_float(answer) is None or _NOT_FOUND_RE.search(answer):
            return True
    return len(planned_sub_questions(row)) > len(pairs)


def missing_operands(row: Any, retrieved: list[str]) -> list[str]:
    """Gold operands the retriever was responsible for and did not return."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    prior = get("prior_gold_answers") or []
    needed = gold_document_operands(get("gold_program"), list(prior))
    _hits, missing = _match_multiset(needed, retrieved)
    return missing


def attribution_rule_id() -> str:
    """A fingerprint of the attribution logic itself, for staleness checks.

    `backfill_attribution` needs to know whether a stored fault count was
    produced by *this* rule. Its first guard asked only whether a run had been
    recomputed at all, which cannot notice the thing that matters: the rule
    changing underneath. That is not hypothetical — the rule changed twice in a
    day (the rewrite, then a tolerance and multiset fix), and the second time
    the guard silently reported every run as already done.

    Derived from source rather than a hand-bumped constant, because a constant
    someone must remember to bump is the same failure with an extra step.
    Hashed at the *module* level, not per named function: the attribution
    logic calls helpers (`_values_match`, `_match_multiset`, `bindings_from`,
    `parse_program`, ...) that a per-function enumeration would have to name
    exhaustively and would therefore forget the next one just as easily as a
    hand-bumped constant. A module hash cannot miss a helper defined in it.

    The cost of getting it wrong is asymmetric: a spurious recompute is a few
    seconds of arithmetic over committed CSVs and no API calls, while a missed
    one leaves the ledger pooling two different measurements under one Wilson
    bound. So this deliberately over-triggers — coarse on purpose — editing a
    comment or docstring anywhere in these modules is enough, and that is the
    cheap direction to be wrong in.
    """
    import hashlib
    import inspect
    import sys

    from convfinqa.evaluation import metrics, program_exec

    parts = [
        inspect.getsource(mod) for mod in (sys.modules[__name__], program_exec, metrics)
    ]
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:12]


def first_fault(row: Any, doc: str | None = None) -> str | None:
    """The first stage whose gold-derived check fails, in pipeline order.

    ``None`` means every check passed, which for a wrong answer is itself
    informative: the failure is outside what gold can adjudicate.

    `doc` is the report's raw text, used only to tell a retrieval miss from a
    gold operand that is not in the document at all. Omitted, it is looked up.
    """
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    from convfinqa.evaluation.program_exec import bind_and_execute

    # 1 — triage. The expected turn type follows from the gold program's shape,
    # so it is derived here rather than read from a column that could drift.
    gold_ops = parse_program(get("gold_program"))
    expected = "number" if gold_ops is None else "program"
    pred_tt = str(get("pred_turn_type") or "").lower()
    if pred_tt and pred_tt != expected:
        return "triage"

    pairs = retrieved_pairs(row)
    retrieved = [answer for _question, answer in pairs]

    if gold_ops is None:
        # Number turn: there is no plan and nothing to compute, so the retriever
        # either surfaced the value or it did not.
        if not any(_values_match(v, get("gold_answer")) for v in retrieved):
            return "retriever"
        return None if get("correct") else "calculator"

    # 2 — operand coverage, and who is answerable for a gap. This runs *before*
    # the plan check because the plan cannot be bound without it.
    missing = missing_operands(row, retrieved)
    if missing:
        if doc is None:
            doc = report_documents().get(str(get("report_id")), "")
        # Only claim the label is wrong when we actually hold the document. With
        # no text every operand reads as absent, which would turn an unmatched
        # report id — a stale manifest, a renamed split — into a store full of
        # `gold_suspect` and quietly empty the fault counts.
        if doc and any(m not in doc for m in missing):
            # Gold cites a number the report never states. Not a pipeline fault.
            return "gold_suspect"
        if retriever_declined(row):
            # Indistinguishable from the record: the retriever may have missed a
            # value that was there, or preprocess may have asked for something
            # the report does not answer. `Double_BLK` and `Double_IPG` produce
            # byte-identical evidence with opposite correct verdicts.
            return "ambiguous"
        # It answered every sub-question it was given, and the operand still is
        # not among them — so none of those questions asked for it.
        return "preprocess"

    # 3 — the plan, bound to the values it planned for and executed.
    reaches = bind_and_execute(get("pred_program"), retrieved, get("gold_answer"))
    if reaches is None:
        # The plan would not bind and run. Every mode of that seen in practice
        # is preprocess's: emitting a bare value or a malformed call instead of
        # a program (`1.0129…`, `A`, `subtract(subtract(B, A))`), or planning a
        # sub-question that nothing could answer, which leaves its placeholder
        # unbound. `unscorable` is kept for a record we genuinely cannot read.
        if parse_program(get("pred_program")) is None:
            return "preprocess"  # no plan was emitted at all
        if any(_to_float(a) is None for _q, a in pairs) or len(
            planned_sub_questions(row)
        ) > len(pairs):
            return "preprocess"  # planned an ask nothing could answer
        return "unscorable"
    if reaches is False:
        return "preprocess"

    # 4 — everything upstream is sound, so a wrong answer is the calculator's.
    return None if get("correct") else "calculator"


def attribute(row: Any, doc: str | None = None) -> str:
    """``first_fault`` with the honest fallback: nothing gold can see, so calculator."""
    return first_fault(row, doc) or "calculator"


def attribute_frame(df: pd.DataFrame) -> pd.Series:
    """Per-row gold-derived attribution for a scored run frame."""
    if "triage_turn_type_ok" not in df.columns:
        score_rows(df)
    docs = report_documents()
    rows = with_prior_gold(df)
    return pd.Series(
        [attribute(r, docs.get(str(r.get("report_id")), "")) for r in rows],
        index=df.index,
    )


def with_prior_gold(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Row dicts carrying `prior_gold_answers` — the conversation's earlier golds.

    Attribution needs them to know which of gold's operands the retriever was
    actually responsible for: an operand that is an earlier answer in the same
    conversation comes from history, not from the document.
    """
    gold_by_conv: dict[str, dict[int, Any]] = {}
    for r in df.itertuples():
        gold_by_conv.setdefault(r.report_id, {})[int(r.turn_index)] = r.gold_answer
    out: list[dict[str, Any]] = []
    for r in df.itertuples():
        row = r._asdict()
        row["prior_gold_answers"] = [
            v
            for t, v in gold_by_conv[row["report_id"]].items()
            if t < int(row["turn_index"])
        ]
        out.append(row)
    return out
