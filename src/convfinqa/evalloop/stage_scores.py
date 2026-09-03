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
            hit = sum(
                1 for g in recall_gold if any(_values_match(v, g) for v in retrieved)
            )
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


# --- Gold-derived attribution (M3/W0) ---------------------------------------
#
# The same four checks, read in pipeline order: the first one that fails is
# where the turn first diverged from what gold says should have happened. This
# replaces asking an LLM to attribute — it is free, deterministic, and it is the
# thing the per-agent panel was already computing. The teacher is told the
# answer and may dissent; a dissent is recorded, never silently overridden.

AGENT_ORDER = ("triage", "preprocess", "retriever", "calculator")


def first_fault(row: Any) -> str | None:
    """The first stage whose gold-derived check fails, in pipeline order.

    ``None`` means every check passed — for a wrong answer that is itself
    informative (the failure is outside what gold can adjudicate), and the
    caller attributes it to the calculator, which owns the final form.
    """
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))

    def _bad(value: Any) -> bool:
        return value is False or value == 0.0 or str(value).lower() == "false"

    if _bad(get("triage_turn_type_ok")):
        return "triage"
    if _bad(get("preprocess_skeleton_ok")):
        return "preprocess"
    recall = get("retriever_operand_recall")
    if recall is not None and not (isinstance(recall, float) and math.isnan(recall)):
        try:
            if float(recall) < 1.0:
                return "retriever"
        except (TypeError, ValueError):
            pass
    if _bad(get("calculator_ok")):
        return "calculator"
    return None


def attribute(row: Any) -> str:
    """``first_fault`` with the honest fallback: nothing gold can see, so calculator."""
    return first_fault(row) or "calculator"


def attribute_frame(df: pd.DataFrame) -> pd.Series:
    """Per-row gold-derived attribution for a scored run frame."""
    if "triage_turn_type_ok" not in df.columns:
        score_rows(df)
    return pd.Series([attribute(r._asdict()) for r in df.itertuples()], index=df.index)
