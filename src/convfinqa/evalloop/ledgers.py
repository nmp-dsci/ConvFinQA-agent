"""Three append-only ledgers: every diagnosis, every rewrite, every gate verdict.

The loop's record used to be scattered — per-run ``diagnoses_<v>_<stamp>.jsonl``
files, ``proposal.json`` / ``prompt_diff.json`` artifacts on propose runs,
``verdict.json`` / ``flips.json`` on gate runs — joined only by reading MLflow
back with a search per question. Three flat files replace that as the primary
record, and MLflow keeps a copy of each appended batch on the run that produced
it (``ledger_rows.jsonl``), so the two can never disagree about what was written.

- ``diagnoses.jsonl`` — one line per diagnosed case.
- ``rewrites.jsonl`` — one line per edit the teacher made.
- ``gates.jsonl`` — one line per gate verdict.

They join by id: a rewrite names the ``diagnosis_ids`` it was written from, a
gate names the ``rewrite_id`` it judged. Every line in a file has the same
columns; nested values are stored as JSON strings so a row stays one row and
``pd.read_json(path, lines=True)`` is the table. Nothing ever rewrites or
reorders a line — extension is by appending columns with defaults, which is why
`load` fills defaults for lines written before a column existed.

The directory is ``evaluation/diagnostics/evalloop/`` (committed, like every
other diagnostics artifact); tests point it elsewhere through the
``CONVFINQA_LEDGER_DIR`` environment variable or the module attribute.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.config import EVAL_ROOT

LEDGER_DIR = EVAL_ROOT / "diagnostics" / "evalloop"
LEDGER_DIR_ENV = "CONVFINQA_LEDGER_DIR"

RUNTIMES = ("multi_agent", "agent_sdk")
#: The writer's edit vocabulary. ``rewrite`` is the multi-agent prompt writer's
#: complete replacement of one subagent's prompt — it is none of the finer
#: kinds, and calling it one would be a lie about what happened.
CHANGE_KINDS = ("rule", "example", "criterion", "removal", "reorder", "rewrite")

# ── Frozen schemas ────────────────────────────────────────────────────────
#
# Each spec is an ordered (column, default) sequence. The order *is* the schema:
# a reordering changes the hash the tests pin, and an unknown column is refused
# at append time. Extension is by appending a (column, default) pair.

_DIAGNOSES_SPEC: tuple[tuple[str, Any], ...] = (
    # identity
    ("diagnosis_id", ""),
    ("diagnosed_at", ""),
    ("runtime", "multi_agent"),
    ("version", ""),
    ("prompt_hash", ""),
    ("eval_run_id", ""),
    ("diagnosis_run_id", ""),
    ("split", ""),
    ("draw_seed", None),
    ("report_id", ""),
    ("question_id", ""),
    ("turn_index", None),
    ("diagnoser_model", ""),
    # inputs · case
    ("question", ""),
    ("history_text", ""),
    ("gold_turn_type", ""),
    ("gold_answer", ""),
    ("gold_program", ""),
    ("pred_turn_type", ""),
    ("pred_answer", ""),
    ("pred_program", ""),
    ("sub_questions", "[]"),
    ("retrieved", "[]"),
    ("calc_trajectory", "[]"),
    # inputs · gold flags
    ("triage_turn_type_ok", None),
    ("preprocess_skeleton_ok", None),
    ("preprocess_plan_ok", None),
    ("retriever_operand_recall", None),
    ("calc_ok", None),
    ("derived_agent", ""),
    ("missing_gold_operands", "[]"),
    # outputs
    ("stage", ""),
    ("label", ""),
    ("what_went_wrong", ""),
    ("evidence", ""),
    ("attribution_reason", ""),
    ("fix_hint", ""),
    ("confidence", None),
    ("gold_suspect", False),
    ("attribution_disputed", False),
    ("adjudicated", False),
    ("adjudication_reason", ""),
    # cost
    ("input_tokens", 0),
    ("output_tokens", 0),
    ("cost_usd", 0.0),
    ("latency_s", None),
)

_REWRITES_SPEC: tuple[tuple[str, Any], ...] = (
    # identity
    ("edit_id", ""),
    ("rewrite_id", ""),
    ("proposed_at", ""),
    ("runtime", "multi_agent"),
    ("campaign", ""),
    ("experiment_n", None),
    ("base_version", ""),
    ("new_version", ""),
    ("prompt_hash_before", ""),
    ("prompt_hash_after", ""),
    ("teacher_run_id", ""),
    ("teacher_model", ""),
    # inputs
    ("target", ""),
    ("failure_class", ""),
    ("n_diagnoses", 0),
    ("diagnosis_ids", "[]"),
    ("wilson_lower", None),
    ("rank", None),
    ("evidence_summary", "{}"),
    ("prior_attempts", "[]"),
    # outputs
    ("change_kind", "rewrite"),
    ("edit_text", ""),
    ("diff", ""),
    ("rationale", ""),
    ("prompt_chars_before", 0),
    ("prompt_chars_after", 0),
    ("validate_ok", True),
    # cost
    ("input_tokens", 0),
    ("output_tokens", 0),
    ("cost_usd", 0.0),
    ("latency_s", None),
)

_GATES_SPEC: tuple[tuple[str, Any], ...] = (
    # identity
    ("gate_id", ""),
    ("gated_at", ""),
    ("runtime", "multi_agent"),
    ("campaign", ""),
    ("experiment_n", None),
    ("rewrite_id", ""),
    ("baseline_version", ""),
    ("candidate_version", ""),
    ("baseline_hash", ""),
    ("candidate_hash", ""),
    ("split", ""),
    ("gate_run_id", ""),
    ("baseline_eval_run_id", ""),
    ("candidate_eval_run_id", ""),
    # evidence
    ("n_paired", 0),
    ("baseline_acc", None),
    ("candidate_acc", None),
    ("delta_pp", None),
    ("fixed", 0),
    ("broken", 0),
    ("p_value", None),
    ("ci_low", None),
    ("ci_high", None),
    ("flips_by_class", "{}"),
    ("panel_baseline", "{}"),
    ("panel_candidate", "{}"),
    # verdict
    ("promoted", False),
    ("reason", ""),
    ("consecutive_rejections", 0),
    ("champion_after", ""),
)

_SPECS: dict[str, tuple[tuple[str, Any], ...]] = {
    "diagnoses": _DIAGNOSES_SPEC,
    "rewrites": _REWRITES_SPEC,
    "gates": _GATES_SPEC,
}

DIAGNOSES_COLUMNS: tuple[str, ...] = tuple(c for c, _ in _DIAGNOSES_SPEC)
REWRITES_COLUMNS: tuple[str, ...] = tuple(c for c, _ in _REWRITES_SPEC)
GATES_COLUMNS: tuple[str, ...] = tuple(c for c, _ in _GATES_SPEC)
COLUMNS: dict[str, tuple[str, ...]] = {
    "diagnoses": DIAGNOSES_COLUMNS,
    "rewrites": REWRITES_COLUMNS,
    "gates": GATES_COLUMNS,
}
LEDGERS = tuple(COLUMNS)

#: The column each ledger is keyed on when `load(version=...)` filters it.
_VERSION_COLUMN = {
    "diagnoses": "version",
    "rewrites": "new_version",
    "gates": "candidate_version",
}


def defaults(name: str) -> dict[str, Any]:
    """The default value of every column of one ledger, in column order."""
    return {c: d for c, d in _spec(name)}


def _spec(name: str) -> tuple[tuple[str, Any], ...]:
    try:
        return _SPECS[name]
    except KeyError:
        raise ValueError(f"unknown ledger {name!r}; one of {LEDGERS}") from None


# ── Location ──────────────────────────────────────────────────────────────


def ledger_dir() -> Path:
    """Where the three files live: the env override, else `LEDGER_DIR`."""
    override = os.environ.get(LEDGER_DIR_ENV)
    return Path(override) if override else LEDGER_DIR


def path(name: str) -> Path:
    """The file one ledger is appended to."""
    _spec(name)
    return ledger_dir() / f"{name}.jsonl"


# ── Ids and time ──────────────────────────────────────────────────────────


def now_iso() -> str:
    """UTC timestamp, seconds precision — the form every ``*_at`` column holds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    """A fresh id: ``<prefix>-<utc stamp>-<random>``, sortable by creation."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def experiment_number(label: str | None) -> int | None:
    """``c01-e03`` → 3. None when the label carries no experiment number."""
    if not label:
        return None
    found = re.search(r"-e(\d+)$", label)
    return int(found.group(1)) if found else None


# ── Append / load ─────────────────────────────────────────────────────────


def _as_cell(value: Any) -> Any:
    """Nested values become JSON strings; numpy scalars become Python ones."""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()  # numpy scalar
        except (AttributeError, ValueError):
            return value
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def normalise(name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """One row in column order with defaults filled. Refuses unknown columns."""
    spec = _spec(name)
    known = {c for c, _ in spec}
    unknown = sorted(set(row) - known)
    if unknown:
        raise ValueError(
            f"{name} ledger has no column(s) {unknown} — the schema is frozen; "
            "extend it by appending a column with a default"
        )
    out: dict[str, Any] = {}
    for column, default in spec:
        value = row.get(column, default)
        out[column] = default if value is None else _as_cell(value)
    return out


def append(name: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Append rows to one ledger. Returns exactly the lines written.

    Opens with ``"a"``, writes, fsyncs: the file is the record, so a line is
    either durably there or was never claimed to be. Never rewrites, never
    reorders, never touches an existing line.
    """
    normalised = [normalise(name, r) for r in rows]
    if not normalised:
        return []
    target = path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as fh:
        for row in normalised:
            fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return normalised


def _read_lines(target: Path) -> list[dict[str, Any]]:
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                out.append(loaded)
    return out


def load(
    name: str,
    *,
    runtime: str | None = None,
    version: str | None = None,
    campaign: str | None = None,
) -> pd.DataFrame:
    """One ledger as a frame with every column, defaults filled for old lines.

    A missing file is an empty frame with the columns, so callers never branch
    on existence. `version` filters the ledger's own version column
    (`diagnoses.version`, `rewrites.new_version`, `gates.candidate_version`);
    `campaign` is refused on the diagnoses ledger, which has no such column.
    """
    spec = _spec(name)
    columns = [c for c, _ in spec]
    lines = _read_lines(path(name))
    records = [{c: line.get(c, d) for c, d in spec} for line in lines]
    frame = pd.DataFrame.from_records(records, columns=columns)
    if runtime is not None:
        frame = frame[frame["runtime"] == runtime]
    if version is not None:
        frame = frame[frame[_VERSION_COLUMN[name]] == version]
    if campaign is not None:
        if "campaign" not in columns:
            raise ValueError(f"the {name} ledger has no campaign column")
        frame = frame[frame["campaign"] == campaign]
    return frame.reset_index(drop=True)


def _ids(cell: Any) -> list[str]:
    """Decode a JSON-list cell of ids; tolerate an already-decoded list."""
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if isinstance(cell, str) and cell.strip():
        try:
            loaded = json.loads(cell)
        except json.JSONDecodeError:
            return []
        return [str(x) for x in loaded] if isinstance(loaded, list) else []
    return []


def trace(
    *, question_id: str | None = None, edit_id: str | None = None
) -> dict[str, pd.DataFrame]:
    """Follow one case forward, or one edit both ways, through the three files.

    ``question_id`` → the diagnoses of that question, the rewrites written from
    any of them (rewrites ⋈ diagnoses on ``diagnosis_ids``), and the gates that
    judged those rewrites (gates ⋈ rewrites on ``rewrite_id``).

    ``edit_id`` → that edit, the diagnoses it was written from, and its gates.
    """
    if (question_id is None) == (edit_id is None):
        raise ValueError("trace takes exactly one of question_id or edit_id")
    diagnoses, rewrites, gates = load("diagnoses"), load("rewrites"), load("gates")
    if question_id is not None:
        d_hit = diagnoses[diagnoses["question_id"] == question_id]
        wanted = set(d_hit["diagnosis_id"])
        r_hit = rewrites[
            rewrites["diagnosis_ids"].map(lambda c: bool(wanted & set(_ids(c))))
        ]
    else:
        r_hit = rewrites[rewrites["edit_id"] == edit_id]
        wanted = {i for c in r_hit["diagnosis_ids"] for i in _ids(c)}
        d_hit = diagnoses[diagnoses["diagnosis_id"].isin(wanted)]
    g_hit = gates[gates["rewrite_id"].isin(set(r_hit["rewrite_id"]))]
    return {
        "diagnoses": d_hit.reset_index(drop=True),
        "rewrites": r_hit.reset_index(drop=True),
        "gates": g_hit.reset_index(drop=True),
    }


# ── flips_by_class ────────────────────────────────────────────────────────

#: Given one flip (as `Flip.as_dict()` writes it) and which side it is on
#: (``"fixed"`` or ``"broken"``), the class to file it under — normally the
#: first-fault stage of the arm on which the question was wrong.
AttributionOf = Callable[[Mapping[str, Any], str], str | None]

UNATTRIBUTED = "unattributed"


def flips_by_class(
    flips: Mapping[str, Any], attribution_of_row: AttributionOf
) -> dict[str, dict[str, int]]:
    """``{class: {fixed, broken}}`` from a gate's ``flips.json``.

    A fixed question was wrong on the baseline, a broken one wrong on the
    candidate, so the caller's `attribution_of_row` is told which side it is
    classifying; `attribution_from_frames` builds the usual one. A flip the
    function cannot place lands under ``unattributed`` rather than vanishing —
    the totals must still equal the gate's fixed/broken counts.
    """
    out: dict[str, dict[str, int]] = {}
    for side in ("fixed", "broken"):
        for flip in flips.get(side) or []:
            cls = attribution_of_row(flip, side) or UNATTRIBUTED
            bucket = out.setdefault(str(cls), {"fixed": 0, "broken": 0})
            bucket[side] += 1
    return out


def attribution_from_frames(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> AttributionOf:
    """First-fault attribution per flip, read off the arm the question failed on.

    Uses `stage_scores.attribute_frame`, which scores the frame (no API calls)
    and needs the report documents for the gold-suspect check; when those are
    unavailable every flip classifies as ``unattributed`` rather than raising,
    because a gate must never fail on its own bookkeeping.
    """
    from convfinqa.evalloop import stage_scores

    def _lookup(df: pd.DataFrame) -> dict[tuple[str, int], str]:
        if df.empty:
            return {}
        try:
            verdicts = stage_scores.attribute_frame(df)
        except Exception:  # noqa: BLE001 — bookkeeping must not sink a gate
            return {}
        return {
            (str(r.report_id), int(r.turn_index)): str(v)
            for r, v in zip(df.itertuples(), verdicts, strict=True)
        }

    tables = {"fixed": _lookup(baseline), "broken": _lookup(candidate)}

    def _of(flip: Mapping[str, Any], side: str) -> str | None:
        key = (str(flip.get("report_id")), int(flip.get("q_order", -1)))
        return tables.get(side, {}).get(key)

    return _of


# ── MLflow mirror ─────────────────────────────────────────────────────────

LEDGER_ROWS_ARTIFACT = "ledger_rows.jsonl"


def _sum_column(rows: Sequence[Mapping[str, Any]], column: str) -> float | None:
    values: list[float] = [float(r[column]) for r in rows if r.get(column) is not None]
    return sum(values) if values else None


def log_rows_to_run(rec: Any, rows: Sequence[Mapping[str, Any]], name: str) -> None:
    """Mirror an appended batch onto the MLflow run that produced it.

    The lines go up verbatim as ``ledger_rows.jsonl`` and the scalar evidence
    as metrics, so a run and the file can be checked against each other. A
    null recorder (no store) makes this a no-op, like every other logging call.
    """
    if not rows:
        return
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / LEDGER_ROWS_ARTIFACT
        p.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        rec.artifact(p)
    metrics: dict[str, float] = {f"ledger_{name}_n_rows": float(len(rows))}
    if name == "gates":
        last = rows[-1]
        for column in ("delta_pp", "p_value", "fixed", "broken"):
            if last.get(column) is not None:
                metrics[f"ledger_{column}"] = float(last[column])
    if name == "rewrites":
        metrics["ledger_n_edits"] = float(len(rows))
    cost = _sum_column(rows, "cost_usd")
    if cost is not None:
        metrics["ledger_cost_usd"] = cost
    rec.metrics(metrics)


# ── Row builders ──────────────────────────────────────────────────────────
#
# The wiring in teacher.py / gate.py is a call to one of these plus an
# `append`. They are pure so they can be tested without the SDK harness.


def _get(row: Any, key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        value = row.get(key, default)
    else:
        value = getattr(row, key, default)
    if isinstance(value, float) and value != value:
        return default
    return value


def _calc_trajectory(row: Any) -> list[Any]:
    raw = _get(row, "calculator_io")
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        io = json.loads(raw)
    except json.JSONDecodeError:
        return []
    trajectory = (io or {}).get("trajectory") if isinstance(io, dict) else None
    return list(trajectory) if isinstance(trajectory, list) else []


def usage_cost(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    """The four cost columns from one SDK call's usage dict (see `sdk.py`)."""
    usage = usage or {}
    raw = usage.get("usage") or {}
    duration = usage.get("duration_ms")
    return {
        "input_tokens": int(
            (raw.get("input_tokens") if isinstance(raw, dict) else 0) or 0
        ),
        "output_tokens": int(
            (raw.get("output_tokens") if isinstance(raw, dict) else 0) or 0
        ),
        "cost_usd": float(usage.get("total_cost_usd") or 0.0),
        "latency_s": round(float(duration) / 1000.0, 3) if duration else None,
    }


def diagnosis_row(
    diagnosis: Mapping[str, Any],
    case: Any,
    *,
    version: str,
    runtime: str = "multi_agent",
    prompt_hash: str = "",
    eval_run_id: str = "",
    diagnosis_run_id: str = "",
    split: str = "",
    draw_seed: int | None = None,
    diagnoser_model: str = "",
    usage: Mapping[str, Any] | None = None,
    diagnosed_at: str | None = None,
    diagnosis_id: str | None = None,
) -> dict[str, Any]:
    """One diagnoses-ledger row from a teacher `Diagnosis` dict and its CSV row.

    `diagnosis` is the shape `diagnose_run` writes (and the per-run files hold):
    ``failed_agent`` → ``stage``, ``failure_mode`` → ``label``,
    ``proposed_rule`` → ``fix_hint``; everything else keeps its name. `case` is
    the scored first-wrong row (a Series or dict) — the report text is not
    copied, join on ``report_id`` for it.
    """
    from convfinqa.evalloop import stage_scores

    try:
        pairs = stage_scores.retrieved_pairs(case)
        retrieved = [{"question": q, "answer": a} for q, a in pairs]
        missing = stage_scores.missing_operands(case, [a for _q, a in pairs])
        sub_questions = stage_scores.planned_sub_questions(case)
    except Exception:  # noqa: BLE001 — an unreadable capture is an empty one
        retrieved, missing, sub_questions = [], [], []

    turn = _get(case, "turn_index", diagnosis.get("turn_index"))
    return {
        "diagnosis_id": diagnosis_id or diagnosis.get("diagnosis_id") or new_id("d"),
        "diagnosed_at": diagnosed_at or now_iso(),
        "runtime": runtime,
        "version": version,
        "prompt_hash": prompt_hash,
        "eval_run_id": eval_run_id or str(_get(case, "run_id", "") or ""),
        "diagnosis_run_id": diagnosis_run_id,
        "split": split or str(_get(case, "split", "") or ""),
        "draw_seed": draw_seed,
        "report_id": str(_get(case, "report_id", diagnosis.get("report_id", ""))),
        "question_id": str(
            _get(case, "question_id", diagnosis.get("question_id", "")) or ""
        ),
        "turn_index": int(turn) if turn is not None else None,
        "diagnoser_model": diagnoser_model,
        "question": str(_get(case, "question", "") or ""),
        "history_text": str(_get(case, "history_text", "") or ""),
        "gold_turn_type": str(_get(case, "gold_turn_type", "") or ""),
        "gold_answer": str(_get(case, "gold_answer", "") or ""),
        "gold_program": str(_get(case, "gold_program", "") or ""),
        "pred_turn_type": str(_get(case, "pred_turn_type", "") or ""),
        "pred_answer": str(_get(case, "pred_answer", "") or ""),
        "pred_program": str(_get(case, "pred_program", "") or ""),
        "sub_questions": sub_questions,
        "retrieved": retrieved,
        "calc_trajectory": _calc_trajectory(case),
        "triage_turn_type_ok": _get(case, "triage_turn_type_ok"),
        "preprocess_skeleton_ok": _get(case, "preprocess_skeleton_ok"),
        "preprocess_plan_ok": _get(case, "preprocess_plan_ok"),
        "retriever_operand_recall": _get(case, "retriever_operand_recall"),
        "calc_ok": _get(case, "calculator_ok"),
        "derived_agent": str(diagnosis.get("derived_agent") or ""),
        "missing_gold_operands": missing,
        "stage": str(diagnosis.get("failed_agent") or ""),
        "label": str(diagnosis.get("failure_mode") or ""),
        "what_went_wrong": str(diagnosis.get("what_went_wrong") or ""),
        "evidence": str(diagnosis.get("evidence") or ""),
        "attribution_reason": str(diagnosis.get("attribution_reason") or ""),
        "fix_hint": str(diagnosis.get("proposed_rule") or ""),
        "confidence": diagnosis.get("confidence"),
        "gold_suspect": bool(diagnosis.get("gold_suspect", False)),
        "attribution_disputed": bool(diagnosis.get("attribution_disputed", False)),
        "adjudicated": bool(diagnosis.get("adjudicated", False)),
        "adjudication_reason": str(diagnosis.get("adjudication_reason") or ""),
        **usage_cost(usage),
    }


def rewrite_row(
    *,
    target: str,
    base_version: str,
    new_version: str,
    prompt_before: str,
    prompt_after: str,
    diff: str,
    rationale: str,
    edit_text: str = "",
    failure_class: str = "",
    diagnosis_ids: Sequence[str] = (),
    n_diagnoses: int | None = None,
    evidence_summary: Mapping[str, Any] | None = None,
    prior_attempts: Sequence[Mapping[str, Any]] = (),
    wilson_lower: float | None = None,
    rank: int | None = None,
    validate_ok: bool = True,
    change_kind: str = "rewrite",
    runtime: str = "multi_agent",
    campaign: str | None = None,
    label: str | None = None,
    teacher_run_id: str = "",
    teacher_model: str = "",
    usage: Mapping[str, Any] | None = None,
    proposed_at: str | None = None,
    rewrite_id: str | None = None,
    edit_id: str | None = None,
) -> dict[str, Any]:
    """One rewrites-ledger row. A whole-prompt replacement is one edit."""
    from convfinqa.tracking.prompt_ledger import prompt_hash

    if change_kind not in CHANGE_KINDS:
        raise ValueError(f"change_kind {change_kind!r} not in {CHANGE_KINDS}")
    return {
        "edit_id": edit_id or new_id("e"),
        "rewrite_id": rewrite_id or new_id("rw"),
        "proposed_at": proposed_at or now_iso(),
        "runtime": runtime,
        "campaign": campaign or "",
        "experiment_n": experiment_number(label),
        "base_version": base_version,
        "new_version": new_version,
        "prompt_hash_before": prompt_hash(prompt_before),
        "prompt_hash_after": prompt_hash(prompt_after),
        "teacher_run_id": teacher_run_id,
        "teacher_model": teacher_model,
        "target": target,
        "failure_class": failure_class,
        "n_diagnoses": len(diagnosis_ids) if n_diagnoses is None else n_diagnoses,
        "diagnosis_ids": list(diagnosis_ids),
        "wilson_lower": wilson_lower,
        "rank": rank,
        "evidence_summary": dict(evidence_summary or {}),
        "prior_attempts": [dict(a) for a in prior_attempts],
        "change_kind": change_kind,
        "edit_text": edit_text or prompt_after,
        "diff": diff,
        "rationale": rationale,
        "prompt_chars_before": len(prompt_before),
        "prompt_chars_after": len(prompt_after),
        "validate_ok": bool(validate_ok),
        **usage_cost(usage),
    }


def gate_row(
    stats: Mapping[str, Any],
    *,
    baseline_version: str,
    candidate_version: str,
    promoted: bool,
    reason: str,
    flips: Mapping[str, Any] | None = None,
    attribution_of_row: AttributionOf | None = None,
    panel_baseline: Mapping[str, Any] | None = None,
    panel_candidate: Mapping[str, Any] | None = None,
    baseline_hash: str = "",
    candidate_hash: str = "",
    baseline_eval_run_id: str = "",
    candidate_eval_run_id: str = "",
    gate_run_id: str = "",
    rewrite_id: str | None = None,
    runtime: str = "multi_agent",
    campaign: str | None = None,
    label: str | None = None,
    consecutive_rejections: int | None = None,
    champion_after: str | None = None,
    gated_at: str | None = None,
    gate_id: str | None = None,
) -> dict[str, Any]:
    """One gates-ledger row from `gate.gate_runs` statistics and the verdict.

    `stats` is the dict `gate_runs` returns (``accuracy_delta`` as a fraction,
    ``fail_to_pass``/``pass_to_fail``, ``cluster_p_one_sided``, ``delta_ci_*``).
    `champion_after` defaults to the registry's current champion — pass it
    explicitly when the row is written before the promotion is applied.
    """
    if champion_after is None:
        try:
            from convfinqa.tracking import registry

            champion_after = registry.champion() or ""
        except Exception:  # noqa: BLE001 — no registry is an empty cell, not a crash
            champion_after = ""
    by_class = (
        flips_by_class(flips, attribution_of_row)
        if flips is not None and attribution_of_row is not None
        else {}
    )
    delta = stats.get("accuracy_delta")
    return {
        "gate_id": gate_id or new_id("g"),
        "gated_at": gated_at or now_iso(),
        "runtime": runtime,
        "campaign": campaign or "",
        "experiment_n": experiment_number(label),
        "rewrite_id": rewrite_id or "",
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
        "baseline_hash": baseline_hash,
        "candidate_hash": candidate_hash,
        "split": str(stats.get("evidence_split", "") or ""),
        "gate_run_id": gate_run_id,
        "baseline_eval_run_id": baseline_eval_run_id,
        "candidate_eval_run_id": candidate_eval_run_id,
        "n_paired": int(stats.get("n_compared", 0) or 0),
        "baseline_acc": stats.get("baseline_accuracy"),
        "candidate_acc": stats.get("candidate_accuracy"),
        "delta_pp": round(float(delta) * 100.0, 4) if delta is not None else None,
        "fixed": int(stats.get("fail_to_pass", 0) or 0),
        "broken": int(stats.get("pass_to_fail", 0) or 0),
        "p_value": stats.get("cluster_p_one_sided"),
        "ci_low": stats.get("delta_ci_lo"),
        "ci_high": stats.get("delta_ci_hi"),
        "flips_by_class": by_class,
        "panel_baseline": dict(panel_baseline or {}),
        "panel_candidate": dict(panel_candidate or {}),
        "promoted": bool(promoted),
        "reason": reason,
        "consecutive_rejections": (
            consecutive_rejections
            if consecutive_rejections is not None
            else (0 if promoted else 1)
        ),
        "champion_after": champion_after,
    }


def bundle_hash(version: str) -> str:
    """The composition's four prompt hashes joined — one id for a bundle."""
    try:
        from convfinqa.tracking import prompt_ledger

        comp = prompt_ledger.resolve(version)
        return ".".join(comp[a]["hash"] for a in prompt_ledger.AGENTS)
    except Exception:  # noqa: BLE001 — an unresolvable version is an empty cell
        return ""


def agent_prompt_hash(version: str, agent: str) -> str:
    """One agent's prompt hash inside a bundle version, or "" when unknown."""
    try:
        from convfinqa.tracking import prompt_ledger

        return str(prompt_ledger.resolve(version)[agent]["hash"])
    except Exception:  # noqa: BLE001
        return ""


def eval_run_param(run_id: str, key: str) -> str | None:
    """One param of a recorded eval run (e.g. ``train_draw_seed``), best effort."""
    if not run_id:
        return None
    try:
        from mlflow.tracking import MlflowClient

        from convfinqa.tracking import mlflow_log

        client = MlflowClient(tracking_uri=mlflow_log.tracking_uri())
        value = client.get_run(run_id).data.params.get(key)
        return str(value) if value is not None else None
    except Exception:  # noqa: BLE001 — an unreachable store is an empty cell
        return None


def eval_run_ids(csv_path: Path | str | None) -> str:
    """The ``run_id`` a predictions CSV was written under ("" when unknown)."""
    if not csv_path or not Path(csv_path).exists():
        return ""
    try:
        head = pd.read_csv(csv_path, usecols=["run_id"], nrows=1)
    except Exception:  # noqa: BLE001 — older CSVs have no run_id column
        return ""
    return str(head["run_id"].iloc[0]) if len(head) else ""


# ── Backfill ──────────────────────────────────────────────────────────────

_DIAG_FILE_RE = re.compile(r"^diagnoses_(v\d+(?:_\d+)?)_(\d{8}_\d{6})\.jsonl$")
_CSV_STAMP_RE = re.compile(r"-(\d{8}_\d{6})\.csv$")


def _stamp_to_iso(stamp: str) -> str:
    return datetime.strptime(stamp, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")


def _csv_candidates(predictions_dir: Path, version: str) -> list[tuple[str, Path]]:
    """Committed CSVs of `version`, as (stamp, path), oldest first."""
    out: list[tuple[str, Path]] = []
    for p in predictions_dir.glob("evalloop-*.csv"):
        name = p.name
        if f"-{version}-" not in name and f"-{version}·" not in name:
            continue
        found = _CSV_STAMP_RE.search(name)
        if found:
            out.append((found.group(1), p))
    return sorted(out)


def _match_csv(
    predictions_dir: Path, version: str, stamp: str, question_ids: set[str]
) -> Path | None:
    """The latest CSV of `version` written before `stamp` holding every case."""
    for csv_stamp, p in reversed(_csv_candidates(predictions_dir, version)):
        if csv_stamp > stamp:
            continue
        try:
            ids = set(
                pd.read_csv(p, usecols=["question_id"])["question_id"].astype(str)
            )
        except Exception:  # noqa: BLE001 — an unreadable CSV is not a match
            continue
        if question_ids <= ids:
            return p
    return None


def _recomputed_attribution(cases: pd.DataFrame) -> dict[str, str]:
    """Current-rule first-fault per question id, or {} when it cannot run."""
    from convfinqa.evalloop import stage_scores

    try:
        docs = stage_scores.report_documents()
    except Exception:  # noqa: BLE001 — no dataset on disk: keep the recorded verdicts
        return {}
    out: dict[str, str] = {}
    for row in stage_scores.with_prior_gold(cases):
        try:
            out[str(row.get("question_id"))] = stage_scores.attribute(
                row, docs.get(str(row.get("report_id")), "")
            )
        except Exception:  # noqa: BLE001
            continue
    return out


def _mlflow_runs_by_kind(kind: str) -> tuple[list[Any], Any] | None:
    """Every run of one kind in the optimisation experiment, or None if down."""
    try:
        from convfinqa.evalloop import ledger

        client = ledger._client()
        return ledger._runs(client, ledger.OPTIMIZATION_EXPERIMENT, kind, 500), client
    except Exception:  # noqa: BLE001 — unreachable store: file-derived part only
        return None


def backfill_ledgers(
    *,
    diagnostics_dir: Path | None = None,
    predictions_dir: Path | None = None,
    use_mlflow: bool = True,
) -> dict[str, int]:
    """Seed the three ledgers from what already exists. Idempotent.

    Diagnoses come from the per-run ``diagnoses_<version>_<stamp>.jsonl`` files
    (runtime ``multi_agent``), each case joined back to its committed CSV by
    version + question id to fill the inputs and gold flags — the CSV chosen is
    the latest one of that version written before the diagnosis file that holds
    every case. Attribution is recomputed under the current rule (the same
    position `ledger.backfill_attribution` takes) with the file's adjudicated
    verdicts reused; a case with no CSV is skipped and counted, never invented.

    Rewrites and gates come from MLflow's propose and gate runs when the store
    is reachable; when it is not, only the file-derived part is written and
    ``mlflow_reachable`` says so.

    A row is refused when its identity is already in the ledger: (runtime,
    version, question_id, diagnosed_at) for a diagnosis, the teacher run id for
    a rewrite, the gate run id for a gate.
    """
    from convfinqa.evalloop import stage_scores

    diag_dir = diagnostics_dir or ledger_dir()
    pred_dir = predictions_dir or (EVAL_ROOT / "predictions" / "evalloop")
    counts: dict[str, int] = {
        "diagnoses": 0,
        "diagnoses_existing": 0,
        "diagnoses_no_csv": 0,
        "rewrites": 0,
        "rewrites_existing": 0,
        "gates": 0,
        "gates_existing": 0,
        "mlflow_reachable": 0,
    }

    # ── diagnoses, from the files ──
    existing = load("diagnoses")
    seen = {
        (r.runtime, r.version, str(r.question_id), r.diagnosed_at)
        for r in existing.itertuples()
    }
    for file in sorted(diag_dir.glob("diagnoses_v*.jsonl")):
        found = _DIAG_FILE_RE.match(file.name)
        if not found:
            continue
        version, stamp = found.group(1), found.group(2)
        diagnosed_at = _stamp_to_iso(stamp)
        lines = _read_lines(file)
        qids = {str(d.get("question_id") or "") for d in lines}
        csv = _match_csv(pred_dir, version, stamp, qids - {""})
        if csv is None:
            counts["diagnoses_no_csv"] += len(lines)
            continue
        from convfinqa.evalloop.teacher import first_wrong_cases

        cases = first_wrong_cases(csv)
        stage_scores._ensure_scored(cases)
        by_qid = {str(r.question_id): r for _, r in cases.iterrows()}
        recomputed = _recomputed_attribution(cases)
        hashes = {a: agent_prompt_hash(version, a) for a in stage_scores.AGENT_ORDER}
        rows: list[dict[str, Any]] = []
        for d in lines:
            qid = str(d.get("question_id") or "")
            key = ("multi_agent", version, qid, diagnosed_at)
            if key in seen:
                counts["diagnoses_existing"] += 1
                continue
            case = by_qid.get(qid)
            if case is None:
                counts["diagnoses_no_csv"] += 1
                continue
            derived = str(d.get("derived_agent") or d.get("failed_agent") or "")
            fresh = recomputed.get(qid)
            if fresh is not None and not (
                fresh == "ambiguous" and d.get("adjudicated")
            ):
                derived = fresh
            rows.append(
                diagnosis_row(
                    {**d, "derived_agent": derived},
                    case,
                    version=version,
                    prompt_hash=hashes.get(derived, ""),
                    diagnosed_at=diagnosed_at,
                )
            )
            seen.add(key)
        append("diagnoses", rows)
        counts["diagnoses"] += len(rows)

    if not use_mlflow:
        return counts
    proposals = _mlflow_runs_by_kind("propose")
    verdicts = _mlflow_runs_by_kind("gate")
    if proposals is None or verdicts is None:
        return counts
    counts["mlflow_reachable"] = 1
    from convfinqa.evalloop import ledger as mem

    # ── rewrites, from propose runs ──
    runs, client = proposals
    have = set(load("rewrites")["teacher_run_id"])
    rewrite_by_version: dict[str, str] = {
        str(r.new_version): str(r.rewrite_id) for r in load("rewrites").itertuples()
    }
    rows = []
    for run in sorted(runs, key=lambda r: r.info.start_time):
        run_id = run.info.run_id
        if run_id in have:
            counts["rewrites_existing"] += 1
            continue
        proposal = mem._artifact_json(client, run_id, "proposal.json") or {}
        diff = (mem._artifact_json(client, run_id, "prompt_diff.json") or {}).get(
            "diff", ""
        )
        params = run.data.params
        target = str(proposal.get("target") or params.get("target_agent", ""))
        base_v = str(proposal.get("base_version") or params.get("prompts_version", ""))
        new_v = str(proposal.get("new_version") or params.get("new_version", ""))
        before = ""
        try:
            import convfinqa.prompts as prompts_pkg

            before = prompts_pkg.load(base_v)[target]
        except Exception:  # noqa: BLE001 — a retired base version hashes as ""
            before = ""
        metrics = run.data.metrics
        row = rewrite_row(
            target=target,
            base_version=base_v,
            new_version=new_v,
            prompt_before=before,
            prompt_after=str(proposal.get("prompt") or ""),
            diff=str(diff or ""),
            rationale=str(proposal.get("rationale") or ""),
            edit_text=str(proposal.get("summary_of_changes") or ""),
            n_diagnoses=int(float(params.get("n_diagnoses", 0) or 0)),
            campaign=params.get("campaign"),
            label=params.get("experiment_label"),
            teacher_run_id=run_id,
            teacher_model=str(params.get("actor_model", "")),
            usage={
                "usage": {
                    "input_tokens": metrics.get("teacher_input_tokens", 0),
                    "output_tokens": metrics.get("teacher_output_tokens", 0),
                },
                "total_cost_usd": metrics.get("teacher_cost_usd", 0.0),
            },
            proposed_at=datetime.fromtimestamp(
                run.info.start_time / 1000, tz=timezone.utc
            ).isoformat(timespec="seconds"),
        )
        if not before:
            row["prompt_hash_before"] = ""
        if not proposal.get("prompt"):
            row["prompt_hash_after"] = ""
        row["prompt_chars_before"] = int(
            metrics.get("prompt_chars_before", len(before)) or 0
        )
        rows.append(row)
        rewrite_by_version.setdefault(new_v, str(row["rewrite_id"]))
    append("rewrites", rows)
    counts["rewrites"] += len(rows)

    # ── gates, from gate runs ──
    runs, client = verdicts
    have = set(load("gates")["gate_run_id"])
    rows = []
    for run in sorted(runs, key=lambda r: r.info.start_time):
        run_id = run.info.run_id
        if run_id in have:
            counts["gates_existing"] += 1
            continue
        params = run.data.params
        verdict = mem._artifact_json(client, run_id, "verdict.json") or {}
        flips = mem._artifact_json(client, run_id, "flips.json") or {}
        base_v = str(params.get("baseline_version", ""))
        cand_v = str(params.get("candidate_version", ""))
        split = str(params.get("evidence_split", "test"))
        stats = {
            "evidence_split": split,
            "n_compared": verdict.get("n_compared", run.data.metrics.get("n_compared")),
            "accuracy_delta": verdict.get(
                "accuracy_delta", run.data.metrics.get("accuracy_delta")
            ),
            "fail_to_pass": verdict.get(
                "fail_to_pass", run.data.metrics.get("fail_to_pass")
            ),
            "pass_to_fail": verdict.get(
                "pass_to_fail", run.data.metrics.get("pass_to_fail")
            ),
            "cluster_p_one_sided": verdict.get(
                "cluster_p_one_sided", run.data.metrics.get("cluster_p_one_sided")
            ),
            "delta_ci_lo": verdict.get(
                "delta_ci_lo", run.data.metrics.get("delta_ci_lo")
            ),
            "delta_ci_hi": verdict.get(
                "delta_ci_hi", run.data.metrics.get("delta_ci_hi")
            ),
        }
        base_csv = mem._run_csv_for(base_v, split, pred_dir)
        cand_csv = mem._run_csv_for(cand_v, split, pred_dir)
        attribution: AttributionOf | None = None
        panel_b: dict[str, float] = {}
        panel_c: dict[str, float] = {}
        if base_csv is not None and cand_csv is not None:
            from convfinqa.evalloop.gate import load_run_csv

            try:
                base_df, cand_df = load_run_csv(base_csv), load_run_csv(cand_csv)
                stats["baseline_accuracy"] = round(float(base_df["correct"].mean()), 6)
                stats["candidate_accuracy"] = round(float(cand_df["correct"].mean()), 6)
                attribution = attribution_from_frames(base_df, cand_df)
                panel_b = stage_scores.run_metrics(base_df)
                panel_c = stage_scores.run_metrics(cand_df)
            except Exception:  # noqa: BLE001 — evidence we cannot read stays blank
                attribution = None
        promoted = bool(
            verdict.get("promoted", run.data.tags.get("promoted") == "true")
        )
        rows.append(
            gate_row(
                stats,
                baseline_version=base_v,
                candidate_version=cand_v,
                promoted=promoted,
                reason=str(verdict.get("reason") or run.data.tags.get("reason", "")),
                flips=flips if attribution is not None else None,
                attribution_of_row=attribution,
                panel_baseline=panel_b,
                panel_candidate=panel_c,
                baseline_hash=bundle_hash(base_v),
                candidate_hash=bundle_hash(cand_v),
                baseline_eval_run_id=eval_run_ids(base_csv),
                candidate_eval_run_id=eval_run_ids(cand_csv),
                gate_run_id=run_id,
                rewrite_id=rewrite_by_version.get(cand_v, ""),
                campaign=params.get("campaign"),
                label=params.get("experiment_label"),
                # The baseline of a recorded gate was the champion it challenged.
                champion_after=cand_v if promoted else base_v,
                gated_at=datetime.fromtimestamp(
                    run.info.start_time / 1000, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            )
        )
    append("gates", rows)
    counts["gates"] += len(rows)
    return counts
