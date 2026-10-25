"""The SFT dataset (s28): gated whole conversations from llm_react_v1's passes.

Selection is by conversation, as the owner set it: a conversation is kept
whole or dropped whole, and every question of a kept one is trained. Each drop
is charged to the first rule it fails, in this order, and written to
``drops.jsonl`` with that rule (tau-telecom-sft's F-rules, rebuilt for
ConvFinQA and DeepSeek):

- F2 split: in ``sft_v1`` train, on no test, sealed-holdout or val page.
- F1 right: every question's answer is right.
- F3 complete: every question answered, no turn error, nothing unscored, the
  history holds one turn per question and each ends on one answer call.
- F4 stages: gold's checkpoints hold on every turn — the turn type is right,
  the plan executes to gold's own program value, and the calculator's result
  is gold's. (Operand recall is recorded, not gated: it reads a value carried
  from an earlier turn as missing — s23 measured that.)
- F5 tools: no tool call was refused and none is repeated within a turn.
- F6 first attempt: no question was re-asked with a correction.
- F7 reasoning: no reasoning block over `MAX_REASONING_TOKENS`.
- F8 length: the rendered row fits `MAX_ROW_TOKENS`, never truncated.
- N1, from `rows`: the text-answer detour collapses, or the row drops.

Writes ``evaluation/sft/<version>/train.jsonl`` with ``stats.json`` (counts,
lengths, every sha256 the trainer checks), ``drops.jsonl`` and ``card.md``.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from convfinqa.config import EVAL_ROOT

MAX_REASONING_TOKENS = 2048
MAX_ROW_TOKENS = 24576
SFT_ROOT = EVAL_ROOT / "sft"
CORRECTION_MARK = "Your last reply did not match the required output schema"
NO_ANSWER_MARK = "I could not produce a valid answer to that question."
RULES = ("F2", "F1", "F3", "F4", "F5", "F6", "F7", "F8", "N1")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _questions() -> dict[str, int]:
    from convfinqa.data.loader import load_raw_dataset

    return {
        str(r["id"]): len(r["dialogue"]["conv_questions"])
        for split in load_raw_dataset().values()
        for r in split
    }


def _bool(value: Any) -> bool | None:
    """A CSV cell as a verdict: True, False, or None when not applicable."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, str):
        return {"true": True, "false": False}.get(value.strip().lower())
    return bool(value)


class _GateError(Exception):
    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(detail)
        self.rule = rule
        self.detail = detail


def _check(
    report_id: str,
    rows: Any,
    messages: list[dict[str, Any]] | None,
    *,
    allowed: set[str],
    n_questions: int | None,
) -> None:
    """Raise `_GateError` with the first rule this conversation fails (F2–F6)."""
    from convfinqa.sft import trajectories as tj

    if report_id not in allowed:
        raise _GateError(
            "F2", "not in the sft_v1 train split (or on a test, holdout or val page)"
        )
    if not bool(rows["correct"].all()):
        wrong = [int(i) for i in rows.loc[~rows["correct"].astype(bool), "turn_index"]]
        raise _GateError("F1", f"wrong at q{wrong[0]}")
    if n_questions is None or len(rows) != n_questions:
        raise _GateError("F3", f"{len(rows)} of {n_questions} questions answered")
    if "error" in rows and rows["error"].notna().any():
        raise _GateError("F3", "a turn recorded an error")
    if "unscored" in rows and rows["unscored"].fillna(False).astype(bool).any():
        raise _GateError("F3", "a turn is unscored")
    if messages is None:
        raise _GateError("F3", "no message history for this conversation")
    turns = tj.turns(messages)
    if len(turns) != len(rows):
        raise _GateError(
            "F3", f"{len(turns)} turns in the history for {len(rows)} questions"
        )
    for _, row in rows.iterrows():
        q = int(row["turn_index"])
        if _bool(row.get("triage_turn_type_ok")) is False:
            raise _GateError("F4", f"q{q}: the turn type is not gold's")
        if _bool(row.get("preprocess_plan_ok")) is False:
            raise _GateError("F4", f"q{q}: the plan does not execute to gold's value")
        if _bool(row.get("calculator_ok")) is False:
            raise _GateError("F4", f"q{q}: the calculator's result is not gold's")
    for q, turn in enumerate(turns):
        if CORRECTION_MARK in turn["prompt"]:
            raise _GateError("F6", f"q{q} was re-asked with a correction")
        seen: set[tuple[str, str]] = set()
        for response in tj.responses(turn):
            texts = " ".join(
                str(p.get("content") or "") for p in tj.parts(response, "text")
            )
            if NO_ANSWER_MARK in texts:
                raise _GateError("F3", f"q{q} never produced a valid answer")
            for call in tj.parts(response, "tool-call"):
                key = (
                    str(call.get("tool_name")),
                    json.dumps(tj.tool_args(call), sort_keys=True),
                )
                if key in seen and key[0] != "StructuredOutput":
                    raise _GateError(
                        "F5", f"q{q}: {key[0]} repeated with the same arguments"
                    )
                seen.add(key)


def build(
    csvs: list[Path], *, version: str = "v1", prompt_version: str = "sdk_v1"
) -> dict[str, Any]:
    """Gate every conversation in `csvs`, write the dataset, return its stats."""
    import pandas as pd

    from convfinqa.sft import rows as rows_mod
    from convfinqa.sft import splits, tokens
    from convfinqa.sft import trajectories as tj

    manifest = splits.load()
    sealed_pages = splits.excluded_pages()
    val_pages = {splits.page_of(r) for r in manifest["splits"]["val"]}
    allowed = {
        r
        for r in manifest["splits"]["train"]
        if splits.page_of(r) not in sealed_pages | val_pages
    }
    n_q = _questions()
    out_dir = SFT_ROOT / version

    kept: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for csv in csvs:
        name = csv.stem
        frame = pd.read_csv(csv)
        if "split" in frame and set(frame["split"].dropna().astype(str)) - {"train"}:
            raise ValueError(f"{csv.name}: only train passes are training data")
        if "-test" in name or "evalloop-test" in name:
            raise ValueError(f"{csv.name}: a test pass is never training data")
        histories = tj.load(tj.path_for(name))
        sources.append(
            {
                "run_name": name,
                "csv_sha256": _sha(csv.read_bytes()),
                "trajectories_sha256": _sha(tj.path_for(name).read_bytes()),
            }
        )
        for report_id, group in frame.groupby("report_id", sort=False):
            rid = str(report_id)
            if rid in seen:
                continue
            group = group.sort_values("turn_index")
            try:
                _check(
                    rid,
                    group,
                    histories.get(rid),
                    allowed=allowed,
                    n_questions=n_q.get(rid),
                )
                row, meta = rows_mod.to_row(rid, histories[rid], version=prompt_version)
                thinking = [
                    tokens.count_text(str(p.get("content") or ""))
                    for m in histories[rid]
                    for p in tj.parts(m, "thinking")
                ]
                if thinking and max(thinking) > MAX_REASONING_TOKENS:
                    raise _GateError(
                        "F7", f"a reasoning block of {max(thinking)} tokens"
                    )
                n_tokens = tokens.count(row)
                if n_tokens > MAX_ROW_TOKENS:
                    raise _GateError("F8", f"{n_tokens} tokens")
            except (_GateError, rows_mod.RowRejectedError) as exc:
                drops.append(
                    {
                        "report_id": rid,
                        "run_name": name,
                        "rule": exc.rule,
                        "detail": exc.detail,
                    }
                )
                continue
            seen.add(rid)
            row["meta"].update(
                {
                    "run_name": name,
                    "tokens": n_tokens,
                    "type2": rid.startswith("Double_"),
                    "operand_recall": [
                        None if pd.isna(v) else float(v)
                        for v in group["retriever_operand_recall"]
                    ]
                    if "retriever_operand_recall" in group
                    else None,
                }
            )
            kept.append(row)

    train = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept).encode()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_bytes(train)
    (out_dir / "drops.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in drops)
    )
    wire = rows_mod.wire(prompt_version)
    lengths = sorted(int(r["meta"]["tokens"]) for r in kept)
    stats: dict[str, Any] = {
        "version": version,
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unit": "conversation",
        "row_shape": "A: one row per whole conversation, loss on every assistant message",
        "student": f"{tokens.STUDENT}@{tokens.REVISION}",
        "prompt": f"student_v1 = {prompt_version} + STUDENT_FORMAT",
        "system_sha256": _sha(wire["system"].encode()),
        "tools_sha256": _sha(json.dumps(wire["tools"], sort_keys=True).encode()),
        "train_sha256": _sha(train),
        "sources": sources,
        "n_rows": len(kept),
        "n_turns": sum(int(r["meta"]["n_turns"]) for r in kept),
        "n_type2": sum(bool(r["meta"]["type2"]) for r in kept),
        "n_detours_collapsed": sum(int(r["meta"]["n_detours"]) for r in kept),
        "n_dropped": len(drops),
        "drops_by_rule": {
            rule: n for rule in RULES if (n := Counter(d["rule"] for d in drops)[rule])
        },
        "tokens": {
            "total": sum(lengths),
            "median": int(statistics.median(lengths)) if lengths else 0,
            "max": lengths[-1] if lengths else 0,
            "min": lengths[0] if lengths else 0,
        },
        "limits": {
            "max_row_tokens": MAX_ROW_TOKENS,
            "max_reasoning_tokens": MAX_REASONING_TOKENS,
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out_dir / "card.md").write_text(card(stats))
    return stats


def card(stats: dict[str, Any]) -> str:
    """The data card: what the dataset is, where it came from, what was dropped."""
    drops = (
        "\n".join(f"| {k} | {v} |" for k, v in stats["drops_by_rule"].items())
        or "| — | 0 |"
    )
    sources = "\n".join(f"- `{s['run_name']}`" for s in stats["sources"])
    t = stats["tokens"]
    return f"""# SFT dataset {stats["version"]} — llm_react_v1 → {stats["student"]}

Built {stats["built_at"]} by `convfinqa-sft build` (s28). Unit: the whole conversation.
Row shape {stats["row_shape"]}.

- **Rows:** {stats["n_rows"]} conversations, {stats["n_turns"]} questions, {stats["n_type2"]} Type II.
- **Tokens:** {t["total"]:,} in all; median {t["median"]:,}, min {t["min"]:,}, max {t["max"]:,} per row
  (the student's tokenizer and chat template).
- **Prompt:** {stats["prompt"]}; system sha256 `{stats["system_sha256"][:12]}`, tools sha256
  `{stats["tools_sha256"][:12]}` — read off the student's own request, not rebuilt.
- **Reasoning:** DeepSeek V4.1 Flash's own thinking, verbatim, as a visible `<reasoning>`
  block before the call it leads to. {stats["n_detours_collapsed"]} text-answer detours collapsed (N1).
- **train.jsonl sha256:** `{stats["train_sha256"]}` — the trainer refuses any other file.

## Sources

{sources}

## Dropped ({stats["n_dropped"]}), by the first rule each failed

| rule | conversations |
|---|---|
{drops}

Test 100 (`eval_loop_v2` test) is the holdout: nothing here comes from it or its pages.
"""
