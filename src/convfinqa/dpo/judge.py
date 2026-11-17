"""The trace judge (s39): can an LLM tell a gold-hinted trace from a real right one?

A two-alternative forced choice. The judge sees the conversation so far, the
question, and two traces that both reach the right answer; it is told that one
of them was written by a model that had been told the answer, and must say
which. It never sees gold or the note. Every pair is judged twice, the order
swapped, and a question scores 1 (built found both times), 0 (fooled both
times) or 0.5 (it picked the same slot twice: position, not content).

Three arms on the same questions:

- **T** test: a built trace (passed G1–G6) vs the real run-1 trace.
- **C0** null: the run-2 real trace, labelled "built", vs the run-1 real trace.
  Both are real, so this is the judge's chance level on this data, with any
  position or length bias it has.
- **C2** positive: the built trace with a leak sentence injected into its
  reasoning. A judge that cannot catch this has no power, and its failure to
  separate T says nothing.

Pass (s36 §05): C2 ≥ 90%, C0's slot-1 rate in 40–60%, and the upper end of
the 95% CI on T − C0 at most +10 points.

Runs on the Agent SDK (subscription) through `evalloop.sdk.run_structured`,
the traced chokepoint, at Haiku 5.5 / medium effort (owner, 2026-11-13).
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from convfinqa.dpo import builder

MODEL = "claude-haiku-5-5"
EFFORT = "medium"
ARMS = ("T", "C0", "C2")
LEAK_LINE = (
    "Since I already know the correct final answer is {gold}, I'll make sure the "
    "calculation lands on it."
)

SYSTEM = """You compare two reasoning traces written by AI agents answering the same \
question about a company's financial report. Both traces reach the same, correct answer \
using calculator tools.

One of the two agents was secretly told the correct final answer before it started and \
asked to hide that fact. The other worked it out with no help.

Decide which trace (A or B) came from the agent that was told the answer. Useful signs: \
reasoning that works backwards from a number, certainty that appears before any \
calculation, operands chosen to hit a target, checks against an answer that was never \
derived, or references to having been given the answer. Style differences alone are not \
evidence: both agents are the same model.

You must pick A or B. Give a confidence from 0.5 (a guess) to 1.0 (certain) and one or two \
sentences of reason."""


class Pick(BaseModel):
    pick: Literal["A", "B"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


def trace_text(completion: list[dict[str, Any]], *, inject: str | None = None) -> str:
    """A rendered turn (assistant/tool messages) as plain text for the judge."""
    lines: list[str] = []
    injected = inject is None
    for m in completion:
        if m["role"] == "assistant":
            content = str(m.get("content") or "").strip()
            if content:
                content = (
                    content.replace("<reasoning>", "")
                    .replace("</reasoning>", "")
                    .strip()
                )
                if not injected:
                    content = f"{inject}\n{content}"
                    injected = True
                lines.append(f"[thinking] {content}")
            for call in m.get("tool_calls") or []:
                fn = call["function"]
                label = (
                    "final answer" if fn["name"] == "StructuredOutput" else "tool call"
                )
                lines.append(f"[{label}] {fn['name']} {fn['arguments']}")
        elif m["role"] == "tool":
            lines.append(f"[tool result] {m.get('content')}")
    if not injected:
        lines.insert(0, f"[thinking] {inject}")
    return "\n".join(lines)


def _history(rid: str, k: int) -> str:
    d = builder._csv(builder.RUNS[0])
    g = d[(d.report_id == rid) & (d.turn_index < k)].sort_values("turn_index")
    if g.empty:
        return "(this is the first question)"
    return "\n".join(
        f"Q{r.turn_index}: {r.question}\nA: {r.pred_answer}" for r in g.itertuples()
    )


def view(item: dict[str, Any], a: str, b: str) -> str:
    """The judge's user message: history, question, the two traces."""
    return (
        f"Conversation so far:\n{_history(item['report_id'], item['k'])}\n\n"
        f"Question: {item['question']}\n\n"
        f"=== Trace A ===\n{a}\n\n=== Trace B ===\n{b}\n\n"
        "Which trace came from the agent that was told the answer?"
    )


def arm_traces(rec: dict[str, Any]) -> dict[str, tuple[str, str]] | None:
    """(designated "built", real) texts for each arm, or None if nothing passed the gates."""
    good = [t for t in rec["built"] if t.get("gates", {}).get("passed")]
    if not good:
        return None
    rid, k = rec["report_id"], rec["k"]
    convs = builder.conversations
    prefix1 = builder.tj.prefix(convs(builder.RUNS[0])[rid], k)
    prefix2 = builder.tj.prefix(convs(builder.RUNS[1])[rid], k)
    real = trace_text(
        builder.completion(rid, prefix1, builder.real_turn(builder.RUNS[0], rid, k))
    )
    real2 = trace_text(
        builder.completion(rid, prefix2, builder.real_turn(builder.RUNS[1], rid, k))
    )
    built_comp = builder.completion(rid, prefix1, good[0]["messages"])
    return {
        "T": (trace_text(built_comp), real),
        "C0": (real2, real),
        "C2": (trace_text(built_comp, inject=LEAK_LINE.format(gold=rec["gold"])), real),
    }


async def judge_one(
    item: dict[str, Any], arm: str, built: str, real: str, order: int
) -> dict[str, Any]:
    """One call: order 0 puts the designated-built trace in slot A, order 1 in slot B."""
    from convfinqa.evalloop.sdk import run_structured

    a, b = (built, real) if order == 0 else (real, built)
    parsed, usage = await run_structured(
        view(item, a, b),
        schema=Pick,
        system_prompt=SYSTEM,
        max_turns=2,
        refs=None,
        model=MODEL,
        effort=EFFORT,
    )
    built_slot = "A" if order == 0 else "B"
    return {
        "question_id": item["question_id"],
        "report_id": item["report_id"],
        "arm": arm,
        "order": order,
        "pick": parsed.pick,
        "hit": parsed.pick == built_slot,
        "confidence": parsed.confidence,
        "reason": parsed.reason,
        "len_built": len(built),
        "len_real": len(real),
        "duration_ms": usage.get("duration_ms"),
    }


async def judge_set(
    name: str, *, arms: tuple[str, ...] = ARMS, concurrency: int = 8, hint: str = "h1"
) -> Path:
    """Judge every gated question of a built set, both orders, every arm; resumable."""
    from convfinqa.evalloop.sdk import TeacherRateLimitError

    src = builder.ROOT / f"{name}_built{builder.tag(hint)}.jsonl"
    out = builder.ROOT / f"{name}_judge{builder.tag(hint)}.jsonl"
    done = set()
    if out.exists():
        done = {
            (r["question_id"], r["arm"], r["order"])
            for r in (json.loads(x) for x in out.open() if x.strip())
        }
    jobs: list[tuple[dict[str, Any], str, str, str, int]] = []
    unrenderable = 0
    for line in src.open():
        rec = json.loads(line)
        try:
            traces = arm_traces(rec)
        except builder.rows_mod.RowRejectedError:
            # a real trace (or its history) the student's wire cannot carry: the
            # same rows the DPO data would drop, so counted and left out
            unrenderable += 1
            continue
        if traces is None:
            continue
        for arm in arms:
            for order in (0, 1):
                if (rec["question_id"], arm, order) not in done:
                    jobs.append((rec, arm, *traces[arm], order))
    random.Random(2039).shuffle(jobs)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stop = asyncio.Event()

    async def one(job: tuple[dict[str, Any], str, str, str, int]) -> None:
        if stop.is_set():
            return
        async with sem:
            if stop.is_set():
                return
            try:
                row = await judge_one(*job)
            except TeacherRateLimitError as exc:
                print(f"rate limited, stopping: {exc}")
                stop.set()
                return
            except Exception as exc:  # noqa: BLE001 - recorded, the pass goes on
                print(f"{job[0]['question_id']} {job[1]} {job[4]}: {exc!r}"[:300])
                return
        async with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(jobs)} judge calls to make; {unrenderable} questions unrenderable")
    await asyncio.gather(*(one(j) for j in jobs))
    return out


def _boot(
    per_q: pd.DataFrame, col: str, *, n: int = 4000, seed: int = 2039
) -> tuple[float, float]:
    rng = random.Random(seed)
    convs = per_q.groupby("report_id")[col].agg(["sum", "count"])
    keys = list(convs.index)
    stats = []
    for _ in range(n):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        s = convs.loc[pick]
        stats.append(float(s["sum"].sum() / s["count"].sum()))
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def report(name: str, hint: str = "h1") -> dict[str, Any]:
    """Per-arm scores, position bias, T − C0 with a conversation bootstrap, and the verdict."""
    t = builder.tag(hint)
    built = [json.loads(x) for x in (builder.ROOT / f"{name}_built{t}.jsonl").open()]
    rows = pd.DataFrame(
        [json.loads(x) for x in (builder.ROOT / f"{name}_judge{t}.jsonl").open()]
    )
    tries = [t for r in built for t in r["built"]]
    gate_keys = ("G1_answer", "G2_executes", "G5_no_leak", "G6_clean", "passed")
    out: dict[str, Any] = {
        "set": name,
        "hint": hint,
        "questions": len(built),
        "questions_gated": sum(
            any(t["gates"].get("passed") for t in r["built"]) for r in built
        ),
        "tries": len(tries),
        "gate_pass_rate": {
            g: round(
                sum(bool(t["gates"].get(g)) for t in tries) / max(1, len(tries)), 3
            )
            for g in gate_keys
        },
        "by_kind_gated": {
            kind: f"{sum(any(t['gates'].get('passed') for t in r['built']) for r in built if r['kind'] == kind)}"
            f"/{sum(r['kind'] == kind for r in built)}"
            for kind in ("program", "number")
        },
        "judge_calls": len(rows),
    }
    complete = rows.groupby(["question_id", "arm"]).order.nunique().reset_index()
    full = complete[complete.order == 2][["question_id", "arm"]]
    r2 = rows.merge(full, on=["question_id", "arm"])
    per_q = r2.groupby(["question_id", "report_id", "arm"]).hit.mean().reset_index()
    arms: dict[str, Any] = {}
    for arm in ARMS:
        q = per_q[per_q.arm == arm]
        c = r2[r2.arm == arm]
        if q.empty:
            continue
        lo, hi = _boot(q, "hit")
        arms[arm] = {
            "n_questions": int(len(q)),
            "score": round(float(q.hit.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "both_right": int((q.hit == 1).sum()),
            "split": int((q.hit == 0.5).sum()),
            "both_wrong": int((q.hit == 0).sum()),
            "slot_A_rate": round(float((c.pick == "A").mean()), 4),
            "mean_confidence": round(float(c.confidence.mean()), 3),
        }
    out["arms"] = arms
    if "T" in arms and "C0" in arms:
        w = (
            per_q.pivot_table(
                index=["question_id", "report_id"], columns="arm", values="hit"
            )
            .dropna(subset=["T", "C0"])
            .reset_index()
        )
        w["d"] = w["T"] - w["C0"]
        lo, hi = _boot(w.assign(one=1.0), "d")
        diff = float(w.d.mean())
        out["T_minus_C0"] = {
            "n": int(len(w)),
            "delta": round(diff, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }
        c0_slot = arms["C0"]["slot_A_rate"]
        c2 = arms.get("C2", {}).get("score")
        out["verdict"] = {
            "C2_power_ok": c2 is not None and c2 >= 0.90,
            "C0_position_ok": 0.40 <= c0_slot <= 0.60,
            "T_indistinguishable": hi <= 0.10,
        }
        out["verdict"]["passed"] = all(out["verdict"].values())
    return out
