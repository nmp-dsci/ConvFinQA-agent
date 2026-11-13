"""The gold-hinted trace writer (s36 revision 1, s39): DeepSeek Flash told the answer.

For a question the student never gets right and the teacher (llm_react_v1) also
misses, there is no right trace to use as a DPO chosen. The builder asks the
teacher again, on the teacher's own history, with the gold answer appended to
the question as a note it must not mention. Before any built trace is used, it
has to be shown indistinguishable from a real right trace (`dpo.judge`).

**The validation pool** is every question of the 438 backfill conversations the
teacher answered right in *both* of its unhinted runs, with every earlier turn
right in both. For those, run 1's turn is the real trace a built one is compared
with, and run 2's is a second real trace: the judge's chance level (control C0)
is measured on two real traces of the same question. Cut by conversation into
`dev` (iterate on) and `sealed` (decide on, once).

**Gates**, on every built trace (deterministic):

- G1 answer: right under the scorer and the strict check (`pairs.close`).
- G2 executes: a program turn's answer is the result of its own last calculator
  call, so the number came from a tool, not from the note.
- G5 no leak: no reference to the note in the reasoning, the visible text or
  the sub-questions.
- G6 clean: no harness retry, no error, and the conversation renders through
  `sft.rows.to_row`.

(G3 grounded and G4 gold program from the s36 plan are not yet implemented;
s39 says so.)
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.config import EVAL_ROOT
from convfinqa.dpo import pairs
from convfinqa.sft import rows as rows_mod
from convfinqa.sft import trajectories as tj

ROOT = EVAL_ROOT / "dpo" / "v1" / "builder"
RUNS = (
    "open-evalloop-train438-sdk_v1·s1-pyd-deepseek-flash-high-20261113_202234",
    "open-evalloop-train438-sdk_v1·s1-pyd-deepseek-flash-high-20261113_212051",
)
TARGETS = EVAL_ROOT / "dpo" / "v1" / "all_wrong_tries8.json"
MODEL = "deepseek-flash"
EFFORT = "high"
HINTS = {
    # iteration 1 (dev, 2026-11-13): judge spots 73% vs 55% chance; tells were
    # "Target: X", "expected X", "Yes! Confirms", checking against the number
    "h1": (
        "\n\n(Note for this question only: the correct final answer is {gold}. "
        "Work it out from the report with the tools exactly as you normally would, "
        "and write your reasoning as though you had not been given this note. "
        "Never mention the note.)"
    ),
    # iteration 2: forbid the verification behaviour the judge read as the tell
    "h2": (
        "\n\n(Private note for this question only: the correct final answer is "
        "{gold}. Use it only to avoid a wrong turn. Derive the answer from the "
        "report and the tools in your usual order and style, as if you did not "
        "know it: do not state the result before you have computed or found it, "
        "do not check your work against it, and do not write words such as "
        "target, expected, confirm or confirms. Never mention this note.)"
    ),
}
HINT = HINTS["h1"]
LEAK = re.compile(
    r"\b(note|hint|told|given (the )?answer|correct (final )?answer|expected answer|"
    r"target answer|answer (is|was) given|we know the answer|should (be|equal|come out))\b",
    re.IGNORECASE,
)


def _csv(run: str) -> pd.DataFrame:
    d = pd.read_csv(
        EVAL_ROOT / "predictions" / "evalloop" / f"{run}.csv",
        dtype=str,
        keep_default_na=False,
    )
    d["turn_index"] = d.turn_index.astype(int)
    return d


def pool() -> list[dict[str, Any]]:
    """Questions both unhinted teacher runs got right (strictly), on right histories."""
    a, b = (_csv(r) for r in RUNS)
    m = a.merge(b, on=["report_id", "turn_index"], suffixes=("", "_2"))
    m = m.sort_values(["report_id", "turn_index"])

    def strict(pred: str, gold: str) -> bool:
        g = pairs.grade(pred, gold)
        return bool(g["right"] and pairs.close(pred, gold))

    m["ok1"] = [strict(p, g) for p, g in zip(m.pred_answer, m.gold_answer, strict=True)]
    m["ok2"] = [
        strict(p, g) for p, g in zip(m.pred_answer_2, m.gold_answer, strict=True)
    ]
    hist1 = m.groupby("report_id").ok1.transform(
        lambda s: s.shift(fill_value=True).cummin()
    )
    hist2 = m.groupby("report_id").ok2.transform(
        lambda s: s.shift(fill_value=True).cummin()
    )
    keep = m[m.ok1 & m.ok2 & hist1 & hist2]
    targets = {(t["rid"], int(t["k"])) for t in json.loads(TARGETS.read_text())}
    return [
        {
            "report_id": r.report_id,
            "k": int(r.turn_index),
            "question_id": r.question_id,
            "question": r.question,
            "gold": r.gold_answer,
            "kind": "program" if r.gold_turn_type.lower() == "program" else "number",
            "student_all_wrong": (r.report_id, int(r.turn_index)) in targets,
            "real_answer": r.pred_answer,
            "real2_answer": r.pred_answer_2,
        }
        for r in keep.itertuples()
    ]


def split(
    items: list[dict[str, Any]],
    *,
    n_dev: int = 120,
    n_sealed: int = 300,
    seed: int = 2039,
) -> dict[str, list[dict[str, Any]]]:
    """Dev and sealed sets cut by conversation, so no conversation is in both."""
    convs: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        convs.setdefault(it["report_id"], []).append(it)
    order = sorted(convs)
    random.Random(seed).shuffle(order)
    out: dict[str, list[dict[str, Any]]] = {"dev": [], "sealed": []}
    for rid in order:
        dest = "dev" if len(out["dev"]) < n_dev else "sealed"
        if dest == "sealed" and len(out["sealed"]) >= n_sealed:
            break
        out[dest].extend(convs[rid])
    return out


def write_sets(force: bool = False) -> Path:
    """Commit the dev/sealed manifest; refuse to recut one that exists."""
    path = ROOT / "sets.json"
    if path.exists() and not force:
        return path
    ROOT.mkdir(parents=True, exist_ok=True)
    items = pool()
    sets = split(items)
    path.write_text(
        json.dumps(
            {
                "runs": list(RUNS),
                "pool": len(items),
                "seed": 2039,
                "unit": "conversation",
                "sets": sets,
            },
            indent=1,
        )
        + "\n"
    )
    return path


def tag(hint: str) -> str:
    """File suffix for a hint variant; h1 keeps the original names."""
    return "" if hint == "h1" else f"_{hint}"


def prepare() -> None:
    """Render the student's wire once, outside any event loop (`rows.wire` runs its own)."""
    rows_mod.wire("sdk_v1")


def load_sets() -> dict[str, list[dict[str, Any]]]:
    """The committed dev and sealed sets."""
    return dict(json.loads((ROOT / "sets.json").read_text())["sets"])


_CONVS: dict[str, dict[str, list[dict[str, Any]]]] = {}


def conversations(run: str) -> dict[str, list[dict[str, Any]]]:
    """One unhinted run's trajectories by conversation, loaded once."""
    if run not in _CONVS:
        _CONVS[run] = tj.load(tj.fetch(run))
    return _CONVS[run]


def real_turn(run: str, rid: str, k: int) -> list[dict[str, Any]]:
    """Question k's messages from an unhinted run, with its prompt text kept."""
    msgs = conversations(run)[rid]
    out: list[dict[str, Any]] = []
    seen = -1
    for message in msgs:
        prompts = (
            tj.parts(message, "user-prompt") if message.get("kind") == "request" else []
        )
        if prompts:
            seen += 1
            if seen == k:
                opening = copy.deepcopy(message)
                opening["parts"] = [copy.deepcopy(prompts[0])]
                out.append(opening)
                continue
            if seen > k:
                break
        if seen == k:
            out.append(copy.deepcopy(message))
    return out


def completion(
    rid: str, prefix: list[dict[str, Any]], turn: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The turn as the student's wire renders it: assistant and tool messages."""
    row, _ = rows_mod.to_row(rid, prefix + turn, version="sdk_v1")
    msgs: list[dict[str, Any]] = list(row["messages"])
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    comp = msgs[last_user + 1 :]
    while comp and comp[-1]["role"] == "tool":
        comp = comp[:-1]
    return comp


def _texts(turn: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for m in turn:
        if m.get("kind") != "response":
            continue
        for p in m.get("parts") or []:
            kind = p.get("part_kind")
            if kind in ("thinking", "text"):
                out.append(str(p.get("content") or ""))
            elif kind == "tool-call" and p.get("tool_name") == rows_mod.OUTPUT_TOOL:
                out.append(json.dumps(tj.tool_args(p)))
    return "\n".join(out)


def gates(
    item: dict[str, Any], built: dict[str, Any], prefix: list[dict[str, Any]]
) -> dict[str, Any]:
    """G1, G2, G5, G6 on one built trace; `passed` when all hold."""
    g: dict[str, Any] = {}
    answer = str(built.get("answer") or "")
    graded = pairs.grade(answer, item["gold"])
    g["G1_answer"] = bool(graded["right"] and pairs.close(answer, item["gold"]))
    results = [
        e.get("result")
        for e in built.get("trajectory") or []
        if e.get("event") == "tool_return" and e.get("result") is not None
    ]
    if item["kind"] == "program":
        g["G2_executes"] = bool(results) and pairs.close(answer, results[-1])
    else:
        g["G2_executes"] = True
    leak = LEAK.search(_texts(built["messages"]))
    g["G5_no_leak"] = leak is None
    g["G5_match"] = leak.group(0) if leak else ""
    clean = not built.get("error") and not built.get("harness_retry")
    if clean:
        try:
            completion(item["report_id"], prefix, built["messages"])
        except rows_mod.RowRejectedError as exc:
            clean = False
            g["G6_rule"] = exc.rule
    g["G6_clean"] = clean
    g["passed"] = all(
        g[x] for x in ("G1_answer", "G2_executes", "G5_no_leak", "G6_clean")
    )
    return g


async def build_one(
    item: dict[str, Any], *, tries: int = 1, hint: str = "h1"
) -> dict[str, Any]:
    """`tries` gold-hinted teacher answers to one question, on run 1's history."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    from convfinqa import llm
    from convfinqa.config import settings
    from convfinqa.dpo.sample import ask_group

    rid, k = item["report_id"], item["k"]
    msgs = conversations(RUNS[0])[rid]
    prefix = tj.prefix(msgs, k)
    prompt = tj.turns(msgs)[k]["prompt"]
    system = next(m["instructions"] for m in msgs if m.get("instructions"))
    history = list(ModelMessagesTypeAdapter.validate_python(prefix))
    group = await ask_group(
        system_prompt=system,
        make_model=lambda: llm.open_agent_model(MODEL, EFFORT),
        model_name=MODEL,
        prompt=prompt + HINTS[hint].format(gold=item["gold"]),
        history=history,
        tries=tries,
        max_turns=settings.sdk_max_turns,
    )
    built = []
    for t in group:
        t.pop("_history", None)
        if "messages" in t:
            # the stored prompt is blanked; put back the question without the note,
            # which is what a chosen made from this trace would be asked
            for m in t["messages"]:
                for p in m.get("parts") or []:
                    if p.get("part_kind") == "user-prompt" and p.get("content") == "":
                        p["content"] = prompt
                        break
                else:
                    continue
                break
        t["gates"] = gates(item, t, prefix) if "messages" in t else {"passed": False}
        built.append(t)
    return {**item, "hint": hint, "built": built}


async def build_set(
    name: str, *, tries: int = 1, concurrency: int = 16, hint: str = "h1"
) -> Path:
    """Build every question of one set; append to `<set>_built.jsonl`, resumable."""
    out = ROOT / f"{name}_built{tag(hint)}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["question_id"] for line in out.open() if line.strip()}
    todo = [it for it in load_sets()[name] if it["question_id"] not in done]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(it: dict[str, Any]) -> None:
        async with sem:
            rec = await build_one(it, tries=tries, hint=hint)
        async with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    await asyncio.gather(*(one(it) for it in todo))
    return out
