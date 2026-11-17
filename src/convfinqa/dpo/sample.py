"""Branch, sample, grade, pair, continue: one conversation at a time (s33 §04).

Each question is asked `tries` times on the same history through the checked
harness at the serving sampler — `open_agent._ask`, the function a served turn
goes through, with a fresh agent per try so tool trajectories and the H1–H3
retry state never mix. The tries are graded (`pairs.grade`), paired
(`pairs.select`), and one right try is kept as the history for the next
question (`pairs.keep`). A question with no right try stops the conversation.

Every try is stored, with the history it was asked on recoverable from the
record, so a later top-up adds tries to the same group instead of re-running
the conversation. What the record leaves out is only what can be rebuilt
exactly: the system prompt (pinned by hash), and turn 0's user message, which
is the whole report (`agent_sdk._first_message`, pinned by hash).

A provider that will not answer (a cold or stopped server) aborts the
conversation without writing it, so the next invocation picks it up; a
conversation is written whole or not at all, one JSON line, fsynced.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from convfinqa.config import EVAL_ROOT
from convfinqa.dpo import pairs

OUT_DIR = EVAL_ROOT / "dpo" / "v1"
TRIES_PATH = OUT_DIR / "tries.jsonl"
# s34: every conversation at 8 tries; the 4-try record above is topped up into it
TRIES8_PATH = OUT_DIR / "tries8.jsonl"
RUNS_PATH = OUT_DIR / "sample_runs.jsonl"
DEFAULT_MODEL = "student-qwen3-4b-sft2-e1"
HARNESS = "pydantic_ai_checked"
TRIES = 4
# Consecutive conversations a dead server may abort before the run stops.
MAX_ABORTS_IN_A_ROW = 6


def sha(text: str) -> str:
    """A short sha256 that pins text the record leaves out."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _stored_messages(new: list[Any], prompt: str) -> tuple[list[Any], str]:
    """The try's own messages as JSON, without the bytes the record pins.

    The first request's user prompt is the question's prompt (plus, after a
    schema failure, the correction): its content is blanked and the part past
    the prompt returned, so turn 0 does not store the whole report four times.
    Instructions are the system prompt on every request; they are blanked too.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    dumped: list[Any] = ModelMessagesTypeAdapter.dump_python(new, mode="json")
    suffix = ""
    blanked = False
    for msg in dumped:
        if msg.get("instructions"):
            msg["instructions"] = None
        if blanked or msg.get("kind") != "request":
            continue
        for part in msg.get("parts", []):
            if part.get("part_kind") == "user-prompt" and isinstance(
                part.get("content"), str
            ):
                content = part["content"]
                suffix = (
                    content[len(prompt) :] if content.startswith(prompt) else content
                )
                part["content"] = ""
                blanked = True
                break
    return dumped, suffix


async def ask_group(
    *,
    system_prompt: str,
    make_model: Callable[[], Any],
    model_name: str,
    prompt: str,
    history: list[Any],
    tries: int,
    max_turns: int,
    start_i: int = 0,
) -> list[dict[str, Any]]:
    """`tries` answers to one prompt on one history, concurrently.

    Each try is a dict with its reply, usage, stored messages and the live
    `all_messages` (key ``_history``, never written) a kept try continues on.
    Raises `SdkRateLimitError` when the server would not answer.
    """
    from convfinqa.backends import open_agent
    from convfinqa.backends.agent_sdk import SdkTurnError

    async def one(i: int) -> dict[str, Any]:
        trajectory: list[dict[str, Any]] = []
        checks: dict[str, Any] = {"retried": False}
        agent = open_agent.build_agent(
            system_prompt, make_model(), trajectory, checks=checks
        )
        t0 = time.perf_counter()
        try:
            result, usage, everything = await open_agent._ask(
                agent, prompt, history, model=model_name, max_turns=max_turns
            )
        except SdkTurnError as exc:
            return {
                "i": i,
                "error": str(exc)[:500],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        new = everything[len(history) :]
        stored, suffix = _stored_messages(new, prompt)
        return {
            "i": i,
            "answer": result.answer,
            "turn_type": result.turn_type,
            "program": result.program,
            "reply": result.model_dump(mode="json"),
            "trajectory": list(trajectory),
            "attempts": usage.get("attempts", 1),
            "harness_retry": checks.get("retried") or "",
            "prompt_suffix": suffix,
            "usage": {
                k: usage.get(k)
                for k in (
                    "num_turns",
                    "input_tokens",
                    "cache_read_input_tokens",
                    "output_tokens",
                    "latency_ms",
                )
            },
            "messages": stored,
            "_history": everything,
        }

    return list(
        await asyncio.gather(*(one(i) for i in range(start_i, start_i + tries)))
    )


def _prompt_for(report_id: str, k: int, question: str) -> str:
    from convfinqa.backends.agent_sdk import _first_message, _later_message
    from convfinqa.data.loader import _DOCS

    if k == 0:
        return _first_message(report_id, _DOCS[report_id], question)
    return _later_message(question)


def _grade_group(group: list[dict[str, Any]], gold: Any) -> None:
    for t in group:
        if not t.get("error") and "right" not in t:
            t.update(pairs.grade(t["answer"], gold))


def _question_record(
    k: int,
    q: dict[str, Any],
    prompt: str,
    group: list[dict[str, Any]],
    chosen_pairs: list[tuple[int, int]],
    kept: int | None,
    history_from: int | None,
) -> dict[str, Any]:
    return {
        "k": k,
        "question_id": q["question_id"],
        "question": q["question"],
        "gold": str(q["gold"]),
        "gold_turn_type": q.get("turn_type"),
        "prompt_sha": sha(prompt),
        "prompt": None if k == 0 else prompt,
        "history_from": history_from,
        "tries": [{key: v for key, v in t.items() if key != "_history"} for t in group],
        "pairs": [list(p) for p in chosen_pairs],
        "kept": kept,
    }


def _stop_reason(group: list[dict[str, Any]], k: int) -> str:
    graded = [t for t in group if not t.get("error")]
    return f"no right try at q{k}" if graded else f"every try dropped at q{k}"


async def _walk(
    report_id: str,
    questions: list[dict[str, Any]],
    *,
    start: int,
    history: list[Any],
    record_qs: list[dict[str, Any]],
    system_prompt: str,
    make_model: Callable[[], Any],
    model_name: str,
    tries: int,
    max_turns: int,
) -> str:
    """Sample questions[start:] on `history`, appending to `record_qs`; the stop."""
    for k in range(start, len(questions)):
        q = questions[k]
        prompt = _prompt_for(report_id, k, q["question"])
        group = await ask_group(
            system_prompt=system_prompt,
            make_model=make_model,
            model_name=model_name,
            prompt=prompt,
            history=history,
            tries=tries,
            max_turns=max_turns,
        )
        _grade_group(group, q["gold"])
        rng = random.Random(pairs.question_seed(q["question_id"]))
        chosen_pairs = pairs.select(group, rng)
        kept = pairs.keep(group, chosen_pairs, rng)
        record_qs.append(
            _question_record(
                k,
                q,
                prompt,
                group,
                chosen_pairs,
                kept,
                None if k == 0 else record_qs[-1]["kept"],
            )
        )
        if kept is None:
            return _stop_reason(group, k)
        history = group[kept]["_history"]
    return "completed"


async def sample_conversation(
    report_id: str,
    questions: list[dict[str, Any]],
    *,
    system_prompt: str,
    make_model: Callable[[], Any],
    model_name: str,
    tries: int = TRIES,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Walk one conversation; return its whole record (see the module docstring).

    `questions` are ``{"question_id", "question", "gold", "turn_type"}`` in order.
    """
    from convfinqa.config import settings

    started = datetime.now(UTC)
    t0 = time.perf_counter()
    record_qs: list[dict[str, Any]] = []
    stop = await _walk(
        report_id,
        questions,
        start=0,
        history=[],
        record_qs=record_qs,
        system_prompt=system_prompt,
        make_model=make_model,
        model_name=model_name,
        tries=tries,
        max_turns=max_turns or settings.sdk_max_turns,
    )
    return {
        "report_id": report_id,
        "started_at": started.isoformat(timespec="seconds"),
        "wall_s": round(time.perf_counter() - t0, 2),
        "n_questions": len(questions),
        "stop": stop,
        "questions": record_qs,
    }


def rebuild_history(
    report_id: str, record_qs: list[dict[str, Any]], upto: int, system_prompt: str
) -> list[Any]:
    """The live message history question `upto` was asked on, from the record.

    The kept try of each earlier question, its blanked first user prompt (the
    question's prompt, plus any suffix) and its instructions put back.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    dumped = restored_messages(report_id, record_qs, upto, system_prompt)
    return list(ModelMessagesTypeAdapter.validate_python(dumped))


def restored_messages(
    report_id: str,
    record_qs: list[dict[str, Any]],
    upto: int,
    system_prompt: str,
    last: dict[str, Any] | None = None,
) -> list[Any]:
    """`rebuild_history` as JSON; with `last`, that try of question `upto` appended.

    The kept try stands for each earlier question; `last` is any try of the
    question the history was asked for (a DPO pair's chosen or rejected).
    """
    import copy

    dumped: list[Any] = []
    for j in range(upto + (last is not None)):
        q = record_qs[j]
        t = (
            last
            if j == upto and last is not None
            else next(x for x in q["tries"] if x["i"] == q["kept"])
        )
        prompt = _prompt_for(report_id, j, q["question"])
        if sha(prompt) != q["prompt_sha"]:
            raise ValueError(f"{q['question_id']}: the prompt changed since sampling")
        msgs = copy.deepcopy(t["messages"])
        text = prompt + (t.get("prompt_suffix") or "")
        restored = False
        if j > 0 and msgs and msgs[0].get("kind") != "request":
            # Pydantic AI folded this prompt into the previous turn's closing
            # tool-return request, which the history ends on: put it back there.
            tail = dumped[-1]
            stamp = tail["parts"][-1].get("timestamp") if tail["parts"] else None
            tail["parts"].append(
                {"content": text, "timestamp": stamp, "part_kind": "user-prompt"}
            )
            restored = True
        for msg in msgs:
            if msg.get("kind") != "request":
                continue
            if "instructions" in msg:
                msg["instructions"] = system_prompt
            if restored:
                continue
            for part in msg.get("parts", []):
                if part.get("part_kind") == "user-prompt" and part.get("content") == "":
                    part["content"] = text
                    restored = True
                    break
        dumped += msgs
    return dumped


async def topup_conversation(
    old: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    system_prompt: str,
    make_model: Callable[[], Any],
    model_name: str,
    tries: int,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Bring a sampled conversation up to `tries` a question, on the same histories.

    Each recorded question gets the missing tries on the history it was asked
    on, rebuilt from the record. A question that already had a kept try keeps
    it, since every later question was asked on it. One that stopped the
    conversation may now have a right try; then the walk goes on from there at
    `tries` a question.
    """
    from convfinqa.config import settings

    turns = max_turns or settings.sdk_max_turns
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    record_qs: list[dict[str, Any]] = []
    stop = old["stop"]
    for k, oq in enumerate(old["questions"]):
        q = questions[k]
        assert q["question_id"] == oq["question_id"]
        prompt = _prompt_for(old["report_id"], k, q["question"])
        history = rebuild_history(old["report_id"], old["questions"], k, system_prompt)
        have = list(oq["tries"])
        extra = await ask_group(
            system_prompt=system_prompt,
            make_model=make_model,
            model_name=model_name,
            prompt=prompt,
            history=history,
            tries=max(0, tries - len(have)),
            max_turns=turns,
            start_i=len(have),
        )
        for t in extra:
            t["round"] = 2
        _grade_group(extra, q["gold"])
        group = have + extra
        rng = random.Random(pairs.question_seed(q["question_id"]))
        chosen_pairs = pairs.select(group, rng)
        kept = oq["kept"]
        if kept is None:
            kept = pairs.keep(group, chosen_pairs, rng)
        record_qs.append(
            _question_record(
                k, q, prompt, group, chosen_pairs, kept, oq.get("history_from")
            )
        )
        if oq["kept"] is None:
            if kept is None:
                stop = _stop_reason(group, k)
            else:  # the top-up found a right try: carry on past the old stop
                stop = await _walk(
                    old["report_id"],
                    questions,
                    start=k + 1,
                    history=group[kept]["_history"],
                    record_qs=record_qs,
                    system_prompt=system_prompt,
                    make_model=make_model,
                    model_name=model_name,
                    tries=tries,
                    max_turns=turns,
                )
            break
    return {
        "report_id": old["report_id"],
        "started_at": started.isoformat(timespec="seconds"),
        "wall_s": round(old.get("wall_s", 0) + time.perf_counter() - t0, 2),
        "topped_up_from": old.get("tries"),
        "n_questions": len(questions),
        "stop": stop,
        "questions": record_qs,
    }


def resolve(path: Path) -> Path:
    """The file a record path names: itself, or its committed ``.gz`` copy."""
    gz = path.with_name(path.name + ".gz")
    return gz if not path.exists() and gz.exists() else path


def open_text(path: Path) -> IO[str]:
    """A JSONL record for reading, gzipped (as committed) or not (as written)."""
    path = resolve(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open()


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    """A tries file by conversation id (``.jsonl`` or its ``.jsonl.gz``)."""
    if not resolve(path).exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open_text(path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["report_id"]] = rec
    return out


def done_ids(path: Path = TRIES_PATH) -> set[str]:
    """Conversations already on record, so a re-run skips them."""
    if not resolve(path).exists():
        return set()
    ids = set()
    with open_text(path) as fh:
        for line in fh:
            if line.strip():
                ids.add(json.loads(line)["report_id"])
    return ids


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def questions_for(report_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Each conversation's questions with gold, in order."""
    from convfinqa.data.loader import training_data

    qa = training_data()
    qa = qa[qa["report_id"].isin(report_ids)].sort_values(["report_id", "q_order"])
    out: dict[str, list[dict[str, Any]]] = {}
    for r, g in qa.groupby("report_id"):
        out[str(r)] = [
            {
                "question_id": row.question_id,
                "question": row.conv_questions,
                "gold": row.conv_answers,
                "turn_type": row.turn_type,
            }
            for row in g.itertuples()
        ]
    return out


async def run(
    *,
    upto: int,
    concurrency: int,
    model_name: str = DEFAULT_MODEL,
    tries: int = TRIES,
    out: Path = TRIES_PATH,
    runs_log: Path = RUNS_PATH,
    topup_from: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Sample manifest order until `upto` conversations are on record.

    With `topup_from`, a conversation already in that file is topped up to
    `tries` (`topup_conversation`) instead of sampled afresh.
    """
    from convfinqa.backends.agent_sdk import SdkRateLimitError
    from convfinqa.dpo import manifest
    from convfinqa.llm import STUDENT_SAMPLING, open_agent_model
    from convfinqa.sft.prompt import student_prompt
    from convfinqa.tracking.prompt_ledger import prompt_hash

    m = manifest.load()
    split_of = {r: s for s, ids in m["splits"].items() for r in ids}
    have = done_ids(out)
    todo = [r for r in m["order"] if r not in have][: max(0, upto - len(have))]
    if not todo:
        log(f"{len(have)} conversations on record; nothing to do below {upto}")
        return {"sampled": 0}
    qs = questions_for(todo)
    old = load_records(topup_from) if topup_from else {}
    system_prompt = student_prompt("sdk_v1")
    header = {
        "manifest": manifest.NAME,
        "model": model_name,
        "harness": HARNESS,
        "tries": tries,
        "sampler": {**STUDENT_SAMPLING, "seed": None},
        "system_prompt": f"student_v1@{prompt_hash(system_prompt)}",
    }

    def make_model() -> Any:
        return open_agent_model(model_name, "none")

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    state = {"done": 0, "aborted": 0, "aborts_in_a_row": 0, "stopped": False}
    totals = {"tries": 0, "right": 0, "pairs": 0, "groups": 0, "flip": 0}
    started = datetime.now(UTC)
    t0 = time.perf_counter()

    async def one(report_id: str) -> None:
        if state["stopped"]:
            return
        async with sem:
            if state["stopped"]:
                return
            try:
                if report_id in old:
                    rec = await topup_conversation(
                        old[report_id],
                        qs[report_id],
                        system_prompt=system_prompt,
                        make_model=make_model,
                        model_name=model_name,
                        tries=tries,
                    )
                else:
                    rec = await sample_conversation(
                        report_id,
                        qs[report_id],
                        system_prompt=system_prompt,
                        make_model=make_model,
                        model_name=model_name,
                        tries=tries,
                    )
            except SdkRateLimitError as exc:
                async with lock:
                    state["aborted"] += 1
                    state["aborts_in_a_row"] += 1
                    log(
                        f"  abort {report_id}: server unavailable ({exc.refusal[:120]})"
                    )
                    if state["aborts_in_a_row"] >= MAX_ABORTS_IN_A_ROW:
                        state["stopped"] = True
                        log("  stopping: the server would not answer")
                return
            except Exception as exc:  # one bad conversation must not end the run
                async with lock:
                    state["aborted"] += 1
                    log(f"  failed {report_id}: {type(exc).__name__}: {str(exc)[:200]}")
                return
            rec = {**header, "split": split_of.get(report_id), **rec}
            async with lock:
                _append(out, rec)
                state["done"] += 1
                state["aborts_in_a_row"] = 0
                for q in rec["questions"]:
                    graded = [t for t in q["tries"] if not t.get("error")]
                    rights = sum(bool(t.get("right")) for t in graded)
                    totals["groups"] += 1
                    totals["tries"] += len(q["tries"])
                    totals["right"] += rights
                    totals["pairs"] += len(q["pairs"])
                    totals["flip"] += int(0 < rights < len(graded))
                el = time.perf_counter() - t0
                if state["done"] % 10 == 0 or state["done"] == len(todo):
                    log(
                        f"  {len(have) + state['done']}/{upto} conversations · "
                        f"{totals['groups']} questions · right {totals['right']}/"
                        f"{totals['tries']} · flips {totals['flip']} · pairs "
                        f"{totals['pairs']} · {el / 60:.1f} min"
                    )

    n_top = sum(r in old for r in todo)
    log(
        f"sampling {len(todo)} conversations ({len(have)} on record, {n_top} to top "
        f"up) · {tries} tries · "
        f"{model_name} · concurrency {concurrency}"
    )
    await asyncio.gather(*(one(r) for r in todo))
    wall = round(time.perf_counter() - t0, 1)
    summary = {
        **header,
        "started_at": started.isoformat(timespec="seconds"),
        "wall_s": wall,
        "concurrency": concurrency,
        "upto": upto,
        "topup_from": str(topup_from) if topup_from else None,
        "sampled": state["done"],
        "aborted": state["aborted"],
        "stopped_early": state["stopped"],
        **{f"n_{k}": v for k, v in totals.items()},
    }
    _append(runs_log, summary)
    log(json.dumps(summary))
    return summary
