"""Selected pairs → DPO rows (s33 P2): one shared prompt, a chosen and a rejected turn.

A row is TRL's conversational preference shape with tools::

    {"prompt": [system, user, assistant, tool, …, user],   # the history it was asked on
     "chosen": [assistant, tool, …, assistant],            # one question's turn
     "rejected": [assistant, tool, …, assistant],
     "tools": [...], "meta": {...}}

Both sides are rendered by `sft.rows.to_row` from the history the tries were
asked on (`sample.restored_messages`) plus the try itself, so a pair carries
the bytes the student was served, and the shared prompt is checked to be the
same for both rather than assumed.

The completion holds the turn's **tool returns** as well as the student's own
messages: a calculator return is what the environment said, not what the
policy chose. `completion_spans` records which tokens of each side are the
student's, so the trainer can score only those (P4); `check` asserts the
split holds.

A history turn the harness retried (a kept try with a refused call) cannot be
rendered by `to_row` (F5), so a pair asked on such a history is skipped and
counted, never rendered with the retry silently dropped.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from convfinqa.dpo.sample import load_records, open_text, restored_messages
from convfinqa.sft import rows as rows_mod
from convfinqa.sft import tokens

GEN = "<|im_start|>assistant\n"


def _split(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A rendered conversation cut after its last user message."""
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    return messages[: last_user + 1], messages[last_user + 1 :]


def _side(
    rid: str, qs: list[dict[str, Any]], k: int, t: dict[str, Any], system: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    msgs = restored_messages(rid, qs, k, system, last=t)
    row, _ = rows_mod.to_row(rid, msgs, version="sdk_v1")
    prompt, completion = _split(row["messages"])
    # the StructuredOutput call's own return ("Final result processed.") closes
    # the turn after the student has decided; it is not part of what is compared
    while completion and completion[-1]["role"] == "tool":
        completion = completion[:-1]
    return prompt, completion, row["tools"]


def _answer_of(completion: list[dict[str, Any]]) -> str:
    call = completion[-1]["tool_calls"][-1]["function"]
    return str(json.loads(call["arguments"]).get("answer"))


def completion_spans(
    prompt: list[dict[str, Any]],
    completion: list[dict[str, Any]],
    tools_: list[dict[str, Any]],
) -> dict[str, Any]:
    """Token counts of the prompt, and of the student's vs the tools' completion text.

    The template renders a run of consecutive tool messages as one block, closed
    only after the last of them, so the text is cut where the next message is
    not a tool: each piece is one assistant message or one run of tool returns.
    """
    base = tokens.render({"messages": prompt, "tools": tools_})
    student = tool = 0
    done = base
    n = len(completion)
    for i in range(n):
        if i + 1 < n and completion[i]["role"] == completion[i + 1]["role"] == "tool":
            continue
        upto = tokens.render(
            {"messages": prompt + completion[: i + 1], "tools": tools_}
        )
        if not upto.startswith(done):
            raise ValueError("the template does not render the completion as a suffix")
        piece = tokens.count_text(upto[len(done) :])
        if completion[i]["role"] == "assistant":
            student += piece
        else:
            tool += piece
        done = upto
    return {
        "prompt_tokens": tokens.count_text(base),
        "student_tokens": student,
        "tool_tokens": tool,
    }


def check(row: dict[str, Any]) -> list[str]:
    """What is wrong with a DPO row; empty when it is sound."""
    problems: list[str] = []
    tools_ = row["tools"]
    base = tokens.render({"messages": row["prompt"], "tools": tools_})
    for side in ("chosen", "rejected"):
        comp = row[side]
        if not comp or comp[0]["role"] != "assistant":
            problems.append(f"{side}: does not open with an assistant message")
            continue
        if comp[-1]["role"] != "assistant" or [
            c["function"]["name"] for c in comp[-1].get("tool_calls") or []
        ] != [rows_mod.OUTPUT_TOOL]:
            problems.append(f"{side}: does not end on one StructuredOutput call")
        if any(m["role"] in ("user", "system") for m in comp):
            problems.append(f"{side}: a user or system message inside the completion")
        full = tokens.render({"messages": row["prompt"] + comp, "tools": tools_})
        if not full.startswith(base + GEN):
            problems.append(f"{side}: the prompt is not a prefix of prompt + {side}")
    if row["chosen"] == row["rejected"]:
        problems.append("chosen and rejected are the same")
    meta = row["meta"]
    if row["chosen"] and _answer_of(row["chosen"]) != str(meta["chosen_answer"]):
        problems.append("the chosen's final answer is not the chosen try's")
    if row["rejected"] and _answer_of(row["rejected"]) != str(meta["rejected_answer"]):
        problems.append("the rejected's final answer is not the rejected try's")
    return problems


def render_all(tries_path: Path, pairs_path: Path, out: Path) -> dict[str, Any]:
    """Render every selected pair; write the rows and return what was skipped and why."""
    from convfinqa.sft.prompt import student_prompt

    system = student_prompt("sdk_v1")
    records = load_records(tries_path)
    with open_text(pairs_path) as fh:
        selected = [json.loads(line) for line in fh if line.strip()]
    counts: Counter[str] = Counter()
    toks: Counter[str] = Counter()
    longest = 0
    with out.open("w") as fh:
        for p in selected:
            rec = records[p["report_id"]]
            qs = rec["questions"]
            k = int(p["k"])
            by_i = {t["i"]: t for t in qs[k]["tries"]}
            try:
                prompt_c, chosen, tools_ = _side(
                    p["report_id"], qs, k, by_i[p["chosen_i"]], system
                )
                prompt_r, rejected, _ = _side(
                    p["report_id"], qs, k, by_i[p["rejected_i"]], system
                )
            except rows_mod.RowRejectedError as exc:
                counts[f"skipped:{exc.rule}"] += 1
                continue
            if prompt_c != prompt_r:
                counts["skipped:prompts differ"] += 1
                continue
            row: dict[str, Any] = {
                "prompt": prompt_c,
                "chosen": chosen,
                "rejected": rejected,
                "tools": tools_,
                "meta": {
                    k_: p[k_]
                    for k_ in (
                        "report_id",
                        "question_id",
                        "k",
                        "split",
                        "kind",
                        "gold",
                        "chosen_i",
                        "rejected_i",
                        "chosen_answer",
                        "rejected_answer",
                    )
                },
            }
            problems = check(row)
            if problems:
                counts["failed check"] += 1
                counts.update(f"problem:{x.split(':')[0]}" for x in problems)
                continue
            sc = completion_spans(prompt_c, chosen, tools_)
            sr = completion_spans(prompt_c, rejected, tools_)
            row["meta"]["tokens"] = {
                "prompt": sc["prompt_tokens"],
                "chosen_student": sc["student_tokens"],
                "chosen_tool": sc["tool_tokens"],
                "rejected_student": sr["student_tokens"],
                "rejected_tool": sr["tool_tokens"],
            }
            total = sc["prompt_tokens"] + max(
                sc["student_tokens"] + sc["tool_tokens"],
                sr["student_tokens"] + sr["tool_tokens"],
            )
            longest = max(longest, total)
            for key, v in row["meta"]["tokens"].items():
                toks[key] += v
            counts["rendered"] += 1
            counts[f"rendered:{p['split']}"] += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = max(1, counts["rendered"])
    return {
        "selected": len(selected),
        **dict(counts),
        "mean_tokens": {k: round(v / n) for k, v in toks.items()},
        "longest_row_tokens": longest,
    }
