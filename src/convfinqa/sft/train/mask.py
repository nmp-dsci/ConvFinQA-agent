"""The loss-mask check, copied from tau-telecom-sft (s28): the tokens with loss are exactly the
agent's turns. Here an agent turn may open with a visible `<reasoning>` block; it is the
message's content, so the same check covers it.

TRL's `assistant_only_loss` swaps in its training copy of the student's chat template, which
wraps each assistant turn in `{% generation %}` markers, and labels every other token -100.
This module reads the trainer's own mask back from a tokenized example, cuts the rendered row
into trained and untrained segments (the viewer's tokens lens and `mask.json`), and checks
them against the row: one trained span per agent turn, each holding that turn's text and tool
calls and ending the turn, and nothing trained that the agent did not write.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

END = "<|im_end|>"
HEADER = "<|im_start|>assistant\n"


def segments(
    tokenizer: Any, input_ids: Sequence[int], mask: Sequence[int]
) -> list[dict[str, Any]]:
    """The rendered row cut where the mask changes: [{text, trained, tokens}]."""
    out: list[dict[str, Any]] = []
    start = 0
    for i in range(1, len(input_ids) + 1):
        if i == len(input_ids) or bool(mask[i]) != bool(mask[start]):
            out.append(
                {
                    "text": tokenizer.decode(
                        list(input_ids[start:i]), skip_special_tokens=False
                    ),
                    "trained": bool(mask[start]),
                    "tokens": i - start,
                }
            )
            start = i
    return out


def tokenize(
    tokenizer: Any, row: dict[str, Any], chat_template: str | None
) -> dict[str, Any]:
    """The row as TRL tokenizes it under `assistant_only_loss` (input_ids, assistant_masks)."""
    enc = tokenizer.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        chat_template=chat_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    return {
        "input_ids": list(enc["input_ids"]),
        "assistant_masks": list(enc["assistant_masks"]),
    }


def _call_text(tc: dict[str, Any]) -> str:
    fn = tc.get("function", tc)
    args = fn["arguments"]
    return f'{{"name": "{fn["name"]}", "arguments": {args if isinstance(args, str) else json.dumps(args)}}}'


def check(row: dict[str, Any], segs: list[dict[str, Any]]) -> dict[str, Any]:
    """True only if every agent turn is trained, whole, and nothing else is."""
    problems: list[str] = []
    turns = [m for m in row["messages"] if m["role"] == "assistant"]
    trained = [s for s in segs if s["trained"]]
    if len(trained) != len(turns):
        problems.append(f"{len(trained)} trained spans for {len(turns)} agent turns")
    for k, (span, turn) in enumerate(zip(trained, turns, strict=False)):
        text = span["text"]
        if not text.rstrip("\n").endswith(END):
            problems.append(
                f"turn {k}: the trained span does not end the turn with {END}"
            )
        if "<|im_start|>" in text or "<tool_response>" in text:
            problems.append(f"turn {k}: a header or a tool result is trained")
        body = text.rstrip("\n").removesuffix(END)
        if turn.get("content") and turn["content"].strip() not in body:
            problems.append(f"turn {k}: the agent's text is not all trained")
        for tc in turn.get("tool_calls") or []:
            if _call_text(tc) not in body:
                problems.append(
                    f"turn {k}: tool call {tc['function']['name']} is not all trained"
                )
    for i, s in enumerate(segs):
        if s["trained"] and (i == 0 or not segs[i - 1]["text"].endswith(HEADER)):
            problems.append(
                f"segment {i}: a trained span does not follow an assistant header"
            )
    context = [
        m["content"] for m in row["messages"] if m["role"] in ("system", "user", "tool")
    ]
    for text in (s["text"] for s in trained):
        if any(len(c) > 24 and c in text for c in context):
            problems.append("a user turn, tool result or the system prompt is trained")
            break
    return {
        "ok": not problems,
        "assistant_turns": len(turns),
        "trained_turns": len(trained),
        "trained_tokens": sum(s["tokens"] for s in trained),
        "total_tokens": sum(s["tokens"] for s in segs),
        "problems": problems,
    }
