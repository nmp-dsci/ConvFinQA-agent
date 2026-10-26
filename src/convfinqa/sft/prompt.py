"""The student's prompt (s28, D5): the sdk prompt verbatim, plus one paragraph.

The teacher, llm_react_v1 on DeepSeek, reasons in hidden thinking before each
calculator call. The student (Qwen3-4B-Instruct-2507) does not think, so the
training rows put that same reasoning in a visible ``<reasoning>`` block ahead
of the call it leads to, and this paragraph tells the student about the block.
The base model and the tuned one get the same prompt, so the comparison
between them is fair.

The text is derived, never copied: ``student_prompt("sdk_v1")`` is the
committed ``SDK_PROMPT`` with ``STUDENT_FORMAT`` appended, and its hash goes
into the data card and onto every student pass.
"""

from __future__ import annotations

REASONING_OPEN = "<reasoning>"
REASONING_CLOSE = "</reasoning>"

STUDENT_FORMAT = f"""

## Show your reasoning before you act

Before a tool call, write your reasoning for that step inside \
{REASONING_OPEN}...{REASONING_CLOSE} in the same message, then make the call. Say \
what the question asks, which figures it needs from the document or from \
earlier answers, and the plan. A step that needs no thought, such as passing a \
calculator's result on, can be the call alone. The answer itself is always \
the StructuredOutput call, never text."""


def student_prompt(version: str = "sdk_v1") -> str:
    """The sdk prompt `version` with the reasoning paragraph appended."""
    from convfinqa import prompts

    return prompts.load_sdk(version).rstrip() + STUDENT_FORMAT


def reasoning_block(text: str) -> str:
    """DeepSeek's thinking as the visible block a training row carries."""
    return f"{REASONING_OPEN}\n{text.strip()}\n{REASONING_CLOSE}"
