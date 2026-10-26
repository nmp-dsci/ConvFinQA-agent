# SFT dataset v2 — llm_react_v1 → Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554

Built 2026-10-25T11:32:18Z by `convfinqa-sft build` (s28). Unit: the whole conversation.
Row shape A: one row per whole conversation, loss on every assistant message; plus prefix rows (a conversation's turns before its first wrong one).

- **Rows:** 916 conversations, 3486 questions, 308 Type II;
  278 are prefix rows (729 questions). Manifest `sft_v2`.
- **Tokens:** 7,369,883 in all; median 7,991, min 5,463, max 12,573 per row
  (the student's tokenizer and chat template).
- **Prompt:** student_v1 = sdk_v1 + STUDENT_FORMAT; system sha256 `819137b929da`, tools sha256
  `02fce9ed893e` — read off the student's own request, not rebuilt.
- **Reasoning:** DeepSeek V4.1 Flash's own thinking, verbatim, as a visible `<reasoning>`
  block before the call it leads to. 288 text-answer detours collapsed (N1).
- **train.jsonl sha256:** `fc63c1d8e64bfc0812be515057424c1bd8709a1823e0258ac501e2e4533db8f8` — the trainer refuses any other file.

## Sources

- `open-evalloop-train552-sdk_v1·s1-pyd-deepseek-flash-high-20261025_142402`
- `open-evalloop-train600-sdk_v1·s1-pyd-deepseek-flash-high-20261025_215639`

## Dropped (514), by the first rule each failed

A drop with a prefix row still counts here: the conversation was not kept whole.

| rule | conversations |
|---|---|
| F1 | 455 |
| F4 | 33 |
| F5 | 3 |
| F6 | 1 |
| F7 | 18 |
| N1 | 4 |

Test 100 (`eval_loop_v2` test) is the holdout: nothing here comes from it or its pages.
