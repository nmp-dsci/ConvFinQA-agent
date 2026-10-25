# SFT dataset v1 — llm_react_v1 → Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554

Built 2026-10-25T03:51:16Z by `convfinqa-sft build` (s28). Unit: the whole conversation.
Row shape A: one row per whole conversation, loss on every assistant message.

- **Rows:** 334 conversations, 1187 questions, 94 Type II.
- **Tokens:** 2,657,000 in all; median 7,791, min 6,185, max 11,160 per row
  (the student's tokenizer and chat template).
- **Prompt:** student_v1 = sdk_v1 + STUDENT_FORMAT; system sha256 `819137b929da`, tools sha256
  `02fce9ed893e` — read off the student's own request, not rebuilt.
- **Reasoning:** DeepSeek V4.1 Flash's own thinking, verbatim, as a visible `<reasoning>`
  block before the call it leads to. 170 text-answer detours collapsed (N1).
- **train.jsonl sha256:** `79739f33ad4497330fd4c656dd7fc43481bd81ab279921898c1044a592c2bbd5` — the trainer refuses any other file.

## Sources

- `open-evalloop-train552-sdk_v1·s1-pyd-deepseek-flash-high-20261025_142402`

## Dropped (218), by the first rule each failed

| rule | conversations |
|---|---|
| F1 | 199 |
| F4 | 9 |
| F5 | 1 |
| F7 | 6 |
| N1 | 3 |

Test 100 (`eval_loop_v2` test) is the holdout: nothing here comes from it or its pages.
