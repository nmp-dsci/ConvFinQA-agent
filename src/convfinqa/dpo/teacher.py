"""Teacher backfill (s37): llm_react_v1's right turns as chosens for the student's 0/8 questions.

The student got these questions wrong in all 8 tries, so `pairs.selection`
has no chosen for them. The teacher (DeepSeek Flash, the SFT teacher) answered
the same conversations; where its turn at that question is right under the
strict check, and every earlier turn of its own conversation was right too, its
turn becomes a chosen:

- **The prompt is the student's.** The teacher's turn is spliced into the
  student's restored history (`sample.restored_messages`) as though it were one
  more try, so chosen and rejected share one prompt byte for byte, exactly as in
  v1's pairs. The teacher's own history is never used; it only has to have been
  right, so the turn's references to earlier answers hold.
- **The rejected sides are the student's own wrong tries**, already sampled.
- **Rendering is v1's** (`render._side`, `render.check`, `completion_spans`):
  the teacher's thinking becomes the visible reasoning block, as in SFT.

A teacher turn the harness retried cannot be rendered (F5) and is skipped and
counted, never rendered with the retry dropped.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.dpo import pairs
from convfinqa.dpo.render import _side, check, completion_spans
from convfinqa.dpo.sample import load_records
from convfinqa.sft import rows as rows_mod
from convfinqa.sft import trajectories as tj

SOURCE = "teacher"


def teacher_turn(messages: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Question `k`'s messages from a teacher conversation, shaped like a student try.

    A try's messages open with the request that asks the question, its
    user-prompt blanked (`restored_messages` refills it with the student's own
    prompt text) and its system prompt likewise replaced. Parts the request
    carries for the previous turn (a folded closing tool return) are dropped.
    """
    seen = -1
    out: list[dict[str, Any]] = []
    for message in messages:
        prompts = (
            tj.parts(message, "user-prompt") if message.get("kind") == "request" else []
        )
        if prompts:
            seen += 1
            if seen == k:
                opening = copy.deepcopy(message)
                opening["parts"] = [{**copy.deepcopy(prompts[0]), "content": ""}]
                out.append(opening)
                continue
            if seen > k:
                break
        if seen == k:
            out.append(copy.deepcopy(message))
    if not out:
        raise ValueError(f"the conversation has no question {k}")
    return out


def _answers(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df["turn_index"] = df.turn_index.astype(int)
    return df


def candidates(
    passes: list[tuple[Path, Path]], targets: list[dict[str, Any]]
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], Counter[str]]:
    """Right teacher turns per target question, over every (csv, trajectories) pass."""
    want = {(t["rid"], int(t["k"])): t for t in targets}
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    counts: Counter[str] = Counter()
    for n, (csv, traj) in enumerate(passes):
        df = _answers(csv)
        convs = tj.load(traj)
        for (rid, k), t in want.items():
            g = df[df.report_id == rid].sort_values("turn_index")
            row = g[g.turn_index == k]
            if row.empty or rid not in convs:
                counts["missing"] += 1
                continue
            answer = row.iloc[0].pred_answer
            graded = pairs.grade(answer, t["gold"])
            if not (graded["right"] and pairs.close(answer, t["gold"])):
                counts["teacher wrong"] += 1
                continue
            if (g[g.turn_index < k].correct != "True").any():
                counts["teacher history wrong"] += 1
                continue
            out.setdefault((rid, k), []).append(
                {
                    "i": f"{SOURCE}{n}",
                    "answer": answer,
                    "messages": teacher_turn(convs[rid], k),
                    "prompt_suffix": "",
                    "pass": csv.name,
                    **graded,
                }
            )
            counts["teacher right"] += 1
    return out, counts


def build(
    passes: list[tuple[Path, Path]],
    targets_path: Path,
    tries_path: Path,
    split_of: dict[str, str],
    out: Path,
    *,
    cap: int = 4,
    seed: int = 2036,
) -> dict[str, Any]:
    """Render teacher-chosen pairs for the 0/8 questions; write rows, return the card."""
    from convfinqa.sft.prompt import student_prompt

    system = student_prompt("sdk_v1")
    targets = json.loads(targets_path.read_text())
    records = load_records(tries_path)
    found, counts = candidates(passes, targets)
    toks: Counter[str] = Counter()
    by_q: Counter[str] = Counter()
    n_rows = 0
    with out.open("w") as fh:
        for (rid, k), chosens in sorted(found.items()):
            rec = records[rid]
            qs = rec["questions"]
            q = qs[k]
            wrong = [t for t in q["tries"] if not t.get("error") and not t.get("right")]
            rng = random.Random(f"{seed}:{q['question_id']}")
            # distinct wrong answers first, so the pairs cover the different mistakes
            rng.shuffle(wrong)
            seen: set[str] = set()
            distinct: list[dict[str, Any]] = []
            repeats: list[dict[str, Any]] = []
            for t in wrong:
                (repeats if str(t["answer"]) in seen else distinct).append(t)
                seen.add(str(t["answer"]))
            rejecteds = (distinct + repeats)[:cap]
            made = 0
            for j, rejected in enumerate(rejecteds):
                chosen = chosens[j % len(chosens)]
                try:
                    prompt_c, comp_c, tools_ = _side(rid, qs, k, chosen, system)
                    prompt_r, comp_r, _ = _side(rid, qs, k, rejected, system)
                except rows_mod.RowRejectedError as exc:
                    counts[f"skipped:{exc.rule}"] += 1
                    continue
                if prompt_c != prompt_r:
                    counts["skipped:prompts differ"] += 1
                    continue
                row: dict[str, Any] = {
                    "prompt": prompt_c,
                    "chosen": comp_c,
                    "rejected": comp_r,
                    "tools": tools_,
                    "meta": {
                        "report_id": rid,
                        "question_id": q["question_id"],
                        "k": k,
                        "split": split_of[rid],
                        "kind": "program"
                        if str(q.get("gold_turn_type")).lower() == "program"
                        else "number",
                        "gold": q["gold"],
                        "chosen_i": chosen["i"],
                        "rejected_i": rejected["i"],
                        "chosen_answer": chosen["answer"],
                        "rejected_answer": rejected["answer"],
                        "source": SOURCE,
                        "teacher_pass": chosen["pass"],
                    },
                }
                problems = check(row)
                if problems:
                    counts["failed check"] += 1
                    counts.update(f"problem:{x.split(':')[0]}" for x in problems)
                    continue
                sc = completion_spans(prompt_c, comp_c, tools_)
                sr = completion_spans(prompt_c, comp_r, tools_)
                row["meta"]["tokens"] = {
                    "prompt": sc["prompt_tokens"],
                    "chosen_student": sc["student_tokens"],
                    "chosen_tool": sc["tool_tokens"],
                    "rejected_student": sr["student_tokens"],
                    "rejected_tool": sr["tool_tokens"],
                }
                for key, v in row["meta"]["tokens"].items():
                    toks[key] += v
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                made += 1
                n_rows += 1
                counts[f"rendered:{split_of[rid]}"] += 1
            if made:
                by_q[q["question_id"]] = made
    n = max(1, n_rows)
    return {
        "targets": len(targets),
        "questions_with_teacher_chosen": len(found),
        "questions_with_pairs": len(by_q),
        "pairs": n_rows,
        "cap": cap,
        "passes": [c.name for c, _ in passes],
        **dict(counts),
        "mean_tokens": {k: round(v / n) for k, v in toks.items()},
    }
