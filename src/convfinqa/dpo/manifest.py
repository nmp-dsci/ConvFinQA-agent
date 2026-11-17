"""The `dpo_v1` manifest: the fresh pool, in a seeded order, with an eval slice.

The fresh pool is what `sft_v2` left: every conversation an aimed draw could
take (`sft.splits.aim_candidates` — no page shared with test, the v1 holdout or
val, and none of sft_v1's seed-2027 draw) less the 600 sft_v2 aimed and
collected on. 1,489 conversations, 4,394 questions, none ever answered by the
student. Val and test are not in it and never will be.

Order is a seeded shuffle so that the first 300 conversations sampled are a
fair sample of the rest (P3 projects cost and yield from them). 5% of the
conversations, chosen by *page*, are held back as fixed eval pairs, so no
report page is in both the DPO train and eval slices.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from convfinqa.config import EVAL_ROOT

NAME = "dpo_v1"
PATH = EVAL_ROOT / "splits" / f"{NAME}.json"
ORDER_SEED = 2033
EVAL_SHARE = 0.05


def fresh_pool() -> list[str]:
    """The pool sft_v2 left, sorted: never collected on, page-clean."""
    from convfinqa.sft import splits

    v2 = splits.load("sft_v2")
    aimed = {i["report_id"] for i in v2["aim"]["ids"]}
    return sorted(splits.aim_candidates() - aimed)


def build() -> dict[str, Any]:
    """The manifest, with the leak checks asserted before it returns."""
    from convfinqa.data.loader import training_data
    from convfinqa.sft import splits

    pool = fresh_pool()
    order = list(pool)
    random.Random(ORDER_SEED).shuffle(order)

    pages = sorted({splits.page_of(r) for r in pool})
    random.Random(ORDER_SEED + 1).shuffle(pages)
    target = round(len(pool) * EVAL_SHARE)
    by_page: dict[str, list[str]] = {}
    for r in pool:
        by_page.setdefault(splits.page_of(r), []).append(r)
    eval_ids: list[str] = []
    for page in pages:
        if len(eval_ids) >= target:
            break
        eval_ids += by_page[page]
    eval_set = set(eval_ids)
    train = [r for r in order if r not in eval_set]
    evals = [r for r in order if r in eval_set]

    v1 = splits.load("sft_v1")
    val_pages = {splits.page_of(r) for r in v1["splits"]["val"]}
    bad = splits.excluded_pages() | val_pages
    for name, ids in (("train", train), ("eval", evals)):
        if {splits.page_of(r) for r in ids} & bad:
            raise AssertionError(
                f"dpo_v1 {name} shares a page with test, holdout or val"
            )
    if {splits.page_of(r) for r in train} & {splits.page_of(r) for r in evals}:
        raise AssertionError("dpo_v1 train and eval share a page")
    if set(pool) & set(splits.load("sft_v2")["splits"]["train"]):
        raise AssertionError("dpo_v1 overlaps sft_v2's train")

    qa = training_data()
    n_q = qa[qa["report_id"].isin(pool)].groupby("report_id").size()
    return {
        "name": NAME,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "parent": "sft_v2",
        "unit": "conversation",
        "page_rule": v1["page_rule"],
        "source": "sft.splits.aim_candidates() minus sft_v2's aimed ids",
        "order_seed": ORDER_SEED,
        "eval_share": EVAL_SHARE,
        "splits": {"train": train, "eval": evals},
        "order": order,
        "stats": {
            "conversations": len(pool),
            "questions": int(n_q.sum()),
            "train": len(train),
            "eval": len(evals),
            "eval_questions": int(n_q[evals].sum()),
            "pages": len(pages),
        },
    }


def write(*, force: bool = False) -> Path:
    """Write `dpo_v1.json`; refuses to overwrite a manifest tries refer to."""
    if PATH.exists() and not force:
        raise FileExistsError(
            f"{PATH} exists; it is the record every sampled try refers to. "
            "Pass force=True only before anything has been sampled against it."
        )
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    return PATH


def load() -> dict[str, Any]:
    """The committed `dpo_v1` manifest."""
    manifest: dict[str, Any] = json.loads(PATH.read_text())
    return manifest
