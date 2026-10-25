"""The `sft_v1` manifest: whole conversations, split by report page.

The prompt loop splits by conversation id, which is enough for scoring prompts
but not for training weights. ConvFinQA builds a Type II conversation by
joining two Type I conversations from the same report page, so a test
conversation's page siblings carry its document and some of its questions.
Every set here is therefore cut by *page*: an id is normalised by dropping the
``Single_``/``Double_`` prefix and the ``-N`` suffix, and no page appears in two
sets.

Train is the prompt loop's own seeded draw, ``draw_train(seed=2027, n=600)``,
which holds the committed seed-2027 draw of 100 id for id. The draw is not
page-aware (and it reaches ``eval_loop_v1``'s sealed holdout too), so the page
filter runs *after* the draw: it removes conversations without moving a single
drawn id, which keeps the seed reproducible.
"""

from __future__ import annotations

import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from convfinqa.config import EVAL_ROOT

GATE_MANIFEST = "eval_loop_v2"
HOLDOUT_MANIFEST = "eval_loop_v1"
TRAIN_SEED = 2027
TRAIN_DRAW = 600
VAL_SEED = 2031
VAL_SIZE = 100
SFT_MANIFEST_PATH = EVAL_ROOT / "splits" / "sft_v1.json"
TRAIN_IDS_PATH = EVAL_ROOT / "sft" / "v1" / "train_ids.txt"

_PREFIX = re.compile(r"^(Single|Double)_")
_SUFFIX = re.compile(r"-\d+$")


def page_of(report_id: str) -> str:
    """The report page a conversation is about: `Double_X/2015/page_74.pdf` and
    `Single_X/2015/page_74.pdf-2` are the same page."""
    return _SUFFIX.sub("", _PREFIX.sub("", str(report_id)))


def _manifest(name: str) -> dict[str, Any]:
    from convfinqa.evalloop.splits import load_manifest, manifest_path

    return load_manifest(manifest_path(name))


def sealed_ids() -> dict[str, set[str]]:
    """The conversations no SFT set may touch: the gate's test and v1's holdout."""
    return {
        "test": set(_manifest(GATE_MANIFEST)["splits"]["test"]),
        "holdout": set(_manifest(HOLDOUT_MANIFEST)["splits"]["holdout"]),
    }


def excluded_pages() -> set[str]:
    """Every page that holds a test or sealed-holdout conversation."""
    return {page_of(r) for ids in sealed_ids().values() for r in ids}


def train_draw(seed: int = TRAIN_SEED, n: int = TRAIN_DRAW) -> list[str]:
    """The prompt loop's seeded draw against the gate manifest, in draw order."""
    from convfinqa.evalloop.splits import draw_train, manifest_path

    ids, _ = draw_train(seed=seed, n_reports=n, path=manifest_path(GATE_MANIFEST))
    return ids


def train_ids(seed: int = TRAIN_SEED, n: int = TRAIN_DRAW) -> list[str]:
    """The draw with every conversation on an excluded page removed, in draw order."""
    bad = excluded_pages()
    return [r for r in train_draw(seed, n) if page_of(r) not in bad]


def val_ids(
    train: list[str], *, seed: int = VAL_SEED, size: int = VAL_SIZE
) -> list[str]:
    """`size` conversations sharing no page with test, the holdout or the train draw.

    Drawn from `draw_train`'s own pool (which already leaves out the 260
    conversations GEPA/s7 saw), stratified on Type II the way the loop's draws
    are, so val looks like train.
    """
    from convfinqa.evalloop.splits import _pool_frame, _stratified_draw

    sealed = sealed_ids()
    taken = set(train_draw()) | sealed["test"] | sealed["holdout"]
    used_pages = excluded_pages() | {page_of(r) for r in train}
    pool = _pool_frame()
    pool = pool[~pool["report_id"].isin(taken)]
    pool = pool[~pool["report_id"].map(page_of).isin(used_pages)]
    return _stratified_draw(pool, size, random.Random(seed), {True: 0, False: 0})


def build_manifest() -> dict[str, Any]:
    """The `sft_v1` manifest, with the leak checks asserted before it returns."""
    sealed = sealed_ids()
    drawn = train_draw()
    train = train_ids()
    val = val_ids(train)
    bad = excluded_pages()
    test = sorted(sealed["test"])
    sets = {"train": train, "val": val, "test": test}
    pages = {name: {page_of(r) for r in ids} for name, ids in sets.items()}
    for a in ("train", "val"):
        if pages[a] & bad:
            raise AssertionError(f"{a} shares a page with test or the sealed holdout")
    if pages["train"] & pages["val"]:
        raise AssertionError("train and val share a page")
    return {
        "name": "sft_v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "unit": "conversation",
        "page_rule": "strip ^(Single|Double)_ and -N$; no page in two sets",
        "gate_manifest": GATE_MANIFEST,
        "holdout_manifest": HOLDOUT_MANIFEST,
        "train_draw": {"seed": TRAIN_SEED, "n_reports": TRAIN_DRAW},
        "val_draw": {"seed": VAL_SEED, "n_reports": VAL_SIZE},
        "splits": {"train": train, "val": val, "test": test},
        "stats": {
            "train_drawn": len(drawn),
            "train_kept": len(train),
            "train_dropped_test_page": sum(
                page_of(r) in {page_of(t) for t in sealed["test"]} for r in drawn
            ),
            "train_dropped_holdout_page": sum(
                page_of(r) in {page_of(h) for h in sealed["holdout"]}
                and page_of(r) not in {page_of(t) for t in sealed["test"]}
                for r in drawn
            ),
            "val": len(val),
            "test": len(test),
            "excluded_pages": len(bad),
        },
    }


def write_manifest(*, force: bool = False) -> Path:
    """Write `sft_v1.json` and the train id list the Sonnet 5.5 pass runs on."""
    if SFT_MANIFEST_PATH.exists() and not force:
        raise FileExistsError(
            f"{SFT_MANIFEST_PATH} exists; it is the record every SFT run used. "
            "Pass force=True only before anything has run against it."
        )
    manifest = build_manifest()
    SFT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SFT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n")
    TRAIN_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_IDS_PATH.write_text("\n".join(manifest["splits"]["train"]) + "\n")
    return SFT_MANIFEST_PATH


def load() -> dict[str, Any]:
    """The committed `sft_v1` manifest."""
    manifest: dict[str, Any] = json.loads(SFT_MANIFEST_PATH.read_text())
    return manifest
