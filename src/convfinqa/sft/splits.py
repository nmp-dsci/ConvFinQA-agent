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

`sft_v2` (s29) keeps sft_v1's train, val and test and adds an aimed draw
(`aim`): the 600 conversations left in the pool that epoch 2 is expected to
miss most turns of, ranked from gold by its val pass. Val and test do not
move, so every v2 checkpoint is compared with epoch 2 on the same questions.
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
# s29 (N2): sft_v1's train plus an aimed draw from what the pool has left.
AIM_SIZE = 600
AIM_SOURCE = "open-evalloop-train100-sdk_v1·s1-pyd-student-qwen3-4b-sft1-e2-none-20261025_183905.csv"
SFT_V2_MANIFEST_PATH = EVAL_ROOT / "splits" / "sft_v2.json"
COLLECT_V2_PATH = EVAL_ROOT / "sft" / "v2" / "collect_ids.txt"

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


def load(name: str = "sft_v1") -> dict[str, Any]:
    """A committed SFT manifest: `sft_v1` or `sft_v2`."""
    paths = {"sft_v1": SFT_MANIFEST_PATH, "sft_v2": SFT_V2_MANIFEST_PATH}
    if name not in paths:
        raise ValueError(
            f"unknown SFT manifest {name!r}; expected one of {sorted(paths)}"
        )
    manifest: dict[str, Any] = json.loads(paths[name].read_text())
    return manifest


def aim_candidates() -> set[str]:
    """What an aimed draw may take: the pool, less sft_v1's whole seed-2027 draw
    of 600 (kept or not), val, test and the holdout, and every page they touch."""
    from convfinqa.evalloop.splits import _pool_frame

    v1 = load("sft_v1")
    sealed = sealed_ids()
    val = set(v1["splits"]["val"])
    taken = set(train_draw()) | val | sealed["test"] | sealed["holdout"]
    bad = excluded_pages() | {page_of(r) for r in val}
    pool = _pool_frame()
    return {
        str(r) for r in pool["report_id"] if r not in taken and page_of(r) not in bad
    }


def build_manifest_v2(val_csv: Path, *, n: int = AIM_SIZE) -> dict[str, Any]:
    """`sft_v2`: sft_v1's train plus the `n` conversations epoch 2 is expected to
    miss most on (`aim`), with val and test unchanged and the leak checks asserted."""
    from convfinqa.sft import aim

    v1 = load("sft_v1")
    rates = aim.miss_rates(val_csv)
    candidates = aim_candidates()
    picked = aim.top(candidates, rates, n)
    aimed = [r for r, _ in picked]
    train = [*v1["splits"]["train"], *aimed]
    val = list(v1["splits"]["val"])
    bad = excluded_pages()
    if {page_of(r) for r in aimed} & (bad | {page_of(r) for r in val}):
        raise AssertionError(
            "an aimed conversation shares a page with test, holdout or val"
        )
    if set(aimed) & set(train_draw()):
        raise AssertionError("an aimed conversation is already in sft_v1's draw")
    if len(set(train)) != len(train):
        raise AssertionError("sft_v2 train lists a conversation twice")
    return {
        "name": "sft_v2",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "parent": "sft_v1",
        "unit": "conversation",
        "page_rule": v1["page_rule"],
        "gate_manifest": GATE_MANIFEST,
        "holdout_manifest": HOLDOUT_MANIFEST,
        "train_draw": v1["train_draw"],
        "val_draw": v1["val_draw"],
        "aim": {
            **rates,
            "n": n,
            "n_candidates": len(candidates),
            "ids": [
                {"report_id": r, "expected_misses": round(s, 4)} for r, s in picked
            ],
        },
        "splits": {"train": train, "val": val, "test": list(v1["splits"]["test"])},
        "stats": {
            "train_v1": len(v1["splits"]["train"]),
            "train_aimed": len(aimed),
            "train": len(train),
            "val": len(val),
            "test": len(v1["splits"]["test"]),
        },
    }


def write_manifest_v2(val_csv: Path, *, force: bool = False) -> Path:
    """Write `sft_v2.json` and the aimed ids the teacher collects on."""
    if SFT_V2_MANIFEST_PATH.exists() and not force:
        raise FileExistsError(
            f"{SFT_V2_MANIFEST_PATH} exists; it is the record v2 is built on. "
            "Pass force=True only before anything has run against it."
        )
    manifest = build_manifest_v2(val_csv)
    SFT_V2_MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n")
    COLLECT_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLLECT_V2_PATH.write_text(
        "\n".join(i["report_id"] for i in manifest["aim"]["ids"]) + "\n"
    )
    return SFT_V2_MANIFEST_PATH
