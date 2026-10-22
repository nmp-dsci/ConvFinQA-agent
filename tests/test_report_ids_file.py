"""`run --report-ids-file` narrows a seeded train draw; it never adds to one."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from convfinqa.evalloop.runner import narrow_draw

DRAW = {"manifest": "eval_loop_v2", "draw_seed": 2027, "n_reports": 4}
IDS = [
    "Single_A/2010/page_1.pdf-1",
    "Double_B/2011/page_2.pdf",
    "Single_C/2012/page_3.pdf-2",
    "Single_D/2013/page_4.pdf-1",
]


def test_keeps_the_listed_ids_in_draw_order(tmp_path: Path) -> None:
    f = tmp_path / "ids.txt"
    f.write_text(f"{IDS[3]}\n\n{IDS[1]}\n")
    kept, draw = narrow_draw(IDS, DRAW, f)
    assert kept == [IDS[1], IDS[3]]
    assert draw["draw_seed"] == 2027
    assert draw["n_reports_drawn"] == 4 and draw["n_reports"] == 2
    assert draw["report_ids_file"] == "ids.txt"
    assert draw["report_ids_file_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()


def test_refuses_an_id_the_draw_does_not_hold(tmp_path: Path) -> None:
    f = tmp_path / "ids.txt"
    f.write_text(f"{IDS[0]}\nSingle_Z/2020/page_9.pdf-1\n")
    with pytest.raises(ValueError, match="never adds to one"):
        narrow_draw(IDS, DRAW, f)


def test_refuses_an_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "ids.txt"
    f.write_text("\n\n")
    with pytest.raises(ValueError, match="lists no report ids"):
        narrow_draw(IDS, DRAW, f)


async def test_run_split_refuses_the_file_without_a_seed(tmp_path: Path) -> None:
    from convfinqa.evalloop.runner import run_split

    f = tmp_path / "ids.txt"
    f.write_text(IDS[0])
    with pytest.raises(ValueError, match="pass it with --train-seed"):
        await run_split("train", "sdk_v1", runtime="agent_sdk", report_ids_file=f)
