"""The open-agent smoke (s25): the committed set, and how a pass is compared."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from convfinqa.evalloop import smoke


def test_the_committed_smoke_set_is_a_subset_of_the_champions_draws() -> None:
    """Each id must be in its seed's draw, or `--report-ids-file` refuses it."""
    from convfinqa.evalloop.splits import draw_train, manifest_path
    from convfinqa.sft.splits import excluded_pages, page_of

    manifest = json.loads(smoke.manifest_path().read_text())
    assert manifest["baseline_runs"] == {
        str(k): v for k, v in smoke.CHAMPION_TRAIN_RUNS.items()
    }
    ids_by_seed = {int(k): v for k, v in manifest["ids_by_seed"].items()}
    assert sum(len(v) for v in ids_by_seed.values()) == smoke.SMOKE_SIZE
    bad = excluded_pages()
    for seed, ids in ids_by_seed.items():
        listed = smoke.ids_path(seed).read_text().split()
        assert listed == ids
        drawn, _ = draw_train(
            seed=seed, n_reports=100, path=manifest_path("eval_loop_v2")
        )
        assert set(ids) <= set(drawn), seed
        assert not {page_of(i) for i in ids} & bad
    assert smoke._covers(pd.DataFrame(manifest["conversations"]))


def test_features_read_what_a_conversation_asks() -> None:
    from convfinqa.data.loader import load_raw_dataset

    record = next(
        r
        for r in load_raw_dataset()["train"]
        if r["id"] == "Single_JKHY/2009/page_28.pdf-3"
    )
    features = smoke.conversation_features(record)
    assert features["n_turns"] == 4 and features["n_program"] == 2
    assert features["ops"] == "divide+subtract" and features["pct_answer"]
    assert not features["type2"]


def _csv(path: Path, turns: dict[str, list[bool]]) -> Path:
    rows = [
        {"report_id": rid, "turn_index": i, "correct": ok}
        for rid, oks in turns.items()
        for i, ok in enumerate(oks)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_compare_reads_a_full_pass_in_the_champions_early_stopped_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke, "_predictions_dir", lambda: tmp_path)
    monkeypatch.setattr(smoke, "manifest_path", lambda name="x": tmp_path / "m.json")
    monkeypatch.setattr(
        smoke, "CHAMPION_TRAIN_RUNS", {2027: "champ27", 2028: "champ28"}
    )
    (tmp_path / "m.json").write_text(
        json.dumps({"ids_by_seed": {"2027": ["A", "B"], "2028": ["C"]}})
    )
    # The champion stopped at its first wrong turn; "Z" is outside the smoke.
    _csv(
        tmp_path / "champ27.csv", {"A": [True, True], "B": [True, False], "Z": [False]}
    )
    _csv(tmp_path / "champ28.csv", {"C": [True, True, True]})
    # The candidate answered every turn.
    cand27 = _csv(tmp_path / "c27.csv", {"A": [True, False], "B": [True, True, True]})
    cand28 = _csv(tmp_path / "c28.csv", {"C": [False, True, True]})

    result = smoke.compare({2027: cand27, 2028: cand28})
    assert result["champion"] == {
        "conversations": 3,
        "right_to_end": 2,
        "answered_turns": 2 + 2 + 3,
        "answered_right": 2 + 1 + 3,
        "mean_right_before_error": round(6 / 3, 4),
    }
    # Truncated at its first wrong turn: A stops at q1, C at q0.
    assert result["candidate_early_stop"]["answered_turns"] == 2 + 3 + 1
    assert result["candidate_early_stop"]["answered_right"] == 1 + 3 + 0
    assert result["candidate_full"]["answered_turns"] == 2 + 3 + 3
    assert result["candidate_full"]["answered_right"] == 1 + 3 + 2
    assert (result["fixed"], result["broken"]) == (1, 2)  # fixed B, broke A and C
    assert [c["report_id"] for c in result["per_conversation"]] == ["A", "B", "C"]
