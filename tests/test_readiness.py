"""The readiness scorecard: the committed record is current, and drift is caught."""

# ruff: noqa: D103

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from convfinqa.evalloop import readiness
from convfinqa.serving import app as api_app


def test_committed_scorecard_has_no_problems() -> None:
    assert readiness.READINESS_PATH.exists()
    assert readiness.problems() == []


def test_committed_scorecard_shape() -> None:
    data = readiness.load_readiness()
    assert [r["ref"] for r in data["rows"]] == list(readiness.EXPECTED_REFS)
    assert data["score"] == "6 / 9"
    assert set(data["statuses"]) == set(readiness.STATUSES)
    assert all(r["order"] == i for i, r in enumerate(data["rows"], start=1))
    assert all(r["question"] and r["label"] and r["how"] for r in data["rows"])


def _tampered() -> dict[str, Any]:
    return copy.deepcopy(readiness.load_readiness())


def test_missing_proof_path_is_reported() -> None:
    data = _tampered()
    data["rows"][0]["proof"].append("src/convfinqa/does_not_exist.py")
    found = readiness.problems(data)
    assert len(found) == 1
    assert "R1" in found[0] and "does_not_exist.py" in found[0]


def test_unknown_status_is_reported() -> None:
    data = _tampered()
    data["rows"][2]["status"] = "done"
    found = readiness.problems(data)
    # A shipped row turned into a non-status also changes the shipped count,
    # so the score is reported alongside the bad vocabulary.
    assert any("R3" in p and "'done'" in p for p in found)
    assert any("score" in p for p in found)


def test_wrong_score_is_reported() -> None:
    data = _tampered()
    data["score"] = "9 / 9"
    found = readiness.problems(data)
    assert len(found) == 1
    assert "score '9 / 9'" in found[0] and "'6 / 9'" in found[0]


def test_unknown_route_and_empty_how_are_reported() -> None:
    data = _tampered()
    data["rows"][1]["app"].append("/admin/nowhere")
    data["rows"][8]["how"] = ""
    found = readiness.problems(data)
    assert any("R2" in p and "/admin/nowhere" in p for p in found)
    assert any("R9" in p and "`how`" in p for p in found)


def test_rows_out_of_order_are_reported() -> None:
    data = _tampered()
    data["rows"].reverse()
    found = readiness.problems(data)
    assert any("in order" in p for p in found)


def test_missing_file_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(readiness, "READINESS_PATH", tmp_path / "readiness.json")
    assert readiness.problems() == [f"{tmp_path / 'readiness.json'} is missing"]


def test_story_check_folds_readiness_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from convfinqa.evalloop import story_check

    broken = _tampered()
    broken["score"] = "0 / 9"
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(broken))
    monkeypatch.setattr(readiness, "READINESS_PATH", path)
    assert any(p.startswith("readiness.json: score") for p in story_check.problems())


def test_readiness_route_serves_the_file(demo_mode: None) -> None:
    with TestClient(api_app.create_app(eviction_interval_seconds=3600)) as client:
        response = client.get("/eval/readiness")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["score"] == "6 / 9"
    assert len(body["rows"]) == 9
    assert body == readiness.load_readiness()


def test_readiness_route_404s_without_the_file(
    demo_mode: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(readiness, "READINESS_PATH", tmp_path / "readiness.json")
    with TestClient(api_app.create_app(eviction_interval_seconds=3600)) as client:
        response = client.get("/eval/readiness")
    assert response.status_code == 404
    assert "readiness" in response.json()["detail"]
