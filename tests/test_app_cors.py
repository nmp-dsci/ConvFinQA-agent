# ruff: noqa: D103

from __future__ import annotations

from fastapi.testclient import TestClient

from convfinqa.serving import app as api_app


def _client() -> TestClient:
    return TestClient(
        api_app.create_app(session_ttl_seconds=60, eviction_interval_seconds=3600)
    )


def test_cors_preflight_allows_vite_dev_origin() -> None:
    with _client() as client:
        response = client.options(
            "/sessions",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code in (200, 204)
        allow_origin = response.headers.get("access-control-allow-origin", "")
        assert allow_origin == "http://localhost:5173"


def test_cors_actual_request_carries_allow_origin() -> None:
    rid = api_app.REPORT_IDS[0]
    with _client() as client:
        response = client.get("/reports?limit=1", headers={"Origin": "http://localhost:5173"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
        # Sanity: returns expected payload
        assert isinstance(response.json(), list)
        # rid is read solely to ensure the test fixture is healthy
        assert rid in api_app.REPORT_IDS
