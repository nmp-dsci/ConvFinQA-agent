"""Shared test fixtures.

Two properties this file is responsible for keeping true:

* **No test ever reaches the network.** Agents are always driven by pydantic-ai's
  `TestModel` or an httpx `MockTransport`; the placeholder key below exists only
  so a model object can be *constructed*, never so a call can succeed.
* **No test touches real project state.** The trace store is redirected to a
  temp directory, so running the suite never writes `.traces/` into the repo.

The placeholder key is scoped to the fixture rather than exported in CI, which is
what lets `test_config.py` verify the opposite property — that the package boots,
and read-only routes serve, with no key at all.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

TEST_API_KEY = "test-placeholder-not-a-real-key"


@pytest.fixture(autouse=True)
def _isolated_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Give each test its own trace store, no demo mode, and a constructible key."""
    from convfinqa.config import settings

    monkeypatch.setattr(settings, "demo_mode", False, raising=False)
    monkeypatch.setattr(settings, "trace_capture_enabled", False, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", TEST_API_KEY)

    from pydantic import SecretStr

    monkeypatch.setattr(
        settings, "deepseek_api_key", SecretStr(TEST_API_KEY), raising=False
    )

    import convfinqa.llm as llm
    from convfinqa.backends import pydantic as backend
    from convfinqa.serving.demo_pack import store as pack_store
    from convfinqa.tracking import traces

    monkeypatch.setattr(traces, "default_db_path", lambda: tmp_path / "traces.db")
    llm.reset_provider()
    backend.reset_default_agents()
    traces.reset_store()
    pack_store.reset_cache()
    yield
    llm.reset_provider()
    backend.reset_default_agents()
    traces.reset_store()
    pack_store.reset_cache()


@pytest.fixture
def demo_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run the enclosed test as the public demo: flag on, no key present."""
    from convfinqa.config import settings

    monkeypatch.setattr(settings, "demo_mode", True, raising=False)
    monkeypatch.setattr(settings, "deepseek_api_key", None, raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    import convfinqa.llm as llm
    from convfinqa.backends import pydantic as backend

    llm.reset_provider()
    backend.reset_default_agents()
    yield
    llm.reset_provider()
    backend.reset_default_agents()


@pytest.fixture
def pipeline_agents() -> dict[str, object]:
    """The four pipeline agents, built once for override-based tests."""
    from convfinqa.backends.pydantic import default_agents

    return dict(default_agents())


os.environ.setdefault("DEEPSEEK_API_KEY", TEST_API_KEY)
