"""The choke point's model settings.

One regression is pinned here and it is the one that took the whole live pipeline
down: DeepSeek v4 defaults to thinking mode, and a thinking-mode request rejects
the `tool_choice` pydantic-ai sends for every structured `output_type` with a
400. Every stage of this pipeline uses a structured output, so the first model
call of every turn failed — not degraded, failed.

The setting is asserted on the constructed model rather than on the constant, so
a refactor that keeps the constant and stops passing it fails here rather than in
production.
"""

from __future__ import annotations

import pytest

from convfinqa import llm
from convfinqa.llm import DISABLE_THINKING_BODY, get_model, model_settings


def test_thinking_is_disabled_on_every_model_this_module_builds() -> None:
    """Without this, `tool_choice` is rejected and no turn reaches stage two."""
    model = get_model()
    settings = model.settings
    assert settings is not None, "no ModelSettings attached to the model"
    assert settings.get("extra_body", {}).get("thinking") == {"type": "disabled"}


def test_the_max_model_gets_the_same_treatment() -> None:
    """The s7 agents use the pro model through the same constructor."""
    model = get_model("deepseek-v4-pro")
    assert model.settings is not None
    assert model.settings.get("extra_body", {}).get("thinking") == {"type": "disabled"}


def test_settings_are_not_shared_between_models() -> None:
    """A caller mutating one model's settings must not reach into another's."""
    first = model_settings()
    second = model_settings()
    assert first == second
    first["extra_body"]["thinking"] = {"type": "enabled"}
    assert second["extra_body"]["thinking"] == {"type": "disabled"}
    assert DISABLE_THINKING_BODY["thinking"] == {"type": "disabled"}


def test_subscription_env_blanks_the_api_key_rather_than_omitting_it() -> None:
    """The Agent SDK merges `env` over os.environ, so omission is not removal.

    `config.load_dotenv` puts every key in ~/.env into os.environ at import time,
    so ANTHROPIC_API_KEY is present in this process whether or not anyone
    exported it. If the teacher's child environment merely lacks the key, the
    merge puts it back and the CLI bills the API instead of the subscription —
    silently, with the only symptom being a "Credit balance is too low" error
    from an account that has an active subscription.
    """
    import os

    from convfinqa.llm import subscription_env

    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-should-not-reach-the-child"
    try:
        env = subscription_env()
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)

    assert env["ANTHROPIC_API_KEY"] == ""
    assert not any(k.startswith("CLAUDE_CODE_SESSION") for k in env)
    assert "CLAUDECODE" not in env


def test_api_env_keeps_the_key_and_strips_the_session_variables() -> None:
    """The qa_agent runtime bills the API by default, so the key must *stay*.

    Everything else `subscription_env` strips is stripped here too: a
    `CLAUDE_CODE_*` variable would make the child inherit the identity of the
    Claude Code session the eval is driven from, and a dotfile OAuth token
    would silently turn an API-billed pass into a subscription one.
    """
    import os

    from convfinqa.llm import api_env

    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-keep-me"
    os.environ["CLAUDE_CODE_SESSION_ID"] = "session-of-the-operator"
    os.environ["CLAUDECODE"] = "1"
    try:
        env = api_env()
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
        os.environ.pop("CLAUDECODE", None)

    assert env["ANTHROPIC_API_KEY"] == "sk-ant-keep-me"
    assert not any(k.startswith("CLAUDE_CODE_SESSION") for k in env)
    assert "CLAUDECODE" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_api_env_refuses_to_run_without_a_key() -> None:
    """No key means the CLI would fall back to its own login and bill that."""
    import os

    import pytest

    from convfinqa.llm import api_env

    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
            api_env()
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_sdk_child_env_selects_exactly_one_billing_path() -> None:
    """`api` keeps the key, `subscription` blanks it, anything else is refused."""
    import os

    import pytest

    from convfinqa.llm import sdk_child_env

    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-either-way"
    # The API path is refused unless re-opened deliberately, so this asserts the
    # *mapping* (api -> keeps the key) with the escape hatch set, and the refusal
    # itself is pinned by `test_api_billing_is_refused_by_default`.
    os.environ["SDK_ALLOW_API_BILLING"] = "1"
    try:
        assert sdk_child_env("api")["ANTHROPIC_API_KEY"] == "sk-ant-either-way"
        assert sdk_child_env("subscription")["ANTHROPIC_API_KEY"] == ""
        with pytest.raises(ValueError, match="billing"):
            sdk_child_env("free")
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("SDK_ALLOW_API_BILLING", None)


def test_api_billing_is_refused_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The qa_agent runtime runs on the subscription; the API path is shut.

    Pinned because the refusal is the only thing standing between a campaign and
    an unbudgeted bill, and because an exhausted key fails as *reply text* the
    scorer reads as wrong answers rather than as an error.
    """
    monkeypatch.delenv(llm.API_BILLING_ESCAPE_HATCH, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    with pytest.raises(RuntimeError, match="refused"):
        llm.sdk_child_env("api")
    monkeypatch.setattr(llm.settings, "sdk_billing", "api", raising=False)
    with pytest.raises(RuntimeError, match="refused"):
        llm.sdk_child_env()


def test_api_billing_escape_hatch_reopens_the_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit opt-in still works, so the refusal is a decision not a wall."""
    monkeypatch.setenv(llm.API_BILLING_ESCAPE_HATCH, "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    env = llm.sdk_child_env("api")
    assert env["ANTHROPIC_API_KEY"] == "sk-test-key"


def test_subscription_is_the_default_billing_path() -> None:
    """The setting itself defaults to the subscription, not merely the guard."""
    assert llm.settings.sdk_billing == "subscription"
