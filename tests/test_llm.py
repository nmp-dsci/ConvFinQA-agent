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
