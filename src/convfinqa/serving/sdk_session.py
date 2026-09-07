"""One live Agent SDK session behind one chat session (s12).

The eval runner walks a whole conversation inside one ``async with
ClaudeSDKClient`` block (`backends.agent_sdk.run_conversation`). Serving cannot:
a conversation arrives one HTTP request at a time, so the client has to stay
open between turns and be closed when the session is deleted or evicted. That
lifecycle is the only thing this module adds — every step of a turn (the first
message with the report, the corrective retry, the capture) is the runtime's
own code, called here so a served turn and a scored turn are the same turn.
"""

from __future__ import annotations

from typing import Any

from convfinqa.backends import agent_sdk
from convfinqa.backends.agent_sdk import SdkTurnResult
from convfinqa.tracking import tracing


class SdkSession:
    """A persistent single-session runtime for one report."""

    def __init__(
        self,
        report_id: str,
        *,
        version: str | None = None,
        model: str | None = None,
    ) -> None:
        import convfinqa.prompts as prompts_pkg
        from convfinqa.config import settings
        from convfinqa.llm import sdk_model_name
        from convfinqa.tracking import registry

        resolved = version or registry.sdk_champion()
        if not resolved:
            raise agent_sdk.SdkRuntimeUnavailableError(
                "no sdk_champion is registered, so there is no single-session prompt to serve"
            )
        self.report_id = report_id
        self.version = resolved
        self.system_prompt = prompts_pkg.load_sdk(resolved)
        self.model = model or sdk_model_name()
        self.max_turns = settings.sdk_max_turns
        self.tokens_total = 0
        self._sdk: Any = None
        self._client: Any = None
        self._trajectory: list[dict[str, Any]] = []
        self.turns_asked = 0

    @property
    def open_(self) -> bool:
        """Whether the underlying client is connected."""
        return self._client is not None

    async def open(self) -> None:
        """Connect the client. Idempotent."""
        if self._client is not None:
            return
        from convfinqa.llm import pipeline_sdk_options

        sdk = agent_sdk._load_sdk()
        server = agent_sdk.build_calculator_server(sdk, self._trajectory)
        options = pipeline_sdk_options(
            system_prompt=self.system_prompt,
            mcp_server=server,
            allowed_tools=list(agent_sdk.SDK_ALLOWED_TOOLS),
            output_schema=SdkTurnResult.model_json_schema(),
            max_turns=self.max_turns,
            model=self.model,
        )
        client = agent_sdk.new_client(sdk, options)
        await client.connect()
        self._sdk, self._client = sdk, client

    async def ask(
        self, question: str, *, history_text: str
    ) -> tuple[SdkTurnResult, dict[str, Any], dict[str, Any]]:
        """Ask one question; return the parsed result, its capture and the usage.

        Raises the runtime's own errors (`SdkTurnError`, `SdkRateLimitError`);
        the caller turns those into the stream's `error` frame.
        """
        from convfinqa.data.loader import _DOCS
        from convfinqa.evalloop import prompt_refs

        await self.open()
        assert self._client is not None and self._sdk is not None
        self._trajectory.clear()
        prompt = (
            agent_sdk._first_message(self.report_id, _DOCS[self.report_id], question)
            if self.turns_asked == 0
            else agent_sdk._later_message(question)
        )
        with tracing.span(
            f"q{self.turns_asked}: {question[:60]}",
            attributes={
                "report_id": self.report_id,
                "turn_index": self.turns_asked,
                "question": question,
                "runtime": agent_sdk.RUNTIME,
            },
        ) as qspan:
            result, usage = await agent_sdk._ask(
                self._client,
                self._sdk,
                prompt,
                refs=prompt_refs.sdk_prompt_ref(self.version, self.system_prompt),
                system_prompt=self.system_prompt,
                model=self.model,
                max_turns=self.max_turns,
            )
            self.turns_asked += 1
            self.tokens_total += agent_sdk._tokens_used(usage)
            capture: dict[str, Any] = {"history_text": history_text}
            capture.update(
                agent_sdk.result_to_capture(
                    result,
                    question=question,
                    history_text=history_text,
                    trajectory=self._trajectory,
                    metrics=usage,
                )
            )
            qspan.set(answer=result.answer, tokens_total=self.tokens_total)
        return result, capture, usage

    async def close(self) -> None:
        """Disconnect the client. Safe to call twice; never raises."""
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001 — a session that will not close is already gone
            return
