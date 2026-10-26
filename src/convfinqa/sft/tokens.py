"""A row's length in the student's own tokens (s28's F8 and data card).

The student is ``Qwen/Qwen3-4B-Instruct-2507`` at tau-telecom-sft's pin. Its
tokenizer and chat template are read from the Hugging Face cache — the same
snapshot the trainer and vLLM load on Modal — and the template is rendered
with jinja2 the way ``transformers.apply_chat_template`` renders it, so
counting a row needs neither ``transformers`` nor ``torch`` here. The trainer
re-renders every row with ``transformers`` itself and checks the loss mask on
each; this count only decides which rows fit and what the card says.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

STUDENT = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def snapshot() -> Path:
    """The cached snapshot folder, or a clear error naming how to get it."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        STUDENT,
        revision=REVISION,
        allow_patterns=["tokenizer.json", "tokenizer_config.json"],
    )
    return Path(path)


@functools.cache
def _template() -> Any:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(x: Any, ensure_ascii: bool = False, indent: Any = None) -> str:
        # transformers' own filter: json.dumps, not jinja's HTML-escaping one
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    config = json.loads((snapshot() / "tokenizer_config.json").read_text())
    return env.from_string(config["chat_template"])


@functools.cache
def _tokenizer() -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(snapshot() / "tokenizer.json"))


def render(row: dict[str, Any]) -> str:
    """The text the trainer sees for a row, before the mask."""
    text: str = _template().render(
        messages=row["messages"], tools=row["tools"], add_generation_prompt=False
    )
    return text


def count(row: dict[str, Any]) -> int:
    """A rendered row's length in the student's tokens."""
    return len(_tokenizer().encode(render(row), add_special_tokens=False).ids)


def count_text(text: str) -> int:
    """A piece of text's length in the student's tokens."""
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)
