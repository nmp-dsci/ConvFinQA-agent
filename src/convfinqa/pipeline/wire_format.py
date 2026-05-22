"""Wire-format helpers shared with DSPy ChatAdapter prompts."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def render_chat_inputs(fields: dict[str, Any]) -> str:
    """Render ordered input fields in DSPy ChatAdapter format."""
    parts = []
    for name, value in fields.items():
        if isinstance(value, BaseModel):
            rendered = value.model_dump_json(indent=2)
        elif isinstance(value, (list, dict)):
            try:
                normalized = (
                    [v.model_dump() if isinstance(v, BaseModel) else v for v in value]
                    if isinstance(value, list)
                    else value
                )
                rendered = json.dumps(normalized, indent=2, default=str)
            except TypeError:
                rendered = str(value)
        else:
            rendered = str(value)
        parts.append(f"[[ ## {name} ## ]]\n{rendered}")
    return "\n".join(parts)
