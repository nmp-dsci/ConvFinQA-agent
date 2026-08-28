"""Recorded conversations that let the keyless demo stream like the real thing.

The demo must feel like the product, not like a screenshot of it. So instead of
faking the agent, we record real sessions in dev and replay them through *the
same SSE event types the live path emits* — `stage_start`, `stage_output`,
`tool_call`, `tool_return`, `answer`. The frontend cannot tell the difference,
because from its side there is no difference.

ConvFinQA is unusually well set up for this: `turn_events` already emits typed
events, and every committed prediction row already carries the full per-stage IO
in its `*_io` columns. So a pack can be built from evidence that already exists,
without spending a single new API call — `cli.py` does exactly that.

What is *not* replayed stays genuinely live: report browsing, the document
viewer, gold questions, splits, the answers explorer, traces and experiments all
read committed files and work exactly as they do in dev. That is what makes the
demo honest rather than a mock.
"""

from convfinqa.serving.demo_pack.store import (
    DemoPack,
    PackedTurn,
    load_pack,
    pack_path,
)

__all__ = ["DemoPack", "PackedTurn", "load_pack", "pack_path"]
