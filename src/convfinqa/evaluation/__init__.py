"""Shared evaluation primitives for the ConvFinQA scripts.

`pydantic_agent.py`, `api_eval.py`, and `dspy_agent.py` all need to:
  - score predictions against gold answers under several tolerance rules
  - load a partial CSV cache, identify fully-scored conversations, and merge new
    results back into the same file

Both responsibilities live here so the scripts can't drift apart.
"""

from convfinqa.evaluation.cache import (
    flush_csv_atomic,
    identify_cached_conversations,
    load_cached_conversations,
)
from convfinqa.evaluation.metrics import numeric_match

__all__ = [
    "flush_csv_atomic",
    "identify_cached_conversations",
    "load_cached_conversations",
    "numeric_match",
]
