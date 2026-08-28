"""Load the demo pack and resolve a question to a recorded turn.

Matching is deliberately two-tier, and deliberately willing to fail:

  1. **Exact** match on the normalised gold question — the path a visitor takes
     when they click a showcase chip, which is the overwhelming majority.
  2. **Fuzzy** match, blending `SequenceMatcher` ratio with Jaccard token
     overlap, for a visitor who retypes a question in their own words.

Below the threshold the pack reports an honest miss rather than serving the
nearest recorded answer. Confidently returning the wrong report's number would
be the single worst thing this demo could do — it would make a system whose
entire subject is numerical accuracy look like it invents figures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PACK_DIR = Path(__file__).resolve().parent
PACK_FILENAME = "pack.json"

# Below this blended score the pack declines to answer. Tuned so a reworded
# question still lands ("what was the change in revenue" → "what was the change
# in total revenue") while an unrelated one does not.
MATCH_THRESHOLD = 0.62

_WORD_RE = re.compile(r"[a-z0-9.]+")


def normalise(question: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for matching."""
    return " ".join(_WORD_RE.findall(question.lower()))


def _tokens(question: str) -> set[str]:
    return set(normalise(question).split())


def similarity(a: str, b: str) -> float:
    """Blended sequence/Jaccard similarity in [0, 1].

    Sequence ratio alone over-rewards shared boilerplate ("what was the ..."),
    which every question in this dataset starts with; Jaccard alone ignores word
    order entirely. The mean of the two is steadier than either.
    """
    left, right = normalise(a), normalise(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    ratio = SequenceMatcher(None, left, right).ratio()
    lt, rt = _tokens(a), _tokens(b)
    jaccard = len(lt & rt) / len(lt | rt) if (lt | rt) else 0.0
    return (ratio + jaccard) / 2


@dataclass
class PackedTurn:
    """One recorded turn: its question, its answer, and its event stream."""

    report_id: str
    turn_index: int
    question: str
    answer: str
    program: str
    gold_answer: str
    correct: bool
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PackedTurn:
        """Build a turn from its serialised form."""
        return cls(
            report_id=str(raw.get("report_id", "")),
            turn_index=int(raw.get("turn_index", 0)),
            question=str(raw.get("question", "")),
            answer=str(raw.get("answer", "")),
            program=str(raw.get("program", "")),
            gold_answer=str(raw.get("gold_answer", "")),
            correct=bool(raw.get("correct", False)),
            events=list(raw.get("events", [])),
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "report_id": self.report_id,
            "turn_index": self.turn_index,
            "question": self.question,
            "answer": self.answer,
            "program": self.program,
            "gold_answer": self.gold_answer,
            "correct": self.correct,
            "events": self.events,
        }


@dataclass
class DemoPack:
    """Every recorded conversation available to the demo."""

    turns: list[PackedTurn]
    built_at: str = ""
    bundle: dict[str, Any] = field(default_factory=dict)

    @property
    def report_ids(self) -> list[str]:
        """Reports with at least one recorded turn, in pack order."""
        seen: list[str] = []
        for turn in self.turns:
            if turn.report_id not in seen:
                seen.append(turn.report_id)
        return seen

    def turns_for(self, report_id: str) -> list[PackedTurn]:
        """Recorded turns for one report, in conversation order."""
        return sorted(
            (t for t in self.turns if t.report_id == report_id),
            key=lambda t: t.turn_index,
        )

    def match(self, report_id: str, question: str) -> tuple[PackedTurn | None, float]:
        """Best recorded turn for `question` within `report_id`, and its score.

        Scoped to the report on purpose: the same question text ("what was the
        change?") appears against many reports, and answering it from the wrong
        document is precisely the failure this must not have.
        """
        candidates = self.turns_for(report_id)
        if not candidates:
            return None, 0.0
        target = normalise(question)
        for turn in candidates:
            if normalise(turn.question) == target:
                return turn, 1.0
        best = max(candidates, key=lambda t: similarity(t.question, question))
        score = similarity(best.question, question)
        return (best, score) if score >= MATCH_THRESHOLD else (None, score)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "built_at": self.built_at,
            "bundle": self.bundle,
            "turns": [t.as_dict() for t in self.turns],
        }


def pack_path() -> Path:
    """Path to the committed pack file."""
    return PACK_DIR / PACK_FILENAME


_cached: DemoPack | None = None


def load_pack(path: Path | None = None, *, refresh: bool = False) -> DemoPack:
    """Load the demo pack, cached. An absent pack yields an empty one."""
    global _cached
    if _cached is not None and not refresh and path is None:
        return _cached
    target = path or pack_path()
    if not target.exists():
        pack = DemoPack(turns=[])
    else:
        try:
            raw = json.loads(target.read_text())
        except json.JSONDecodeError:
            pack = DemoPack(turns=[])
        else:
            pack = DemoPack(
                turns=[PackedTurn.from_dict(t) for t in raw.get("turns", [])],
                built_at=str(raw.get("built_at", "")),
                bundle=dict(raw.get("bundle", {})),
            )
    if path is None:
        _cached = pack
    return pack


def reset_cache() -> None:
    """Drop the cached pack. For tests that write their own."""
    global _cached
    _cached = None
