"""In-memory session store.

Kept in process memory behind a small interface, deliberately. The demo runs a
single App Runner instance at max-size 1, so there is no second process to share
with; Redis would be infrastructure bought to solve a problem this deployment
does not have. If it ever does, `SessionStore` is the seam to swap.

This is also why `--workers 1` is load-bearing for the backend: two workers means
two of these, and a session created against one is invisible to the other.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from convfinqa.data.schemas import ConversationHistory
from convfinqa.serving.models import HistoryItem, SessionResponse


@dataclass
class SessionState:
    """One visitor's conversation against one report."""

    session_id: str
    report_id: str
    created_at: datetime
    updated_at: datetime
    conversation: ConversationHistory = field(default_factory=ConversationHistory)

    def touch(self) -> None:
        """Mark the session as recently used, deferring TTL eviction."""
        self.updated_at = datetime.now(timezone.utc)

    def as_response(self) -> SessionResponse:
        """Serialise for the API."""
        return SessionResponse(
            session_id=self.session_id,
            report_id=self.report_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            n_turns=len(self.conversation.pairs),
            history=[
                HistoryItem.model_validate(p.model_dump())
                for p in self.conversation.pairs
            ],
        )


class SessionStore:
    """TTL-evicted sessions, each with its own lock to serialise turns."""

    def __init__(
        self, ttl_seconds: int = 1800, valid_reports: set[str] | None = None
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.sessions: dict[str, SessionState] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self._valid_reports = valid_reports

    def create(self, report_id: str) -> SessionState:
        """Open a session against `report_id`. Raises KeyError if unknown."""
        if self._valid_reports is not None and report_id not in self._valid_reports:
            raise KeyError(report_id)
        now = datetime.now(timezone.utc)
        state = SessionState(
            session_id=str(uuid4()),
            report_id=report_id,
            created_at=now,
            updated_at=now,
        )
        self.sessions[state.session_id] = state
        self.locks[state.session_id] = asyncio.Lock()
        return state

    def get(self, session_id: str) -> SessionState:
        """Fetch a session. Raises KeyError when it has been evicted."""
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(session_id) from exc

    def delete(self, session_id: str) -> None:
        """Drop a session and its lock."""
        self.sessions.pop(session_id, None)
        self.locks.pop(session_id, None)

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """The per-session lock that serialises turns within a conversation."""
        try:
            return self.locks[session_id]
        except KeyError as exc:
            raise KeyError(session_id) from exc

    def evict_expired(self) -> list[str]:
        """Remove sessions idle beyond the TTL; return their ids."""
        now = datetime.now(timezone.utc)
        expired = [
            sid
            for sid, state in self.sessions.items()
            if (now - state.updated_at).total_seconds() > self.ttl_seconds
        ]
        for sid in expired:
            self.delete(sid)
        return expired


def history_items(conversation: ConversationHistory) -> list[HistoryItem]:
    """Render a conversation's turns for the API."""
    return [HistoryItem.model_validate(p.model_dump()) for p in conversation.pairs]
