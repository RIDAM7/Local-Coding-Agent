"""Phase 15 — Resume manager.

Restores a session from its latest checkpoint and continues the governed loop.
The resumed execution behaves exactly like a continuous session because the
full AgentState (plan, timeline, governor, observations, …) is rehydrated
from the checkpoint.
"""

from __future__ import annotations

from typing import Optional, Tuple

from agent.exceptions.errors import SessionError
from agent.session.store import SCHEMA_VERSION, SessionStore
from agent.state.agent_state import AgentState


class ResumeManager:
    """Rehydrate a session from its latest checkpoint.

    Usage::

        mgr = ResumeManager()
        state, step_index = mgr.resume("abc123")
        # continue the governed loop starting from step_index + 1
    """

    def __init__(self, store: Optional[SessionStore] = None):
        self._store = store or SessionStore()

    def resume(self, session_id: str) -> Tuple[AgentState, int]:
        """Restore AgentState from the latest checkpoint.

        Returns ``(state, last_completed_step_index)``.
        Raises ``SessionError`` if the session is missing, corrupted, or
        incompatible.
        """
        row = self._store.get(session_id)
        if row is None:
            raise SessionError(f"Session '{session_id}' not found.")

        if row.schema_version != SCHEMA_VERSION:
            raise SessionError(
                f"Session '{session_id}' has schema v{row.schema_version}, "
                f"expected v{SCHEMA_VERSION}. Cannot resume."
            )

        if row.status == "corrupted":
            raise SessionError(f"Session '{session_id}' is marked as corrupted.")

        if row.status == "completed":
            raise SessionError(f"Session '{session_id}' is already completed.")

        latest = self._store.latest_checkpoint_index(session_id)
        if latest is None:
            raise SessionError(f"Session '{session_id}' has no checkpoints to resume from.")

        try:
            state = self._store.load_checkpoint(session_id, latest)
        except SessionError:
            self._store.update_status(session_id, "corrupted")
            raise

        self._store.update_status(session_id, "active")
        state.add_timeline("engine", f"session resumed from checkpoint step {latest}")

        return state, latest

    def list_sessions(self) -> list:
        """Return a list of session summary dicts for the CLI ``sessions`` command."""
        rows = self._store.list(limit=50)
        return [r.to_dict() for r in rows]
