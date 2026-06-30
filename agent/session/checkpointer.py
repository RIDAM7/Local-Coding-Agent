"""Phase 15 — Checkpointer.

Records a checkpoint of the full (redacted) AgentState after every step.
Checkpoints are idempotent and crash-safe: a corrupt or partial write is
detected on load and reported as a ``SessionError``.
"""

from __future__ import annotations

from typing import Optional

from agent.config import logger, settings
from agent.session.store import SessionStore
from agent.state.agent_state import AgentState


class Checkpointer:
    """Records a checkpoint of AgentState after every step.

    Designed to be called by both the pipeline (``run_incremental``) and the
    agent engine (``_execute_incremental``) at the end of each step, before
    the next step or replan begins. The checkpoint is crash-safe because the
    JSON write is atomic (write to temp → rename).
    """

    def __init__(self, store: Optional[SessionStore] = None):
        self._store = store or SessionStore()

    @property
    def enabled(self) -> bool:
        return settings.session_persistence

    def checkpoint_step(self, state: AgentState, step_index: int) -> None:
        """Record a checkpoint after completing a step.

        If ``SESSION_PERSISTENCE`` is disabled this is a no-op.
        Missing session_id is handled by creating one lazily.
        """
        if not self.enabled:
            return

        sid = state.task.id or settings.get_workspace_path().name
        try:
            self._store.save_checkpoint(sid, step_index, state)
            logger.debug(f"Checkpoint: session={sid}, step={step_index}")
        except Exception as e:
            logger.warning(f"Checkpoint failed (non-fatal): {e}")

    def get_latest_checkpoint(self, state: AgentState) -> Optional[int]:
        """Find the latest checkpoint step for a session. Returns None if none."""
        if not self.enabled:
            return None
        sid = state.task.id
        if not sid:
            return None
        try:
            return self._store.latest_checkpoint_index(sid)
        except Exception:
            return None
