"""Phase 14 - tiny event helpers.

This module exists so subsystems can record timeline events without importing
dashboard/report code. The event log still lives only on AgentState.
"""

from __future__ import annotations

from agent.state.agent_state import AgentState


def record_event(state: AgentState | None, kind: str, message: str) -> None:
    if state is not None:
        state.add_timeline(kind, message)
