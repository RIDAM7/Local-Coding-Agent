"""Phase 10 — common engine interface.

Both strategies (pipeline / agent) implement :class:`ExecutionEngine` and operate
on the *same* shared :class:`AgentState`. The selector picks one; the governor
leashes both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent.state.agent_state import AgentState


class ExecutionEngine(ABC):
    name: str = "engine"

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Run the strategy to completion (or governor stop), mutating ``state``."""
        ...
