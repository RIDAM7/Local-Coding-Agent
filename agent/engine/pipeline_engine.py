"""Phase 10 — Pipeline strategy.

Wraps today's :class:`Orchestrator` flow **unchanged** — this is the byte-for-byte
Round 1 path. The engine only creates/updates the shared :class:`AgentState` around
the call; it does not alter the orchestrator's behavior in any way.
"""

from __future__ import annotations

from agent.config import logger
from agent.engine.base import ExecutionEngine
from agent.state.agent_state import AgentState


class PipelineEngine(ExecutionEngine):
    name = "pipeline"

    def __init__(self, orchestrator=None, **orchestrator_kwargs):
        # An orchestrator may be injected (tests); otherwise built lazily so the
        # parity path constructs exactly the Round 1 orchestrator.
        self._orchestrator = orchestrator
        self._orchestrator_kwargs = orchestrator_kwargs

    async def execute(self, state: AgentState) -> AgentState:
        from agent.orchestrator import Orchestrator  # lazy: heavy import only when used

        orch = self._orchestrator or Orchestrator(**self._orchestrator_kwargs)
        state.execution_mode = "pipeline"
        state.add_timeline("engine", "pipeline strategy selected (Round 1 flow)")
        logger.info("PipelineEngine: delegating to the Round 1 orchestrator flow.")

        # Byte-for-byte: the orchestrator runs exactly as in Round 1.
        report_path = await orch.run(state.user_request)

        state.final_outputs.report_path = report_path
        state.final_outputs.summary = "pipeline run complete (see report)"
        return state
