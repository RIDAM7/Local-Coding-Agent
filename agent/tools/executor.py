"""Phase 10 — tool executor (dispatcher).

Dispatches a named tool call against the registry, operating on the shared
:class:`AgentState`. The per-call bookkeeping (tool_history, timeline, governor
counting) lives in :class:`Tool.run`; the executor adds the missing-tool guard and
a single seam the agent loop calls.
"""

from __future__ import annotations

from typing import Any, Dict

from agent.config import logger
from agent.state.agent_state import AgentState
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def dispatch(self, state: AgentState, name: str, args: Dict[str, Any] | None = None) -> ToolResult:
        args = args or {}
        tool = self.registry.get(name)
        if tool is None:
            msg = f"unknown tool '{name}' (available: {', '.join(self.registry.names())})"
            logger.warning(f"ToolExecutor: {msg}")
            state.add_timeline("tool", f"error: {msg}")
            return ToolResult(ok=False, status="error", summary=msg)
        return await tool.run(state, **args)
