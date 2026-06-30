"""Phase 10 — Tool base class.

Every Round 1 capability is exposed as a :class:`Tool` that **takes and updates the
shared `AgentState`**. The base class owns the bookkeeping every tool must do —
append a redacted :class:`ToolCall` to ``state.tool_history``, push a timeline
event, and count the call against the governor — so individual tools only
implement ``_execute`` and append their domain slice (files_read, files_modified,
validation_results, observations, …).

No tool reimplements logic: each wraps an existing module (files/core, execution,
validation, git, memory, retrieval) and every *write* routes through the
SafetyController + workspace jail.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from agent.safety.redact import redact
from agent.state.agent_state import AgentState, ToolCall


class ToolResult(BaseModel):
    ok: bool = True
    status: str = "ok"               # ok | error | blocked | skipped
    summary: str = ""
    data: Optional[Any] = None


def _redact_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Redact string args before they are stored on the state (never leak secrets)."""
    out: Dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str):
            out[k] = redact(v)
        else:
            out[k] = v
    return out


class Tool:
    """Base tool. Subclasses set ``name``/``description``/``parameters`` and
    implement :meth:`_execute`."""

    name: str = "tool"
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    async def run(self, state: AgentState, **kwargs) -> ToolResult:
        start = time.time()
        try:
            result = await self._execute(state, **kwargs)
        except Exception as e:  # tools never crash the loop
            result = ToolResult(ok=False, status="error", summary=f"{type(e).__name__}: {e}")

        duration = round(time.time() - start, 3)
        state.tool_history.append(ToolCall(
            name=self.name,
            args=_redact_args(kwargs),
            status=result.status,
            result_summary=redact(result.summary or "")[:500],
            duration=duration,
        ))
        state.add_timeline("tool", f"{self.name}: {result.status} — {redact(result.summary or '')[:160]}")
        state.governor.tool_calls_used += 1
        return result

    async def _execute(self, state: AgentState, **kwargs) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def spec(self) -> Dict[str, Any]:
        """The JSON schema spec advertised to a tool-calling model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
