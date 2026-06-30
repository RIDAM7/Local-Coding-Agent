"""Phase 14 - report rendering for observability snapshots."""

from __future__ import annotations

from typing import Any, Dict

from agent.observability.dashboard import render_dashboard_snapshot
from agent.observability.panels import build_observability_snapshot
from agent.state.agent_state import AgentState


def snapshot_for_report(state: AgentState, *, cost_summary: Any = None) -> Dict[str, Any]:
    return build_observability_snapshot(state, cost_summary=cost_summary)


def render_report_view(snapshot: Dict[str, Any]) -> str:
    lines = ["## Observability", "", "```text", render_dashboard_snapshot(snapshot), "```", ""]
    return "\n".join(lines)
