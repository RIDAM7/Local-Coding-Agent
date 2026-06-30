"""Phase 14 - observability panels over AgentState.

Every function here is a read-only projection over AgentState. No panel owns
runtime state; the engines and tools keep appending to AgentState as before.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from agent.safety.redact import redact
from agent.state.agent_state import AgentState


def _parse_dt(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def elapsed_seconds(state: AgentState) -> float:
    start = _parse_dt(state.governor.started_at) or _parse_dt(state.task.created_at)
    end = _parse_dt(state.timeline[-1].at) if state.timeline else None
    if not start or not end:
        return 0.0
    return max(0.0, round((end - start).total_seconds(), 3))


def _progress_bar(done: int, total: int, width: int = 10) -> str:
    if total <= 0:
        return "." * width
    filled = min(width, round(width * done / total))
    return "#" * filled + "." * (width - filled)


def _cost_summary_dict(cost_summary: Any = None) -> Dict[str, Any]:
    if cost_summary is None:
        return {"input_tokens": 0, "output_tokens": 0, "est_cost": 0.0, "roles": []}
    roles = getattr(cost_summary, "per_role", []) or []
    return {
        "input_tokens": int(getattr(cost_summary, "total_input_tokens", 0) or 0),
        "output_tokens": int(getattr(cost_summary, "total_output_tokens", 0) or 0),
        "est_cost": float(getattr(cost_summary, "total_est_cost", 0.0) or 0.0),
        "roles": [
            {
                "role": getattr(r, "role", ""),
                "provider": getattr(r, "provider", ""),
                "model": getattr(r, "model", ""),
                "input_tokens": getattr(r, "input_tokens", 0),
                "output_tokens": getattr(r, "output_tokens", 0),
                "est_cost": getattr(r, "est_cost", 0.0),
            }
            for r in roles
        ],
    }


def build_observability_snapshot(state: AgentState, *, cost_summary: Any = None) -> Dict[str, Any]:
    """Build the structured observability payload used by dashboard + report."""
    total_steps = len(state.plan.steps)
    completed = len([s for s in state.completed_steps if s.status == "done"])
    failed_steps = len([s for s in state.completed_steps if s.status == "failed"])
    tool_counts = Counter(t.name for t in state.tool_history)
    useful_tools = len([t for t in state.tool_history if t.status == "ok"])
    repair_successes = len([r for r in state.repair_attempts if r.success])

    timeline = [
        {"kind": e.kind, "message": redact(e.message), "at": e.at}
        for e in state.timeline
    ]
    return {
        "goal": state.objective or state.plan.objective or state.user_request,
        "mode": state.execution_mode,
        "status": state.final_outputs.status,
        "elapsed_seconds": elapsed_seconds(state),
        "progress": {
            "completed": completed,
            "failed": failed_steps,
            "total": total_steps,
            "bar": _progress_bar(completed, total_steps),
            "current_step": state.current_step.model_dump(),
        },
        "governor": state.governor.model_dump(),
        "tools": {
            "total": len(state.tool_history),
            "useful": useful_tools,
            "by_name": dict(sorted(tool_counts.items())),
        },
        "tokens": _cost_summary_dict(cost_summary),
        "memory": {
            "vector_hits": len(state.memory_refs.vector_ids),
            "markdown_files": list(state.memory_refs.markdown_files),
            "summaries": len(state.memory_refs.summaries),
        },
        "confidence": state.confidence,
        "repairs": {
            "total": len(state.repair_attempts),
            "succeeded": repair_successes,
            "failed": len(state.repair_attempts) - repair_successes,
        },
        "validation": {
            "total": len(state.validation_results),
            "failed": len([v for v in state.validation_results if not v.success]),
        },
        "timeline": timeline,
    }


def summarize_timeline(events: Iterable[Dict[str, Any]], *, limit: int = 20) -> list[str]:
    return [
        f"[{e.get('kind', '')}] {e.get('message', '')}"
        for e in list(events)[-limit:]
    ]
