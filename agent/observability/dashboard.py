"""Phase 14 - plain-text live dashboard renderer."""

from __future__ import annotations

from typing import Any, Dict

from agent.observability.panels import build_observability_snapshot, summarize_timeline
from agent.safety.redact import redact
from agent.state.agent_state import AgentState


def render_dashboard(state: AgentState, *, cost_summary: Any = None) -> str:
    return render_dashboard_snapshot(build_observability_snapshot(state, cost_summary=cost_summary))


def render_dashboard_snapshot(snapshot: Dict[str, Any]) -> str:
    progress = snapshot["progress"]
    governor = snapshot["governor"]
    tools = snapshot["tools"]
    tokens = snapshot["tokens"]
    memory = snapshot["memory"]
    repairs = snapshot["repairs"]
    lines = [
        "Execution Observability",
        "-" * 52,
        f"Goal      : {redact(str(snapshot['goal']))}",
        f"Mode      : {snapshot['mode']}    Status: {snapshot['status']}    "
        f"Elapsed: {snapshot['elapsed_seconds']:.3f}s",
        f"Progress  : {progress['completed']}/{progress['total']} "
        f"[{progress['bar']}] failed={progress['failed']}",
        "Governor  : "
        f"steps {governor['steps_used']}/{governor['max_steps']} | "
        f"tools {governor['tool_calls_used']}/{governor['tool_call_budget'] or 'off'} | "
        f"cost ${governor['cost_used_usd']:.6f}/${governor['run_budget_usd'] or 'off'} | "
        f"stop={governor['stop_reason'] or 'none'}",
        f"Tools     : {tools['useful']}/{tools['total']} useful; {tools['by_name']}",
        f"Tokens    : in {tokens['input_tokens']} / out {tokens['output_tokens']} | "
        f"cost ${tokens['est_cost']:.6f}",
        f"Memory    : vector {memory['vector_hits']} | markdown {len(memory['markdown_files'])} | "
        f"summaries {memory['summaries']}",
        f"Confidence: {snapshot['confidence']:.4f}",
        f"Repairs   : {repairs['succeeded']}/{repairs['total']} succeeded; failed={repairs['failed']}",
        "Timeline  :",
    ]
    timeline_lines = summarize_timeline(snapshot["timeline"])
    lines.extend(f"  - {line}" for line in timeline_lines)
    if not timeline_lines:
        lines.append("  - (no events)")
    return redact("\n".join(lines))
