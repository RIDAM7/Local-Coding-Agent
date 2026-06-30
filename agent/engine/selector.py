"""Phase 10 — strategy selector.

Resolves ``EXECUTION_MODE = auto | pipeline | agent`` into a concrete engine.
Under ``auto`` the :class:`CapabilityDetector` decides — **agent only when the
role model is genuinely agent-capable** (reliable structured output + tool calling
+ adequate context), otherwise **pipeline**. No model-size heuristic anywhere.
"""

from __future__ import annotations

from typing import Optional, Tuple

from agent.config import logger, settings
from agent.engine.base import ExecutionEngine
from agent.engine.capability_detector import Capabilities, CapabilityDetector
from agent.llm.factory import resolve_model, resolve_provider

# The role whose model runs the agent loop's reasoning. The planner role drives
# planning/decisions, so its capabilities decide whether the agent loop is viable.
SELECTION_ROLE = "planner"


async def resolve_mode(*, role: str = SELECTION_ROLE, detector: CapabilityDetector = None,
                       client=None) -> Tuple[str, Optional[Capabilities]]:
    """Return ``(mode, capabilities)`` where mode is 'pipeline' or 'agent'."""
    mode = (settings.execution_mode or "auto").strip().lower()
    if mode == "pipeline":
        return "pipeline", None
    if mode == "agent":
        return "agent", None

    # auto — let capabilities decide.
    provider = resolve_provider(role)
    model = resolve_model(role)
    detector = detector or CapabilityDetector()
    caps = await detector.detect(provider, model, client=client)
    chosen = "agent" if caps.is_agent_capable() else "pipeline"
    logger.info(f"Selector(auto): {provider}/{model} -> {chosen} "
                f"(structured={caps.structured_output}, tools={caps.tool_calling}, "
                f"ctx={caps.context_window}, source={caps.source})")
    return chosen, caps


def build_engine(mode: str, **kwargs) -> ExecutionEngine:
    """Construct the engine for a resolved mode."""
    if mode == "agent":
        from agent.engine.agent_engine import AgentEngine
        agent_kwargs = {k: v for k, v in kwargs.items()
                        if k in ("workspace_path", "safety_mode", "safety_controller",
                                 "policy_client", "memory_manager", "incremental")}
        return AgentEngine(**agent_kwargs)
    from agent.engine.pipeline_engine import PipelineEngine
    return PipelineEngine(**kwargs)
