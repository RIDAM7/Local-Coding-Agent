"""Phase 16 — Stateless specialized Worker.

The Worker contract: a pure function of its input. Receives a scoped unit of
work (``WorkerSpec``), executes it via a governed engine pass, and returns a
``WorkerResult``. The worker contains no orchestration state, no scheduling
logic, and no merge logic — it forgets everything after returning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent.config import logger
from agent.engine.base import ExecutionEngine
from agent.engine.selector import build_engine
from agent.safety.controller import SafetyMode
from agent.state.agent_state import AgentState, FileChange, Observation, ValidationResult
from agent.state.agent_state import TaskMetadata


class WorkerSpec(BaseModel):
    """The unit of work a stateless worker receives.

    Contains everything the worker needs and nothing it doesn't: just a role,
    a sub-task description, and the scoped context/graph views it needs to
    execute. The worker builds its own child AgentState from this.
    """
    role: str                         # backend | frontend | db | test | docs
    sub_task: str                     # the scoped sub-task description
    user_request: str = ""            # original full request for context
    scoped_context: Optional[Any] = None   # ContextBundle subset
    scoped_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    parent_session_id: str = ""
    execution_mode: str = "pipeline"


class WorkerResult(BaseModel):
    """What a worker returns. Every field is a slice the Merger reconciles
    back into the parent AgentState."""
    success: bool
    summary: str = ""
    files_modified: List[FileChange] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    steps_used: int = 0
    tool_calls_used: int = 0
    cost_used_usd: float = 0.0
    replans_used: int = 0
    status: str = "SUCCESS"


class Worker:
    """Stateless, specialized worker.

    Usage::

        result = await Worker.run(spec)
        # result is a WorkerResult — no cross-worker state survives.
    """

    @staticmethod
    async def run(spec: WorkerSpec, *, safety_mode: SafetyMode = None,
                  incremental: bool = True) -> WorkerResult:
        """Execute one scoped unit of work via the governed engine.

        Builds a child AgentState, runs the engine on it, extracts the result
        slices, and returns a ``WorkerResult``. The worker holds no state
        across calls.
        """
        child = AgentState(
            user_request=spec.sub_task,
            task=TaskMetadata(description=spec.sub_task),
            execution_mode=spec.execution_mode,
            objective=spec.sub_task,
        )
        child.loaded_context = spec.scoped_context
        # Attach the parent session id so session persistence propagates.
        child.task.id = spec.parent_session_id or child.task.id

        engine: ExecutionEngine = build_engine(
            child.execution_mode,
            safety_mode=safety_mode or SafetyMode(auto_approve=True),
            incremental=incremental,
        )

        try:
            result_state = await engine.execute(child)
        except Exception as e:
            logger.warning(f"Worker [{spec.role}] failed: {e}")
            return WorkerResult(
                success=False,
                summary=f"engine error: {type(e).__name__}: {e}",
                status="FAILURE",
            )

        return WorkerResult(
            success=result_state.final_outputs.status == "SUCCESS",
            summary=result_state.final_outputs.summary,
            files_modified=list(result_state.files_modified),
            evidence=[e.model_dump() for e in result_state.evidence],
            observations=list(result_state.observations),
            validation_results=list(result_state.validation_results),
            steps_used=result_state.governor.steps_used,
            tool_calls_used=result_state.governor.tool_calls_used,
            cost_used_usd=result_state.governor.cost_used_usd,
            replans_used=result_state.governor.replans_used,
            status=result_state.final_outputs.status,
        )
