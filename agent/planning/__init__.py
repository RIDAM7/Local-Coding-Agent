"""Phase 11 — Incremental Planning & Replanning.

Replaces plan-once-execute-all with plan -> execute step -> observe -> replan.
The plan lives inside the shared :class:`AgentState` (no parallel store); both
engines share the same planner/replanner and step state.

Public surface:
  - :class:`StepPlanner`  — objective -> ordered steps (each with acceptance).
  - :class:`PlanOps`      — operations over the PlanState slice of AgentState.
  - :class:`Replanner`    — revise ONLY the remaining steps, governed.
  - :class:`IncrementalPlanner` / :class:`StepExecutor` / :class:`Observer` — the loop.
  - :func:`render_plan_evolution` — original plan -> revisions, for the report.
"""

from agent.planning.incremental import (
    IncrementalPlanner,
    Observer,
    StepExecutor,
    StepOutcome,
    render_plan_evolution,
)
from agent.planning.plan_state import PlanOps, StepSpec
from agent.planning.replanner import Replanner
from agent.planning.step_planner import StepPlanner

__all__ = [
    "StepPlanner",
    "PlanOps",
    "StepSpec",
    "Replanner",
    "IncrementalPlanner",
    "Observer",
    "StepExecutor",
    "StepOutcome",
    "render_plan_evolution",
]
