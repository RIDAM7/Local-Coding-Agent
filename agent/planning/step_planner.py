"""Phase 11 — Step Planner.

Turns the objective into an ordered list of steps, each carrying its own
acceptance check. It **reuses the Round 1 planner** (`planner/core`) rather than
reimplementing planning: the planner emits a typed :class:`Plan` whose
``steps[*].expected_output`` becomes each step's acceptance check. The result is
written into the shared :class:`AgentState` via :class:`PlanOps`, so the plan is
the single source of truth that lives inside the serializable state.
"""

from __future__ import annotations

from typing import Optional

from agent.config import logger
from agent.models.schemas import Plan, Task
from agent.planner.core import Planner
from agent.planning.plan_state import PlanOps
from agent.state.agent_state import AgentState, PlanState


class StepPlanner:
    """Build a :class:`PlanState` for ``state`` from the objective.

    Reuse, not reimplement: when no precomputed :class:`Plan` is supplied, the
    Round 1 :class:`Planner` is invoked (an injected one, or one built from
    ``client``). Each planner step becomes a step with an acceptance check.
    """

    def __init__(self, *, client=None, planner: Optional[Planner] = None):
        self._client = client
        self._planner = planner

    def _build_planner(self) -> Optional[Planner]:
        if self._planner is not None:
            return self._planner
        if self._client is not None:
            return Planner(self._client)
        return None

    async def plan(self, state: AgentState, *, plan: Optional[Plan] = None,
                   context_bundle=None) -> PlanState:
        objective = (state.user_request or state.objective or "").strip()

        if plan is None:
            planner = self._build_planner()
            if planner is not None:
                try:
                    plan = await planner.create_plan(Task(description=objective), context_bundle)
                except Exception as e:
                    logger.warning(f"StepPlanner: planner failed, using single-step fallback: {e}")
                    plan = None

        ops = PlanOps(state)
        if plan is not None and plan.steps:
            steps = [(s.description, s.expected_output or s.description) for s in plan.steps]
            ops.set_plan(steps, objective, goal=plan.goal, summary=plan.summary,
                         raw_plan=plan.model_dump())
            logger.info(f"StepPlanner: built {len(steps)} steps from the plan.")
        else:
            # Degenerate fallback: a single step that is the whole objective.
            ops.set_plan([(objective or "complete the task", "task complete")], objective)
            logger.info("StepPlanner: no plan available; using a single-step plan.")
        return state.plan
