"""Phase 11 — operations over the ``PlanState`` that lives inside ``AgentState``.

This is **not** a second source of truth: :class:`PlanOps` is a thin set of
operations over the ``state.plan`` slice (and the ``state.completed_steps`` /
``state.current_step`` slices) of the one shared :class:`AgentState`. Because the
plan lives inside the serializable AgentState, step-level persistence (Phase 15)
comes for free.

Vocabulary used throughout the planning package:
  - a *pending* step is one not yet attempted — the **remaining** steps.
  - a *done* / *failed* step is immutable history (also appended to
    ``state.completed_steps``). The replanner only ever rewrites the remaining
    (pending) tail; it never touches completed steps.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from agent.state.agent_state import (
    AgentState,
    PlanRevision,
    StepRef,
    StepResult,
)

# A step the planner emits: (description, acceptance_check).
StepSpec = Tuple[str, str]


class PlanOps:
    """Mutating operations over ``state.plan`` (the AgentState slice)."""

    def __init__(self, state: AgentState):
        self.state = state

    @property
    def plan(self):
        return self.state.plan

    # --- construction -------------------------------------------------------

    def set_plan(self, steps: Sequence[StepSpec], objective: str, *,
                 goal: str = "", summary: str = "", raw_plan=None) -> None:
        """Install the initial ordered plan. Records the original step list so
        the report can show ``original plan -> revisions``."""
        plan = self.plan
        plan.objective = objective
        plan.goal = goal or objective
        plan.summary = summary
        plan.raw_plan = raw_plan
        plan.steps = [
            StepResult(index=i, description=desc, acceptance=acc, status="pending")
            for i, (desc, acc) in enumerate(steps)
        ]
        plan.original_steps = [s.description for s in plan.steps]
        self.state.objective = objective

    # --- queries ------------------------------------------------------------

    def remaining(self) -> List[StepResult]:
        return [s for s in self.plan.steps if s.status == "pending"]

    def completed(self) -> List[StepResult]:
        return [s for s in self.plan.steps if s.status == "done"]

    def all_done(self) -> bool:
        """True when no pending step remains (failures that were not replanned
        away still count as terminal — the loop stops on its own)."""
        return not any(s.status == "pending" for s in self.plan.steps)

    def current(self) -> Optional[StepResult]:
        """The next step in flight — the first pending step — or None when the
        plan is exhausted. Also records it as ``state.current_step``."""
        for s in self.plan.steps:
            if s.status == "pending":
                self.state.current_step = StepRef(index=s.index, description=s.description)
                return s
        self.state.current_step = StepRef()
        return None

    # --- transitions --------------------------------------------------------

    def mark_done(self, step: StepResult, summary: str = "") -> None:
        step.status = "done"
        step.summary = summary
        self.state.completed_steps.append(
            StepResult(index=step.index, description=step.description,
                       acceptance=step.acceptance, status="done", summary=summary)
        )

    def mark_failed(self, step: StepResult, summary: str = "") -> None:
        step.status = "failed"
        step.summary = summary
        self.state.completed_steps.append(
            StepResult(index=step.index, description=step.description,
                       acceptance=step.acceptance, status="failed", summary=summary)
        )

    def revise_remaining(self, new_steps: Sequence[StepSpec], *, reason: str = "") -> None:
        """Replace ONLY the remaining (pending) steps with ``new_steps``.

        Completed and failed steps are immutable — they keep their position and
        are never reindexed away. The new tail is appended after the existing
        history and the whole event is recorded as a :class:`PlanRevision` (plan
        evolution)."""
        plan = self.plan
        before = [s.description for s in self.remaining()]
        kept = [s for s in plan.steps if s.status != "pending"]
        start = len(kept)
        revised = [
            StepResult(index=start + i, description=desc, acceptance=acc, status="pending")
            for i, (desc, acc) in enumerate(new_steps)
        ]
        plan.steps = kept + revised
        plan.replans_used += 1
        plan.revisions.append(PlanRevision(
            reason=reason,
            remaining_before=before,
            remaining_after=[s.description for s in revised],
        ))
