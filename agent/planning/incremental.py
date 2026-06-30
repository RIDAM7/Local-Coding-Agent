"""Phase 11 — the incremental planning loop (plan -> execute -> observe -> replan).

This is the shared driver both engines use. It is deliberately engine-agnostic: it
takes a ``step_executor`` (an async callable that runs ONE step via the selected
engine + governor) and drives the loop:

    Planner   -> ordered Steps (each with its own acceptance check)  [StepPlanner]
      loop:  StepExecutor runs ONE step
             Observer evaluates the result (reuse reflection + validation signals)
             Replanner revises ONLY the remaining steps with what was learned
      until: all steps done & acceptance met   |   governor stop

All plan state lives inside the shared :class:`AgentState` (via :class:`PlanOps`),
so nothing here owns a parallel store and Phase 15 resume comes for free.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from pydantic import BaseModel

from agent.config import logger, settings
from agent.engine import governor as gov
from agent.engine.governor import Governor
from agent.planning.plan_state import PlanOps
from agent.planning.replanner import Replanner
from agent.planning.step_planner import StepPlanner
from agent.state.agent_state import AgentState, StepResult


class StepOutcome(BaseModel):
    """What a step executor reports back for the Observer to evaluate."""
    success: bool
    summary: str = ""


# An executor runs exactly ONE step and returns its outcome. The agent engine
# supplies a tool-loop executor; tests supply a mock. It must mutate ``state``
# (files_modified, validation_results, observations, …) as it goes.
StepExecutor = Callable[[AgentState, StepResult], Awaitable[StepOutcome]]


class Observer:
    """Evaluates a step result, reusing the existing validation signals already
    accumulated on ``AgentState``. A step's executor reports a primary outcome;
    the Observer downgrades it to a failure if a fresh validation signal failed,
    so build/lint/test results count exactly as they do in Round 1."""

    def evaluate(self, state: AgentState, step: StepResult, outcome: StepOutcome,
                 validation_baseline: int) -> StepOutcome:
        new_validations = state.validation_results[validation_baseline:]
        failed = [v for v in new_validations if not v.success]
        if outcome.success and failed:
            detail = "; ".join(f"{v.stage}:{v.detail or 'failed'}" for v in failed)
            return StepOutcome(success=False, summary=f"validation failed ({detail})")
        return outcome


class IncrementalPlanner:
    """Drives plan -> execute step -> observe -> replan over a shared state."""

    def __init__(self, *, step_planner: Optional[StepPlanner] = None,
                 replanner: Optional[Replanner] = None,
                 observer: Optional[Observer] = None):
        self.step_planner = step_planner or StepPlanner()
        self.replanner = replanner
        self.observer = observer or Observer()

    async def run(self, state: AgentState, executor: StepExecutor, *,
                  governor: Optional[Governor] = None, plan=None,
                  context_bundle=None) -> AgentState:
        governor = governor or Governor.configure(state)
        state.add_timeline("engine", "incremental planning: plan -> execute -> observe -> replan")

        # Plan once up front; the plan lives inside AgentState from here on.
        if not state.plan.steps:
            await self.step_planner.plan(state, plan=plan, context_bundle=context_bundle)
        ops = PlanOps(state)

        while True:
            reason = governor.check_before_step()
            if reason:
                break

            step = ops.current()
            if step is None:
                governor.stop(gov.STOP_DONE)
                break

            governor.mark_step()
            step.status = "running"
            state.add_timeline("engine", f"step {step.index}: {step.description}")

            baseline = len(state.validation_results)
            try:
                raw = await executor(state, step)
            except Exception as e:
                logger.warning(f"IncrementalPlanner: step {step.index} executor error: {e}")
                raw = StepOutcome(success=False, summary=f"executor error: {type(e).__name__}: {e}")

            outcome = self.observer.evaluate(state, step, raw, baseline)

            if outcome.success:
                ops.mark_done(step, outcome.summary)
                state.add_observation(f"step {step.index} done: {outcome.summary or 'ok'}")
                continue

            ops.mark_failed(step, outcome.summary)
            state.add_observation(f"step {step.index} failed: {outcome.summary or 'failure'}")

            # Replan ONLY the remaining steps (bounded by the governor). On a
            # blocked/failed replan the loop stops — never a full restart.
            if settings.replan_on_failure and self.replanner is not None:
                revised = await self.replanner.replan(state, governor, reason=outcome.summary)
                if not revised:
                    break
            else:
                break

        self._finalize(state)
        return state

    def _finalize(self, state: AgentState) -> None:
        ops = PlanOps(state)
        done = len(ops.completed())
        total = len(state.plan.steps)
        all_ok = ops.all_done() and not any(s.status == "failed" for s in state.plan.steps)
        clean_stop = state.governor.stop_reason in (gov.STOP_DONE, None)
        status = "SUCCESS" if (all_ok and clean_stop) else "FAILURE"
        state.final_outputs.status = status
        state.final_outputs.summary = (
            f"{done}/{total} steps completed; {state.plan.replans_used} replan(s); "
            f"stop={state.governor.stop_reason}"
        )
        state.final_outputs.applied_changes = [f.path for f in state.files_modified]
        state.add_timeline("engine", f"incremental finished: status={status} "
                                     f"steps={done}/{total} replans={state.plan.replans_used}")


def render_plan_evolution(state: AgentState) -> str:
    """Render the plan evolution (original plan -> revisions) for the report."""
    plan = state.plan
    lines = ["## Plan Evolution", "", "**Original plan:**"]
    if plan.original_steps:
        lines += [f"{i + 1}. {d}" for i, d in enumerate(plan.original_steps)]
    else:
        lines.append("_(no plan recorded)_")

    for n, rev in enumerate(plan.revisions, start=1):
        lines += ["", f"**Revision {n}** — {rev.reason or 'replan'}:"]
        if rev.remaining_after:
            lines += [f"{i + 1}. {d}" for i, d in enumerate(rev.remaining_after)]
        else:
            lines.append("_(no remaining steps)_")

    if not plan.revisions:
        lines += ["", "_No replanning occurred (plan executed as-is)._"]
    return "\n".join(lines)
