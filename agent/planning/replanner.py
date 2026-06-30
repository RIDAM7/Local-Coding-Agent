"""Phase 11 — Replanner.

Given what was learned (``state.observations`` + ``state.evidence``), rewrite ONLY
the remaining steps — never a full restart, completed steps untouched — within a
governed bound (``MAX_REPLANS`` enforced through the Execution Governor).

Reuse, not reimplement: the revised steps come from the same planner model (via
``build_client`` or an injected client) producing a typed :class:`Plan` scoped to
"the steps that still need doing". The governor counts each replan against the
leash (its own ``max_replans`` stop reason).
"""

from __future__ import annotations

from typing import List, Optional

from agent.config import logger
from agent.engine.governor import Governor
from agent.models.schemas import Plan, Task
from agent.planning.plan_state import PlanOps, StepSpec
from agent.state.agent_state import AgentState


class Replanner:
    def __init__(self, *, client=None, model: Optional[str] = None):
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        if self._model:
            return self._model
        return getattr(self._client, "model", None) or ""

    async def replan(self, state: AgentState, governor: Governor, *, reason: str = "") -> bool:
        """Revise the remaining steps. Returns True if the plan was revised, or
        False if the replan budget is exhausted (governor stops the run) or there
        is nothing left to revise."""
        # Account this replan against the governed bound FIRST — if the budget is
        # exhausted the governor stops the run and we do not replan.
        stop = governor.note_replan()
        if stop:
            logger.info(f"Replanner: replan budget exhausted ({stop}); not replanning.")
            state.add_observation(f"replan blocked: {stop}")
            return False

        ops = PlanOps(state)
        remaining = ops.remaining()
        if not remaining:
            logger.info("Replanner: nothing remaining to revise.")
            return False

        try:
            new_steps = await self._generate(state, reason)
        except Exception as e:
            logger.warning(f"Replanner: revision generation failed, keeping remaining steps: {e}")
            new_steps = None

        # Fail-safe: if no client / generation failed, keep the existing remaining
        # steps verbatim (a no-progress replan is still bounded by the governor).
        if not new_steps:
            new_steps = [(s.description, s.acceptance) for s in remaining]

        ops.revise_remaining(new_steps, reason=reason)
        state.add_timeline("engine", f"replan #{state.plan.replans_used}: {reason or 'failure'}")
        logger.info(f"Replanner: revised {len(new_steps)} remaining step(s) "
                    f"(replan #{state.plan.replans_used}).")
        return True

    async def _generate(self, state: AgentState, reason: str) -> Optional[List[StepSpec]]:
        if self._client is None:
            return None

        ops = PlanOps(state)
        done = "\n".join(f"- [done] {s.description}" for s in ops.completed()) or "(none)"
        remaining = "\n".join(f"- {s.description}" for s in ops.remaining()) or "(none)"
        observations = "\n".join(o.note for o in state.observations[-10:]) or "(none)"
        evidence = "\n".join(f"- {e.kind}: {e.detail}" for e in state.evidence[-10:]) or "(none)"

        prompt = f"""You are revising an execution plan after new information.
Do NOT restart the plan or redo completed work. Revise ONLY the steps that still
remain, taking the failure/observations into account.

<objective>
{state.objective or state.user_request}
</objective>

<completed_steps>
{done}
</completed_steps>

<remaining_steps>
{remaining}
</remaining_steps>

<what_we_learned>
reason: {reason}
observations:
{observations}
evidence:
{evidence}
</what_we_learned>

Return a JSON object matching this exact schema, containing ONLY the revised
remaining steps (the ones still to do):
{{
  "goal": "Overall goal",
  "summary": "Brief summary of the revised remaining work",
  "steps": [
    {{ "id": 1, "description": "Step description", "expected_output": "Acceptance check" }}
  ]
}}
You must ONLY return valid JSON. Do not generate code.
"""
        result = await self._client.generate_structured(self.model, prompt, Plan)
        plan: Plan = result.data
        return [(s.description, s.expected_output or s.description) for s in plan.steps]
