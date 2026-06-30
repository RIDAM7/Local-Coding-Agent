"""Phase 11 - incremental planning and bounded replanning. Offline."""

from unittest.mock import AsyncMock

import pytest

from agent.config import settings
from agent.engine import governor as gov
from agent.engine.governor import Governor
from agent.llm.providers.base import LLMResult, Usage
from agent.models.schemas import Plan, PlanStep
from agent.planning import IncrementalPlanner, Replanner, StepOutcome
from agent.state.agent_state import AgentState


def _plan(*descriptions: str) -> Plan:
    return Plan(
        goal="ship the task",
        summary="test plan",
        steps=[
            PlanStep(id=i + 1, description=desc, expected_output=f"{desc} accepted")
            for i, desc in enumerate(descriptions)
        ],
    )


class _ReplanClient:
    model = "planner:test"

    def __init__(self, revised_steps):
        self.revised_steps = list(revised_steps)
        self.prompts = []

    async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
        self.prompts.append(prompt)
        plan = _plan(*self.revised_steps)
        return LLMResult(data=plan, usage=Usage(provider="ollama", model=model))


@pytest.mark.asyncio
async def test_steps_execute_in_order(monkeypatch):
    monkeypatch.setattr(settings, "max_steps", 10)
    monkeypatch.setattr(settings, "replan_on_failure", True)

    state = AgentState(user_request="do three things")
    governor = Governor.configure(state)
    seen = []

    async def executor(st, step):
        seen.append(step.description)
        return StepOutcome(success=True, summary=f"{step.description} done")

    await IncrementalPlanner().run(
        state, executor, governor=governor, plan=_plan("first", "second", "third")
    )

    assert seen == ["first", "second", "third"]
    assert [s.status for s in state.completed_steps] == ["done", "done", "done"]
    assert state.plan.original_steps == ["first", "second", "third"]
    assert state.governor.stop_reason == gov.STOP_DONE


@pytest.mark.asyncio
async def test_step_failure_replans_only_remaining_steps(monkeypatch):
    monkeypatch.setattr(settings, "max_steps", 10)
    monkeypatch.setattr(settings, "max_replans", 3)
    monkeypatch.setattr(settings, "replan_on_failure", True)

    state = AgentState(user_request="do the feature")
    governor = Governor.configure(state)
    client = _ReplanClient(["third revised"])
    seen = []

    async def executor(st, step):
        seen.append(step.description)
        if step.description == "second":
            return StepOutcome(success=False, summary="second failed")
        return StepOutcome(success=True, summary=f"{step.description} done")

    await IncrementalPlanner(replanner=Replanner(client=client)).run(
        state, executor, governor=governor, plan=_plan("first", "second", "third")
    )

    assert seen == ["first", "second", "third revised"]
    assert [s.description for s in state.plan.steps] == ["first", "second", "third revised"]
    assert [s.status for s in state.plan.steps] == ["done", "failed", "done"]
    assert [s.description for s in state.completed_steps] == ["first", "second", "third revised"]
    assert state.plan.revisions[0].remaining_before == ["third"]
    assert state.plan.revisions[0].remaining_after == ["third revised"]
    assert state.plan.original_steps == ["first", "second", "third"]
    assert "second failed" in client.prompts[0]


@pytest.mark.asyncio
async def test_replan_governor_bound_fires(monkeypatch):
    monkeypatch.setattr(settings, "max_steps", 10)
    monkeypatch.setattr(settings, "max_replans", 1)
    monkeypatch.setattr(settings, "replan_on_failure", True)

    state = AgentState(user_request="keep failing")
    governor = Governor.configure(state)

    async def executor(st, step):
        return StepOutcome(success=False, summary=f"{step.description} failed")

    await IncrementalPlanner(replanner=Replanner()).run(
        state, executor, governor=governor, plan=_plan("first", "second", "third")
    )

    assert state.governor.stop_reason == gov.STOP_MAX_REPLANS
    assert state.governor.replans_used == 1
    assert state.plan.replans_used == 1
    assert len(state.plan.revisions) == 1


@pytest.mark.asyncio
async def test_incremental_planning_false_uses_round1_pipeline_path(monkeypatch):
    from agent import cli
    import agent.engine.selector as selector
    import agent.orchestrator as orchestrator_module

    monkeypatch.setattr(settings, "incremental_planning", False)
    monkeypatch.setattr(cli, "tooling_check", lambda: None)
    monkeypatch.setattr(cli, "preflight_check", AsyncMock(return_value=None))
    monkeypatch.setattr(cli, "_ensure_index", lambda orchestrator: None)
    monkeypatch.setattr(selector, "resolve_mode", AsyncMock(return_value=("pipeline", None)))

    fake_orchestrator = AsyncMock()
    fake_orchestrator.memory_manager = None
    fake_orchestrator.run = AsyncMock(return_value="reports/round1.md")
    fake_orchestrator.run_incremental = AsyncMock(return_value="reports/incremental.md")
    monkeypatch.setattr(
        orchestrator_module, "Orchestrator", lambda safety_mode=None: fake_orchestrator
    )

    args = cli.build_parser().parse_args(cli._normalize_argv(["run", "exact request", "--yes"]))
    rc = await cli.cmd_run(args)

    assert rc == 0
    fake_orchestrator.run.assert_awaited_once_with("exact request")
    fake_orchestrator.run_incremental.assert_not_awaited()
