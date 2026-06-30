"""Phase 10 — agent engine: mocked-model loop runs to completion; each governor
cap (max-steps, cost, timeout, oscillation) fires. No network, no live model."""

import asyncio

import pytest

from agent.config import settings
from agent.engine.agent_engine import AgentEngine, AgentAction
from agent.engine import governor as gov
from agent.engine.governor import Governor
from agent.state.agent_state import AgentState
from agent.llm.providers.base import LLMResult, Usage


class _ScriptedPolicy:
    model = "scripted:latest"

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
        if self.i < len(self.actions):
            a = self.actions[self.i]
            self.i += 1
        else:
            a = AgentAction(done=True, final_summary="exhausted")
        return LLMResult(data=a, usage=Usage(provider="ollama", model=model))


class _RepeatPolicy:
    """Always returns the SAME action (drives oscillation)."""
    model = "scripted:latest"

    def __init__(self, action):
        self.action = action

    async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
        return LLMResult(data=self.action, usage=Usage(provider="ollama", model=model))


class _CounterPolicy:
    """Returns a DISTINCT non-done action each call (no oscillation)."""
    model = "scripted:latest"

    def __init__(self):
        self.i = 0

    async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
        self.i += 1
        a = AgentAction(thought=f"step {self.i}", tool="list_dir", args={"path": str(self.i)})
        return LLMResult(data=a, usage=Usage(provider="ollama", model=model))


class _SlowPolicy:
    model = "scripted:latest"

    async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
        await asyncio.sleep(0.3)
        return LLMResult(data=AgentAction(done=True), usage=Usage(provider="ollama", model=model))


@pytest.mark.asyncio
async def test_loop_runs_to_completion(tmp_path):
    policy = _ScriptedPolicy([
        AgentAction(thought="write", tool="apply_patch",
                    args={"op": "create_file", "path": "out.py", "content": "v=1\n"}),
        AgentAction(done=True, final_summary="ok"),
    ])
    engine = AgentEngine(tmp_path, policy_client=policy)
    state = await engine.execute(AgentState(user_request="make out.py"))
    assert (tmp_path / "out.py").exists()
    assert state.governor.stop_reason == gov.STOP_DONE
    assert state.final_outputs.status == "SUCCESS"
    assert state.governor.steps_used >= 2


@pytest.mark.asyncio
async def test_governor_max_steps_fires(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_steps", 3)
    engine = AgentEngine(tmp_path, policy_client=_CounterPolicy())
    state = await engine.execute(AgentState(user_request="loop forever"))
    assert state.governor.stop_reason == gov.STOP_MAX_STEPS
    assert state.governor.steps_used == 3


@pytest.mark.asyncio
async def test_governor_oscillation_fires(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_steps", 25)
    repeat = AgentAction(thought="again", tool="list_dir", args={"path": "."})
    engine = AgentEngine(tmp_path, policy_client=_RepeatPolicy(repeat))
    state = await engine.execute(AgentState(user_request="spin"))
    assert state.governor.stop_reason == gov.STOP_OSCILLATION


@pytest.mark.asyncio
async def test_governor_step_timeout_fires(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "step_timeout_seconds", 0.05)
    engine = AgentEngine(tmp_path, policy_client=_SlowPolicy())
    state = await engine.execute(AgentState(user_request="slow"))
    assert state.governor.stop_reason == gov.STOP_STEP_TIMEOUT


def test_governor_cost_budget_fires(monkeypatch):
    monkeypatch.setattr(settings, "run_budget_usd", 1.0)
    state = AgentState()
    g = Governor.configure(state)
    assert g.add_cost(0.5) is None
    assert g.add_cost(0.8) == gov.STOP_COST_BUDGET
    assert state.governor.stop_reason == gov.STOP_COST_BUDGET
    # And a subsequent pre-step check keeps reporting the stop.
    assert g.check_before_step() == gov.STOP_COST_BUDGET


def test_governor_tool_budget_fires(monkeypatch):
    monkeypatch.setattr(settings, "tool_call_budget", 2)
    state = AgentState()
    g = Governor.configure(state)
    state.governor.tool_calls_used = 2
    assert g.check_before_step() == gov.STOP_TOOL_BUDGET
