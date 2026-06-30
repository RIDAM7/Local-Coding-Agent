"""Phase 10 — capability-driven auto selection + pipeline parity. Offline."""

import pytest
from unittest.mock import AsyncMock

from agent.config import settings
from agent.engine.capability_detector import Capabilities
from agent.engine.selector import resolve_mode, build_engine
from agent.engine.pipeline_engine import PipelineEngine
from agent.engine.agent_engine import AgentEngine
from agent.state.agent_state import AgentState


class _FakeDetector:
    def __init__(self, caps):
        self._caps = caps

    async def detect(self, provider, model, *, client=None, use_cache=True):
        return self._caps


@pytest.mark.asyncio
async def test_explicit_pipeline_mode(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", "pipeline")
    mode, caps = await resolve_mode()
    assert mode == "pipeline" and caps is None


@pytest.mark.asyncio
async def test_explicit_agent_mode(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", "agent")
    mode, caps = await resolve_mode()
    assert mode == "agent" and caps is None


@pytest.mark.asyncio
async def test_auto_selects_agent_when_capable(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", "auto")
    capable = Capabilities(structured_output=True, tool_calling=True, context_window=32000)
    mode, caps = await resolve_mode(detector=_FakeDetector(capable))
    assert mode == "agent"


@pytest.mark.asyncio
async def test_auto_selects_pipeline_when_not_capable(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", "auto")
    weak = Capabilities(structured_output=True, tool_calling=False, context_window=4000)
    mode, caps = await resolve_mode(detector=_FakeDetector(weak))
    assert mode == "pipeline"


def test_build_engine_returns_right_class():
    assert isinstance(build_engine("pipeline"), PipelineEngine)
    assert isinstance(build_engine("agent"), AgentEngine)


@pytest.mark.asyncio
async def test_pipeline_parity_delegates_to_orchestrator_unchanged():
    """The pipeline strategy is byte-for-byte the Round 1 flow: it calls
    orchestrator.run(user_request) and nothing else."""
    orch = AsyncMock()
    orch.run = AsyncMock(return_value="reports/parity.md")
    engine = PipelineEngine(orchestrator=orch)
    state = AgentState(user_request="exact request")
    await engine.execute(state)
    orch.run.assert_awaited_once_with("exact request")
