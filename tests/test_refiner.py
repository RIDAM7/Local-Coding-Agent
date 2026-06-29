"""Phase 3 prompt refiner tests. The LLM client is mocked — no live models."""

from unittest.mock import patch, Mock, AsyncMock

import pytest

from agent.config import settings
from agent.llm.providers.base import LLMResult, Usage
from agent.refiner.core import PromptRefiner
from agent.refiner.schemas import RefinedPrompt
from agent.orchestrator import Orchestrator
from agent.models.schemas import Plan, Patch
from agent.reflection.schemas import ReflectionReport, ReflectionResult
from agent.review.router import ReviewDecision


# --- Unit: PromptRefiner ------------------------------------------------------

class _FakeClient:
    """A BaseLLMClient stand-in that returns a fixed RefinedPrompt."""

    model = "fake-refiner:latest"

    async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
        data = RefinedPrompt(
            refined_task="Build a CLI calculator supporting add and subtract.",
            clarified_goal="A working command-line calculator.",
            assumptions=["Python is the target language."],
            acceptance_criteria=["Adds two numbers", "Subtracts two numbers"],
            open_questions=[],
        )
        return LLMResult(data=data, usage=Usage(provider="ollama", model=model,
                                                input_tokens=3, output_tokens=4))


@pytest.mark.asyncio
async def test_refine_returns_populated_refined_prompt():
    refiner = PromptRefiner(_FakeClient())
    out = await refiner.refine("make a calculator")

    assert isinstance(out, RefinedPrompt)
    assert "calculator" in out.refined_task.lower()
    assert out.acceptance_criteria == ["Adds two numbers", "Subtracts two numbers"]
    # Usage is recorded for telemetry.
    assert refiner.last_usage is not None
    assert refiner.last_usage.input_tokens == 3
    assert refiner.last_usage.output_tokens == 4


# --- Orchestrator integration helpers ----------------------------------------

def _pipeline_mocks(create_plan):
    """Patch the full pipeline so run() completes cleanly; ``create_plan`` is the
    (async) function used for the planner so callers can capture its input."""
    return [
        patch('agent.planner.core.Planner.create_plan', new=create_plan),
        patch('agent.coder.core.Coder.generate_patch',
              AsyncMock(return_value=Patch(operations=[], commands=[]))),
        patch('agent.retrieval.retrieval_manager.RetrievalManager.search_context',
              AsyncMock(return_value=Mock(results=[]))),
        patch('agent.repair.constraints.ConstraintExtractor.extract',
              AsyncMock(return_value=Mock(success=True, constraints=[]))),
        patch('agent.validation.PatchValidator.validate_and_repair',
              return_value=Mock(is_valid=True,
                                modified_patch=Patch(operations=[], commands=[]),
                                errors=[], warnings=[])),
        patch('agent.validation.BuildValidator.validate', return_value=Mock(success=True)),
        patch('agent.reflection.manager.ReflectionManager.reflect',
              AsyncMock(return_value=ReflectionReport(result=ReflectionResult.PASS,
                                                      critiques=[], summary="",
                                                      execution_time_ms=0,
                                                      model_name="test"))),
        patch('agent.review.router.ReviewRouter.route', return_value=ReviewDecision.APPROVE),
    ]


import contextlib


@contextlib.contextmanager
def _apply(patches):
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


@pytest.mark.asyncio
async def test_orchestrator_uses_refined_task_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "refiner_enabled", True)

    refined = RefinedPrompt(
        refined_task="REFINED: build a calculator with add and subtract",
        clarified_goal="calculator",
        assumptions=["python"],
        acceptance_criteria=["supports add", "supports subtract"],
        open_questions=[],
    )

    captured = {}

    async def fake_create_plan(self, task):
        captured["description"] = task.description
        return Plan(goal="g", summary="s", steps=[])

    patches = _pipeline_mocks(fake_create_plan)
    patches.append(patch('agent.refiner.core.PromptRefiner.refine',
                         AsyncMock(return_value=refined)))

    with _apply(patches):
        orch = Orchestrator(claude_enabled=False)
        report_path = await orch.run("make a calculator")

    # Planner saw the refined text plus the appended acceptance criteria.
    assert "REFINED:" in captured["description"]
    assert "supports add" in captured["description"]
    assert report_path  # run completed


@pytest.mark.asyncio
async def test_orchestrator_skips_refiner_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "refiner_enabled", False)

    captured = {}

    async def fake_create_plan(self, task):
        captured["description"] = task.description
        return Plan(goal="g", summary="s", steps=[])

    patches = _pipeline_mocks(fake_create_plan)

    # The refiner class must never be constructed when disabled.
    with patch('agent.refiner.core.PromptRefiner') as MockRefiner:
        with _apply(patches):
            orch = Orchestrator(claude_enabled=False)
            await orch.run("make a calculator")

    MockRefiner.assert_not_called()
    # Byte-for-byte old path: planner saw the raw task untouched.
    assert captured["description"] == "make a calculator"


@pytest.mark.asyncio
async def test_orchestrator_falls_back_to_raw_prompt_on_refiner_failure(monkeypatch):
    monkeypatch.setattr(settings, "refiner_enabled", True)

    captured = {}

    async def fake_create_plan(self, task):
        captured["description"] = task.description
        return Plan(goal="g", summary="s", steps=[])

    patches = _pipeline_mocks(fake_create_plan)
    patches.append(patch('agent.refiner.core.PromptRefiner.refine',
                         AsyncMock(side_effect=TimeoutError("refiner timed out"))))

    with _apply(patches):
        orch = Orchestrator(claude_enabled=False)
        report_path = await orch.run("make a calculator")

    # Graceful fallback: planner saw the RAW prompt and the run still produced a report.
    assert captured["description"] == "make a calculator"
    assert report_path
