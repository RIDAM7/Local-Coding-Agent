"""Phase 4a command-execution tests. The Executor is mocked — no real shell runs."""

import json
import contextlib
from unittest.mock import patch, Mock, AsyncMock

import pytest

from agent.config import settings
from agent.orchestrator import Orchestrator
from agent.models.schemas import Plan, Patch, CommandExecution, ValidationDiagnostic
from agent.reflection.schemas import ReflectionReport, ReflectionResult
from agent.review.router import ReviewDecision


def _diag(stage, success=True):
    return ValidationDiagnostic(stage=stage, command="", success=success,
                                stdout="", stderr="", exit_code=0 if success else 1,
                                duration=0.0)


def _base_pipeline(modified_patch, build_success=True):
    """Patches that carry a patch (with commands) through run() to the command stage."""
    return [
        patch('agent.planner.core.Planner.create_plan',
              AsyncMock(return_value=Plan(goal="g", summary="s", steps=[]))),
        patch('agent.coder.core.Coder.generate_patch',
              AsyncMock(return_value=modified_patch)),
        patch('agent.retrieval.retrieval_manager.RetrievalManager.search_context',
              AsyncMock(return_value=Mock(results=[]))),
        patch('agent.repair.constraints.ConstraintExtractor.extract',
              AsyncMock(return_value=Mock(success=True, constraints=[]))),
        patch('agent.validation.PatchValidator.validate_and_repair',
              return_value=Mock(is_valid=True, modified_patch=modified_patch,
                                errors=[], warnings=[])),
        patch('agent.validation.BuildValidator.validate',
              return_value=_diag("BUILD", build_success)),
        # Lint/test validators are stubbed so the ONLY caller of Executor.run_command
        # is the Phase 4a command stage under test.
        patch('agent.validation.LintValidator.validate', return_value=_diag("LINT", True)),
        patch('agent.validation.TestValidator.validate', return_value=_diag("TEST", True)),
        patch('agent.reflection.manager.ReflectionManager.reflect',
              AsyncMock(return_value=ReflectionReport(result=ReflectionResult.PASS,
                                                      critiques=[], summary="",
                                                      execution_time_ms=0,
                                                      model_name="test"))),
        patch('agent.review.router.ReviewRouter.route', return_value=ReviewDecision.APPROVE),
    ]


@contextlib.contextmanager
def _apply(patches):
    with contextlib.ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in patches]
        yield mocks


def _load(report_path):
    with open(report_path.replace('.md', '.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_commands_run_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "execute_commands", True)
    patch_with_cmd = Patch(operations=[], commands=["echo hi"])

    run_command = AsyncMock(return_value=CommandExecution(
        command="echo hi", stdout="hi", stderr="", exit_code=0, duration=0.01))

    patches = _base_pipeline(patch_with_cmd)
    patches.append(patch('agent.execution.core.Executor.run_command', run_command))

    with _apply(patches):
        orch = Orchestrator(claude_enabled=False)
        report_path = await orch.run("create and run hello")

    run_command.assert_awaited_once_with("echo hi")
    report = _load(report_path)
    assert len(report["commands_executed"]) == 1
    assert report["commands_executed"][0]["command"] == "echo hi"
    assert report["commands_executed"][0]["exit_code"] == 0
    assert report["proposed_commands"] == ["echo hi"]


@pytest.mark.asyncio
async def test_nonzero_exit_triggers_repair_loop(monkeypatch):
    monkeypatch.setattr(settings, "execute_commands", True)
    patch_with_cmd = Patch(operations=[], commands=["false"])

    run_command = AsyncMock(return_value=CommandExecution(
        command="false", stdout="", stderr="boom", exit_code=1, duration=0.01))

    # The repair path: build_context succeeds, generate_repair returns None so the
    # loop aborts cleanly — we just assert the existing self-healing path was entered.
    build_context = AsyncMock(return_value=Mock(
        normalized_diagnostic=Mock(classification="BUILD_FAILURE")))
    generate_repair = AsyncMock(return_value=None)

    patches = _base_pipeline(patch_with_cmd)
    patches.append(patch('agent.execution.core.Executor.run_command', run_command))
    patches.append(patch('agent.repair.manager.RepairManager.build_context', build_context))
    patches.append(patch('agent.repair.manager.RepairManager.generate_repair', generate_repair))

    with _apply(patches):
        orch = Orchestrator(claude_enabled=False)
        report_path = await orch.run("create and run failing command")

    # The failing command fed the EXISTING repair loop.
    run_command.assert_awaited()
    build_context.assert_awaited()
    generate_repair.assert_awaited()
    report = _load(report_path)
    assert report["final_status"] == "FAILURE"


@pytest.mark.asyncio
async def test_commands_not_run_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "execute_commands", False)
    patch_with_cmd = Patch(operations=[], commands=["echo hi"])

    run_command = AsyncMock()

    patches = _base_pipeline(patch_with_cmd)
    patches.append(patch('agent.execution.core.Executor.run_command', run_command))

    with _apply(patches):
        orch = Orchestrator(claude_enabled=False)
        report_path = await orch.run("create and run hello")

    # Nothing executed silently.
    run_command.assert_not_called()
    report = _load(report_path)
    assert report["commands_executed"] == []
    assert report["proposed_commands"] == ["echo hi"]

    # The markdown labels them clearly as proposed-but-not-run.
    with open(report_path, 'r', encoding='utf-8') as f:
        md = f.read()
    assert "NOT executed" in md
    assert "echo hi" in md
