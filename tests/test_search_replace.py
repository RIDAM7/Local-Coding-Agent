"""Phase 7A search_replace (diff-based editing) tests."""

import json
import contextlib
from unittest.mock import patch, Mock, AsyncMock

import pytest

from agent.orchestrator import Orchestrator
from agent.validation.patch import PatchValidator
from agent.files.core import apply_search_replace_text
from agent.models.schemas import Plan, Patch, FileOperation, ValidationDiagnostic
from agent.reflection.schemas import ReflectionReport, ReflectionResult
from agent.review.router import ReviewDecision
from agent.safety.controller import SafetyMode
from agent.exceptions.errors import FileOperationError


def _diag(stage, success=True):
    return ValidationDiagnostic(stage=stage, command="", success=success,
                                stdout="", stderr="", exit_code=0 if success else 1, duration=0.0)


def _pipeline(modified_patch):
    """Carry a patch through run() with the REAL PatchValidator (so search_replace
    exactly-once validation runs); everything else is mocked."""
    return [
        patch('agent.planner.core.Planner.create_plan',
              AsyncMock(return_value=Plan(goal="g", summary="s", steps=[]))),
        patch('agent.coder.core.Coder.generate_patch', AsyncMock(return_value=modified_patch)),
        patch('agent.retrieval.retrieval_manager.RetrievalManager.search_context',
              AsyncMock(return_value=Mock(results=[]))),
        patch('agent.repair.constraints.ConstraintExtractor.extract',
              AsyncMock(return_value=Mock(success=True, constraints=[]))),
        patch('agent.validation.BuildValidator.validate', return_value=_diag("BUILD", True)),
        patch('agent.validation.LintValidator.validate', return_value=_diag("LINT", True)),
        patch('agent.validation.TestValidator.validate', return_value=_diag("TEST", True)),
        patch('agent.reflection.manager.ReflectionManager.reflect',
              AsyncMock(return_value=ReflectionReport(result=ReflectionResult.PASS, critiques=[],
                                                      summary="", execution_time_ms=0, model_name="t"))),
        patch('agent.review.router.ReviewRouter.route', return_value=ReviewDecision.APPROVE),
    ]


@contextlib.contextmanager
def _apply(patches):
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def _load(report_path):
    with open(report_path.replace('.md', '.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


# --- pure text helper --------------------------------------------------------

def test_apply_search_replace_text_exactly_once():
    assert apply_search_replace_text("a\nb\nc\n", "b\n", "B\n") == "a\nB\nc\n"


def test_apply_search_replace_text_zero_and_multiple_raise():
    with pytest.raises(FileOperationError):
        apply_search_replace_text("a\nb\n", "zzz", "x")
    with pytest.raises(FileOperationError):
        apply_search_replace_text("dup\ndup\n", "dup\n", "x")


# --- validator: exactly-once enforcement -------------------------------------

def test_validator_search_replace_exactly_once(tmp_path):
    target = tmp_path / "f.py"
    target.write_text("alpha = 1\nbeta = 2\nDUP\nDUP\n", encoding="utf-8")
    v = PatchValidator(tmp_path)

    ok = v.validate_and_repair(Patch(operations=[
        FileOperation(type="search_replace", path="f.py", search="alpha = 1\n", replace="alpha = 9\n")]))
    assert ok.is_valid
    assert ok.modified_patch.operations[0].type == "search_replace"

    zero = v.validate_and_repair(Patch(operations=[
        FileOperation(type="search_replace", path="f.py", search="NOPE", replace="x")]))
    assert not zero.is_valid

    multi = v.validate_and_repair(Patch(operations=[
        FileOperation(type="search_replace", path="f.py", search="DUP\n", replace="x")]))
    assert not multi.is_valid


def test_validator_search_replace_jail_rejects_traversal(tmp_path):
    v = PatchValidator(tmp_path)
    res = v.validate_and_repair(Patch(operations=[
        FileOperation(type="search_replace", path="../evil.py", search="a", replace="b")]))
    assert not res.is_valid


# --- applier: targeted edit on a large file ----------------------------------

@pytest.mark.asyncio
async def test_search_replace_applies_targeted_change(tmp_path):
    target = tmp_path / "big.py"
    target.write_text("".join(f"line_{i} = {i}\n" for i in range(200)), encoding="utf-8")

    op = FileOperation(type="search_replace", path="big.py",
                       search="line_100 = 100\n", replace="line_100 = 999\n")
    with _apply(_pipeline(Patch(operations=[op], commands=[]))):
        orch = Orchestrator(workspace_path=tmp_path, claude_enabled=False,
                            safety_mode=SafetyMode(auto_approve=True))
        await orch.run("edit the big file")

    content = target.read_text(encoding="utf-8")
    assert content.count("line_100 = 999") == 1
    assert "line_99 = 99" in content       # neighbors preserved
    assert "line_101 = 101" in content


@pytest.mark.asyncio
async def test_search_replace_dry_run_writes_nothing(tmp_path):
    target = tmp_path / "big.py"
    target.write_text("x = 1\n", encoding="utf-8")

    op = FileOperation(type="search_replace", path="big.py", search="x = 1\n", replace="x = 2\n")
    with _apply(_pipeline(Patch(operations=[op], commands=[]))):
        orch = Orchestrator(workspace_path=tmp_path, claude_enabled=False,
                            safety_mode=SafetyMode(dry_run=True))
        await orch.run("edit")

    assert target.read_text(encoding="utf-8") == "x = 1\n"  # unchanged on disk
