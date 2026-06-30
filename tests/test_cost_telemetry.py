"""Phase 7C cost / token telemetry tests."""

import json
import contextlib
from unittest.mock import patch, Mock, AsyncMock

import pytest

from agent.llm.pricing import estimate_cost, price_for
from agent.llm.providers.base import Usage
from agent.orchestrator import Orchestrator
from agent.models.schemas import Plan, Patch, ValidationDiagnostic
from agent.reflection.schemas import ReflectionReport, ReflectionResult
from agent.review.router import ReviewDecision
from agent.safety.controller import SafetyMode


# --- price table -------------------------------------------------------------

def test_estimate_cost_cloud_nonzero():
    # gpt-4o-mini: 0.00015 in + 0.0006 out per 1k tokens.
    assert estimate_cost("openai", "gpt-4o-mini", 1000, 1000) == round(0.00015 + 0.0006, 6)


def test_estimate_cost_ollama_is_zero():
    assert estimate_cost("ollama", "qwen2.5-coder:32b", 5000, 9000) == 0.0


def test_estimate_cost_unknown_cloud_model_zero():
    assert estimate_cost("openai", "totally-unknown-model", 1000, 1000) == 0.0


def test_anthropic_sonnet_priced():
    pin, pout = price_for("anthropic", "claude-3-5-sonnet-20240620")
    assert pin > 0 and pout > 0


# --- aggregation into the report + budget ------------------------------------

def _diag(stage, success=True):
    return ValidationDiagnostic(stage=stage, command="", success=success,
                                stdout="", stderr="", exit_code=0 if success else 1, duration=0.0)


def _pipeline(modified_patch):
    return [
        patch('agent.planner.core.Planner.create_plan',
              AsyncMock(return_value=Plan(goal="g", summary="s", steps=[]))),
        patch('agent.coder.core.Coder.generate_patch', AsyncMock(return_value=modified_patch)),
        patch('agent.retrieval.retrieval_manager.RetrievalManager.search_context',
              AsyncMock(return_value=Mock(results=[]))),
        patch('agent.repair.constraints.ConstraintExtractor.extract',
              AsyncMock(return_value=Mock(success=True, constraints=[]))),
        patch('agent.validation.PatchValidator.validate_and_repair',
              return_value=Mock(is_valid=True, modified_patch=modified_patch, errors=[], warnings=[])),
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


@pytest.mark.asyncio
async def test_cost_summary_aggregates_and_feeds_budget(tmp_path):
    patch_obj = Patch(operations=[], commands=[])
    with _apply(_pipeline(patch_obj)):
        orch = Orchestrator(workspace_path=tmp_path, claude_enabled=False,
                            safety_mode=SafetyMode(auto_approve=True))
        # Inject per-role usage (mocked methods don't overwrite these instance attrs).
        orch.coder.last_usage = Usage(provider="openai", model="gpt-4o-mini",
                                      input_tokens=1000, output_tokens=2000)
        orch.planner.last_usage = Usage(provider="ollama", model="qwen2.5",
                                        input_tokens=500, output_tokens=500)
        report_path = await orch.run("do something")

    report = _load(report_path)
    cs = report["cost_summary"]
    by_role = {r["role"]: r for r in cs["per_role"]}

    assert "coder" in by_role and "planner" in by_role
    assert by_role["planner"]["est_cost"] == 0.0          # local is free
    assert by_role["coder"]["est_cost"] > 0.0             # cloud computed
    assert cs["total_input_tokens"] == 1500
    assert cs["total_output_tokens"] == 2500
    assert cs["total_est_cost"] == by_role["coder"]["est_cost"]

    # Budget manager accounts for the cloud spend.
    assert orch.budget_manager.session_cost >= by_role["coder"]["est_cost"]

    # The markdown report renders the table.
    with open(report_path, "r", encoding="utf-8") as f:
        md = f.read()
    assert "Cost & Token Usage" in md
