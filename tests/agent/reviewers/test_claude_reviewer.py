"""Phase 10 — the single generalized, provider-agnostic Reviewer.

The reviewer now talks to whatever ``build_client("reviewer")`` returns and reads
the standard ``LLMResult``. ``ClaudeReviewer`` remains only as an import alias for
``Reviewer`` (no ``claude_client`` import anywhere). The injected client is mocked
— no network, no live model.
"""

import pytest
from unittest.mock import AsyncMock

from agent.reviewers.reviewer import Reviewer, InternalReviewReport
from agent.reviewers.claude_reviewer import ClaudeReviewer
from agent.reviewers.schemas import ExternalReviewReport, ReviewIssue, ReviewCategory, ReviewSeverity
from agent.llm.providers.base import LLMResult, Usage
from agent.models.schemas import Task, Plan


def _client_returning(data, *, provider="ollama", model="qwen2.5:14b", in_tok=100, out_tok=50):
    client = AsyncMock()
    client.model = model
    client.generate_structured = AsyncMock(return_value=LLMResult(
        data=data, usage=Usage(provider=provider, model=model,
                               input_tokens=in_tok, output_tokens=out_tok)))
    return client


def test_claude_reviewer_is_alias_for_reviewer():
    assert issubclass(ClaudeReviewer, Reviewer)


@pytest.mark.asyncio
async def test_reviewer_success_no_issues():
    data = InternalReviewReport(confidence=1.0, issues=[], summary="Looks great",
                                recommended_action="Proceed")
    reviewer = Reviewer(client=_client_returning(data))
    task = Task(description="Fix the bug")
    plan = Plan(goal="test", summary="test summary", steps=[])

    report = await reviewer.review(task, plan, None, None, None)
    assert isinstance(report, ExternalReviewReport)
    assert report.confidence == 1.0
    assert len(report.issues) == 0
    assert report.review_status == "COMPLETED"
    assert report.tokens_sent == 100
    assert report.tokens_received == 50


@pytest.mark.asyncio
async def test_reviewer_with_issues():
    issue = ReviewIssue(category=ReviewCategory.LIKELY_BUG, severity=ReviewSeverity.CRITICAL,
                        description="Found a bug in logic", suggested_fix="Fix the logic",
                        issue_confidence=0.9)
    data = InternalReviewReport(confidence=0.5, issues=[issue], summary="Needs repair",
                                recommended_action="Fix")
    reviewer = Reviewer(client=_client_returning(data, in_tok=200, out_tok=100))

    report = await reviewer.review(Task(description="Fix the bug"), None, None, None, None)
    assert report.confidence == 0.5
    assert len(report.issues) == 1
    assert report.issues[0].category == ReviewCategory.LIKELY_BUG
    assert report.review_status == "COMPLETED"


@pytest.mark.asyncio
async def test_reviewer_bypassed_on_error():
    client = AsyncMock()
    client.model = "qwen2.5:14b"
    client.generate_structured = AsyncMock(side_effect=Exception("API Timeout"))
    reviewer = Reviewer(client=client)

    report = await reviewer.review(Task(description="Fix the bug"), None, None, None, None)
    assert report.confidence == 1.0
    assert len(report.issues) == 0
    assert report.review_status == "BYPASSED"
    assert "API Timeout" in report.summary
    assert report.tokens_sent == 0
    assert report.tokens_received == 0
    assert report.estimated_cost == 0.0
