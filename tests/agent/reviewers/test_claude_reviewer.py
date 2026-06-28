import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from agent.reviewers.claude_reviewer import ClaudeReviewer
from agent.reviewers.schemas import ExternalReviewReport, ReviewIssue, ReviewCategory, ReviewSeverity
from agent.models.schemas import Task, Plan

class MockReportData:
    def __init__(self, confidence, issues, summary, recommended_action):
        self.confidence = confidence
        self.issues = issues
        self.summary = summary
        self.recommended_action = recommended_action

@pytest.fixture
def mock_claude_client():
    with patch("agent.reviewers.claude_reviewer.ClaudeClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        yield mock_client

@pytest.mark.asyncio
async def test_claude_reviewer_success_no_issues(mock_claude_client):
    reviewer = ClaudeReviewer()
    
    mock_data = MockReportData(
        confidence=1.0,
        issues=[],
        summary="Looks great",
        recommended_action="Proceed"
    )
    
    # Return (data, input_tokens, output_tokens)
    mock_claude_client.generate_structured.return_value = (mock_data, 100, 50)
    
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
async def test_claude_reviewer_with_issues(mock_claude_client):
    reviewer = ClaudeReviewer()
    
    issue = ReviewIssue(
        category=ReviewCategory.LIKELY_BUG,
        severity=ReviewSeverity.CRITICAL,
        description="Found a bug in logic",
        suggested_fix="Fix the logic",
        issue_confidence=0.9
    )
    
    mock_data = MockReportData(
        confidence=0.5,
        issues=[issue],
        summary="Needs repair",
        recommended_action="Fix"
    )
    
    mock_claude_client.generate_structured.return_value = (mock_data, 200, 100)
    
    task = Task(description="Fix the bug")
    
    report = await reviewer.review(task, None, None, None, None)
    
    assert report.confidence == 0.5
    assert len(report.issues) == 1
    assert report.issues[0].category == ReviewCategory.LIKELY_BUG
    assert report.review_status == "COMPLETED"

@pytest.mark.asyncio
async def test_claude_reviewer_bypassed_on_error(mock_claude_client):
    reviewer = ClaudeReviewer()
    
    # Force an exception to trigger BYPASSED
    mock_claude_client.generate_structured.side_effect = Exception("API Timeout")
    
    task = Task(description="Fix the bug")
    
    report = await reviewer.review(task, None, None, None, None)
    
    assert report.confidence == 1.0
    assert len(report.issues) == 0
    assert report.review_status == "BYPASSED"
    assert "API Timeout" in report.summary
    assert report.tokens_sent == 0
    assert report.tokens_received == 0
    assert report.estimated_cost == 0.0
