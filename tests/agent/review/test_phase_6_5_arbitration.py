import pytest
from unittest.mock import Mock, patch
from agent.review.arbitration import Arbitrator
from agent.models.schemas import ArbitrationDecision, ArbitrationReason

class DummyReport:
    def __init__(self, issues=None):
        self.issues = issues

def create_validation(success=True):
    v = Mock()
    v.build_result = Mock(success=success)
    v.test_result = Mock(success=success)
    v.lint_result = Mock(success=success)
    return v

def create_reflection(success=True):
    v = Mock()
    from agent.reflection.schemas import ReflectionResult
    v.result = ReflectionResult.PASS if success else ReflectionResult.FAIL
    return v

@pytest.fixture
def arbitrator():
    return Arbitrator()

def test_approve_validated_unanimous(arbitrator):
    val = create_validation(True)
    ref = create_reflection(True)
    conf = Mock()
    claude = DummyReport(issues=[])
    
    report = arbitrator.arbitrate(val, ref, conf, claude)
    assert report.decision == ArbitrationDecision.APPROVE_VALIDATED
    assert report.reason == ArbitrationReason.UNANIMOUS_APPROVAL
    assert not report.overridden_systems

def test_approve_validated_reflection_fail(arbitrator):
    val = create_validation(True)
    ref = create_reflection(False)
    conf = Mock()
    claude = DummyReport(issues=[])
    
    report = arbitrator.arbitrate(val, ref, conf, claude)
    assert report.decision == ArbitrationDecision.APPROVE_VALIDATED
    assert report.reason == ArbitrationReason.VALIDATION_IS_AUTHORITATIVE
    assert "REFLECTION" in report.overridden_systems

def test_approve_claude_overridden(arbitrator):
    val = create_validation(True)
    ref = create_reflection(True)
    conf = Mock()
    claude = DummyReport(issues=["Issue 1"])
    
    report = arbitrator.arbitrate(val, ref, conf, claude)
    assert report.decision == ArbitrationDecision.APPROVE_CLAUDE_OVERRIDDEN
    assert report.reason == ArbitrationReason.CLAUDE_HALLUCINATION_OVERRIDDEN
    assert "CLAUDE" in report.overridden_systems
    assert "REFLECTION" not in report.overridden_systems

def test_reject_validation_failed_claude_pass(arbitrator):
    val = create_validation(False)
    ref = create_reflection(True)
    conf = Mock()
    claude = DummyReport(issues=[])
    
    report = arbitrator.arbitrate(val, ref, conf, claude)
    assert report.decision == ArbitrationDecision.REJECT_VALIDATION_FAILED
    assert report.reason == ArbitrationReason.VALIDATION_IS_AUTHORITATIVE
    assert "CLAUDE" in report.overridden_systems
    assert "REFLECTION" in report.overridden_systems

def test_reject_validation_failed_unanimous(arbitrator):
    val = create_validation(False)
    ref = create_reflection(False)
    conf = Mock()
    claude = DummyReport(issues=["Issue 1"])
    
    report = arbitrator.arbitrate(val, ref, conf, claude)
    assert report.decision == ArbitrationDecision.REJECT_VALIDATION_FAILED
    assert report.reason == ArbitrationReason.UNANIMOUS_REJECTION
    assert not report.overridden_systems
