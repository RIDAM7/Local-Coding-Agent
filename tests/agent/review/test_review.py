import pytest
from agent.review.confidence import ConfidenceEngine
from agent.review.router import ReviewRouter
from agent.review.schemas import ReviewDecision

@pytest.fixture
def engine():
    return ConfidenceEngine()

@pytest.fixture
def router():
    return ReviewRouter()

def test_perfect_success_path(engine, router):
    report = engine.evaluate(
        build_success=True,
        test_success=True,
        lint_success=True,
        repair_attempts=0,
        memory_count=0,
        files_modified_count=1,
        plan_step_count=2,
        constraint_violations=0
    )
    assert report.confidence_score == 100.0
    assert not report.review_required
    
    decision = router.route(report)
    assert decision == ReviewDecision.APPROVE
    assert report.review_decision == ReviewDecision.APPROVE

def test_build_failure(engine, router):
    report = engine.evaluate(
        build_success=False,
        test_success=False,
        lint_success=True,
        repair_attempts=0,
        memory_count=0,
        files_modified_count=1,
        plan_step_count=2,
        constraint_violations=0
    )
    # Build (30%) + Test (30%) lost = max 40%
    assert report.confidence_score == 40.0
    assert report.review_required

    decision = router.route(report)
    # < 80 -> escalate to mandatory external review.
    assert decision == ReviewDecision.MANDATORY_REVIEW

def test_test_failure(engine, router):
    report = engine.evaluate(
        build_success=True,
        test_success=False,
        lint_success=True,
        repair_attempts=0,
        memory_count=0,
        files_modified_count=1,
        plan_step_count=2,
        constraint_violations=0
    )
    # Test (30%) lost = max 70%
    assert report.confidence_score == 70.0
    assert report.review_required
    # < 80 -> mandatory external review.
    assert router.route(report) == ReviewDecision.MANDATORY_REVIEW

def test_multi_repair_success(engine, router):
    report = engine.evaluate(
        build_success=True,
        test_success=True,
        lint_success=True,
        repair_attempts=2,
        memory_count=0,
        files_modified_count=1,
        plan_step_count=2,
        constraint_violations=0
    )
    # 2 repairs = 80/100 for repair score.
    # 10% weight of repair score -> 0.1 * 80 = 8.0 (instead of 10.0) -> total 98.0
    assert report.confidence_score == 98.0
    assert not report.review_required
    assert router.route(report) == ReviewDecision.APPROVE
    
def test_constraint_violation(engine, router):
    report = engine.evaluate(
        build_success=True,
        test_success=True,
        lint_success=True,
        repair_attempts=0,
        memory_count=0,
        files_modified_count=1,
        plan_step_count=2,
        constraint_violations=1
    )
    # 1 violation -> -20. Total = 80
    assert report.confidence_score == 80.0
    assert report.review_required
    # 80 <= score < 95 -> review required (not auto-approved).
    assert router.route(report) == ReviewDecision.REVIEW_REQUIRED

def test_large_multi_file_modification(engine, router):
    report = engine.evaluate(
        build_success=True,
        test_success=True,
        lint_success=True,
        repair_attempts=0,
        memory_count=0,
        files_modified_count=11,  # >10 gives 40 for file_scope
        plan_step_count=2,
        constraint_violations=0
    )
    # file_scope_score = 40 (instead of 100). Weight 5% -> 40 * 0.05 = 2.0 (instead of 5.0)
    # Lost 3 points. Total = 97
    assert report.confidence_score == 97.0
    assert not report.review_required
    assert router.route(report) == ReviewDecision.APPROVE

def test_complex_planner_output(engine, router):
    report = engine.evaluate(
        build_success=True,
        test_success=True,
        lint_success=True,
        repair_attempts=0,
        memory_count=0,
        files_modified_count=1,
        plan_step_count=11, # >10 gives 40 for planner_complexity
        constraint_violations=0
    )
    # planner_complexity = 40 (instead of 100). Weight 10% -> 40 * 0.10 = 4.0 (instead of 10.0)
    # Lost 6 points. Total = 94
    assert report.confidence_score == 94.0
    assert report.review_required
    # 80 <= score < 95 -> review required.
    assert router.route(report) == ReviewDecision.REVIEW_REQUIRED

def test_multiple_penalties(engine, router):
    report = engine.evaluate(
        build_success=True,
        test_success=True,
        lint_success=True,
        repair_attempts=1, # 90 -> 9.0 (lost 1)
        memory_count=3,    # 70 -> 3.5 (lost 1.5)
        files_modified_count=6, # 60 -> 3.0 (lost 2.0)
        plan_step_count=7, # 60 -> 6.0 (lost 4.0)
        constraint_violations=1 # lost 20.0
    )
    # Total = 100 - 1 - 1.5 - 2 - 4 - 20 = 71.5
    assert report.confidence_score == 71.5
    assert report.review_required
    # < 80 -> mandatory external review.
    assert router.route(report) == ReviewDecision.MANDATORY_REVIEW

def test_clamp_below_zero(engine, router):
    report = engine.evaluate(
        build_success=False,
        test_success=False,
        lint_success=False,
        repair_attempts=5, # 50 -> 5.0
        memory_count=5, # 70 -> 3.5
        files_modified_count=15, # 40 -> 2.0
        plan_step_count=15, # 40 -> 4.0
        constraint_violations=5 # -100
    )
    # Raw score: 0 + 0 + 0 + 5 + 3.5 + 2 + 4 = 14.5. Constraint penalty: -100 -> -85.5.
    # Should clamp to 0
    assert report.confidence_score == 0.0
    assert report.review_required
    # < 80 -> mandatory external review.
    assert router.route(report) == ReviewDecision.MANDATORY_REVIEW
