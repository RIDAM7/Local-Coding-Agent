import pytest
from agent.orchestrator import Orchestrator
from agent.review.confidence import ConfidenceEngine
from agent.review.budget import BudgetManager
from agent.models.schemas import RoutingCause, ExecutionOutcome

def test_confidence_cap_invariant():
    engine = ConfidenceEngine()
    
    # Even if memory/file scope/planner are perfect, if build fails, max is 94
    report = engine.evaluate(
        build_success=False,
        test_success=True,
        lint_success=True,
        repair_attempts=0,
        memory_count=0,
        files_modified_count=1,
        plan_step_count=1,
        constraint_violations=0
    )
    assert report.confidence_score <= 94.0
    assert report.review_required is True

def test_budget_manager():
    budget = BudgetManager(enforcement_enabled=True)
    budget.max_session_cost = 2.0
    assert budget.can_afford() is True
    
    budget.add_cost(2.5)
    assert budget.can_afford() is False

def test_budget_benchmark_bypass():
    budget = BudgetManager(enforcement_enabled=False)
    budget.max_session_cost = 2.0
    budget.add_cost(2.5)
    assert budget.can_afford() is True # Bypassed

def test_payload_overflow():
    budget = BudgetManager(enforcement_enabled=True)
    assert budget.check_payload(65000) is False
    assert budget.check_payload(50000) is True

from unittest.mock import patch, Mock, AsyncMock
from agent.models.schemas import Plan, Patch
from agent.reflection.schemas import ReflectionReport, ReflectionResult
from agent.review.router import ReviewDecision

@pytest.mark.asyncio
async def test_orchestrator_budget_exhaustion():
    # Mock budget manager
    with patch('agent.review.budget.BudgetManager.can_afford', return_value=False), \
         patch('agent.review.router.ReviewRouter.route', return_value=ReviewDecision.REVIEW_REQUIRED), \
         patch('agent.coder.core.Coder.generate_patch', AsyncMock(return_value=Patch(operations=[], commands=[]))), \
         patch('agent.planner.core.Planner.create_plan', AsyncMock(return_value=Plan(goal="test", summary="test", steps=[]))), \
         patch('agent.retrieval.retrieval_manager.RetrievalManager.search_context', AsyncMock(return_value=Mock(results=[]))), \
         patch('agent.repair.constraints.ConstraintExtractor.extract', AsyncMock(return_value=Mock(success=True, constraints=[]))), \
         patch('agent.validation.PatchValidator.validate_and_repair', return_value=Mock(is_valid=True, modified_patch=Patch(operations=[], commands=[]), errors=[], warnings=[])), \
         patch('agent.validation.BuildValidator.validate', return_value=Mock(success=False)), \
         patch('agent.reflection.manager.ReflectionManager.reflect', AsyncMock(return_value=ReflectionReport(result=ReflectionResult.PASS, critiques=[], summary="", execution_time_ms=0, model_name="test"))):
        
        orch = Orchestrator(budget_enforcement=True)
        orch.claude_enabled = True
        
        report_path = await orch.run("test task")
        
        # We can inspect the report if we load it
        import json
        with open(report_path.replace('.md', '.json'), 'r') as f:
            report = json.load(f)
            
        assert report["routing_cause"] == RoutingCause.BUDGET_EXHAUSTED
        assert report["execution_outcome"] == ExecutionOutcome.FAIL_CLOSED

@pytest.mark.asyncio
async def test_orchestrator_payload_overflow():
    # Mock budget manager
    with patch('agent.review.budget.BudgetManager.can_afford', return_value=True), \
         patch('agent.review.budget.BudgetManager.check_payload', return_value=False), \
         patch('agent.review.router.ReviewRouter.route', return_value=ReviewDecision.MANDATORY_REVIEW), \
         patch('agent.coder.core.Coder.generate_patch', AsyncMock(return_value=Patch(operations=[], commands=[]))), \
         patch('agent.planner.core.Planner.create_plan', AsyncMock(return_value=Plan(goal="test", summary="test", steps=[]))), \
         patch('agent.retrieval.retrieval_manager.RetrievalManager.search_context', AsyncMock(return_value=Mock(results=[]))), \
         patch('agent.repair.constraints.ConstraintExtractor.extract', AsyncMock(return_value=Mock(success=True, constraints=[]))), \
         patch('agent.validation.PatchValidator.validate_and_repair', return_value=Mock(is_valid=True, modified_patch=Patch(operations=[], commands=[]), errors=[], warnings=[])), \
         patch('agent.validation.BuildValidator.validate', return_value=Mock(success=False)), \
         patch('agent.reflection.manager.ReflectionManager.reflect', AsyncMock(return_value=ReflectionReport(result=ReflectionResult.PASS, critiques=[], summary="", execution_time_ms=0, model_name="test"))):
    
        orch = Orchestrator(budget_enforcement=True)
        orch.claude_enabled = True
        
        report_path = await orch.run("test task")
    
        import json
        with open(report_path.replace('.md', '.json'), 'r') as f:
            report = json.load(f)
        
    assert report["routing_cause"] == RoutingCause.PAYLOAD_OVERFLOW
    assert report["execution_outcome"] == ExecutionOutcome.FAIL_CLOSED

@pytest.mark.asyncio
async def test_orchestrator_high_confidence_bypass():
    # If >= 95, skips Claude entirely
    with patch('agent.review.router.ReviewRouter.route', return_value=ReviewDecision.APPROVE), \
         patch('agent.coder.core.Coder.generate_patch', AsyncMock(return_value=Patch(operations=[], commands=[]))), \
         patch('agent.planner.core.Planner.create_plan', AsyncMock(return_value=Plan(goal="test", summary="test", steps=[]))), \
         patch('agent.retrieval.retrieval_manager.RetrievalManager.search_context', AsyncMock(return_value=Mock(results=[]))), \
         patch('agent.repair.constraints.ConstraintExtractor.extract', AsyncMock(return_value=Mock(success=True, constraints=[]))), \
         patch('agent.validation.PatchValidator.validate_and_repair', return_value=Mock(is_valid=True, modified_patch=Patch(operations=[], commands=[]), errors=[], warnings=[])), \
         patch('agent.validation.BuildValidator.validate', return_value=Mock(success=True)), \
         patch('agent.reflection.manager.ReflectionManager.reflect', AsyncMock(return_value=ReflectionReport(result=ReflectionResult.PASS, critiques=[], summary="", execution_time_ms=0, model_name="test"))), \
         patch('agent.review.budget.BudgetManager.can_afford', return_value=False): # Mock budget to be exhausted, but since it's APPROVE, it shouldn't matter!
        
        orch = Orchestrator(budget_enforcement=True)
        orch.claude_enabled = True
        
        report_path = await orch.run("test task")
        
        import json
        with open(report_path.replace('.md', '.json'), 'r') as f:
            report = json.load(f)
            
        assert report["routing_cause"] == RoutingCause.SKIPPED_HIGH_CONFIDENCE
