import json
import os
from pathlib import Path
from unittest.mock import patch, Mock
from typing import List

from agent.orchestrator import Orchestrator
from agent.models.schemas import Report, RoutingCause, ExecutionOutcome, ArbitrationDecision
from agent.evaluation.arbitration_schemas import ArbitrationEffectivenessMetrics, CoverageResults, AuthorityVerification
from agent.config import settings

class ArbitrationBenchmarkRunner:
    def __init__(self, workspace_path: str = None, reports_dir: str = None):
        self.workspace_path = Path(workspace_path) if workspace_path else settings.get_workspace_path()
        self.reports_dir = Path(reports_dir) if reports_dir else Path("benchmarks/results")
        os.makedirs(self.reports_dir, exist_ok=True)
        
    async def run_benchmark(self, suite_path: str):
        with open(suite_path, 'r') as f:
            suite = json.load(f)
            
        reports = []
        for task_def in suite.get("tasks", []):
            report = await self._run_mocked_task(task_def)
            reports.append(report)
            
        metrics = self._analyze_reports(reports)
        
        output_path = self.reports_dir / "arbitration_effectiveness.json"
        with open(output_path, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2)
            
        print(f"Arbitration effectiveness results written to {output_path}")
        return metrics

    async def _run_mocked_task(self, task_def: dict) -> Report:
        """Runs the orchestrator with mocked LLMs and validators based on the task definition."""
        task_id = task_def["id"]
        val_pass = task_def.get("validation_pass", True)
        ref_pass = task_def.get("reflection_pass", True)
        claude_pass = task_def.get("claude_pass", True)
        budget_limit = task_def.get("budget_limit", False)
        payload_limit = task_def.get("payload_limit", False)
        
        orchestrator = Orchestrator(workspace_path=self.workspace_path, reports_dir=self.reports_dir)
        
        from agent.models.schemas import PatchValidationResult, RepairPatch, ValidationDiagnostic, FileOperation
        import time
        mock_validate_obj = PatchValidationResult(is_valid=True, errors=[], warnings=[], modified_patch=RepairPatch(operations=[FileOperation(type="create_file", path=f"test_{task_id}_{int(time.time()*1000)}.py", content="mock")], explanation="mock", confidence=0.9))
        mock_step_obj = ValidationDiagnostic(success=val_pass, command="mock", duration=0.1, stdout="mock", stderr="", exit_code=0, stage="TEST")
            
        from agent.reflection.schemas import ReflectionReport, ReflectionResult
        mock_reflect_obj = ReflectionReport(
            result=ReflectionResult.PASS if ref_pass else ReflectionResult.FAIL,
            summary="Mock",
            critiques=[],
            execution_time_ms=100,
            model_name="mock-model"
        )
            
        from agent.reviewers.schemas import ExternalReviewReport, ReviewIssue
        mock_claude_obj = ExternalReviewReport(
            reviewer="Claude",
            confidence=0.9,
            issues=[] if claude_pass else [ReviewIssue(severity="WARNING", category="LIKELY_BUG", description="mock", issue_confidence=0.9)],
            summary="Mock summary",
            recommended_action="APPROVE" if claude_pass else "REJECT",
            review_status="COMPLETED",
            latency_ms=100,
            tokens_sent=10,
            tokens_received=10,
            estimated_cost=0.05
        )
        
        from agent.models.schemas import Plan, PlanStep
        mock_plan_obj = Plan(goal="Mock Goal", summary="Mock Summary", steps=[PlanStep(id="1", description="Mock Step", expected_output="mock", status="TODO")])
            
        mock_code_obj = RepairPatch(operations=[FileOperation(type="create_file", path=f"test_{task_id}.py", content="mock")], explanation="mock", confidence=0.9)
        
        def mock_can_afford(*args, **kwargs):
            return not budget_limit
            
        def mock_check_payload(*args, **kwargs):
            return not payload_limit
                
        from agent.review.schemas import ConfidenceReport
        mock_confidence_obj = ConfidenceReport(
            confidence_score=90.0,
            patch_score=90.0,
            build_score=90.0,
            test_score=90.0,
            lint_score=90.0,
            repair_score=90.0,
            memory_score=90.0,
            file_scope_score=90.0,
            planner_complexity_score=90.0,
            memory_count_used=0,
            files_modified_count=1,
            plan_step_count=1,
            constraint_violations=0,
            review_required=True,
            review_decision="REVIEW_REQUIRED",
            contributing_factors={"mock": 0.0}
        )

        from agent.models.schemas import RetrievedContext
        mock_retrieved_context = RetrievedContext(files=[], results=[], total_files=0, total_chars=0)

        # When validation fails we still need to return a deterministic repair patch
        # so the repair loop never calls Ollama.
        mock_repair_context = Mock()
        mock_repair_context.original_task = f"Arbitration Task {task_id}"

        with patch.object(orchestrator.planner, 'create_plan', return_value=mock_plan_obj), \
             patch.object(orchestrator.retrieval_manager, 'search_context', return_value=mock_retrieved_context), \
             patch.object(orchestrator.coder, 'generate_patch', return_value=mock_code_obj), \
             patch.object(orchestrator.patch_validator, 'validate_and_repair', return_value=mock_validate_obj), \
             patch.object(orchestrator.test_validator, 'validate', return_value=mock_step_obj), \
             patch.object(orchestrator.build_validator, 'validate', return_value=mock_step_obj), \
             patch.object(orchestrator.lint_validator, 'validate', return_value=mock_step_obj), \
             patch.object(orchestrator.reflection_manager, 'reflect', return_value=mock_reflect_obj), \
             patch.object(orchestrator.claude_reviewer, 'review', return_value=mock_claude_obj), \
             patch.object(orchestrator.budget_manager, 'can_afford', return_value=not budget_limit), \
             patch.object(orchestrator.budget_manager, 'check_payload', return_value=not payload_limit), \
             patch.object(orchestrator.confidence_engine, 'evaluate', return_value=mock_confidence_obj), \
             patch.object(orchestrator.constraint_extractor, 'extract', return_value=Mock(success=True, constraints=[])), \
             patch.object(orchestrator.repair_manager, 'build_context', return_value=mock_repair_context), \
             patch.object(orchestrator.repair_manager, 'generate_repair', return_value=None):
                 
             task_desc = f"Arbitration Task {task_id}"
             # We catch exceptions so it produces a report cleanly
             try:
                 report_path = await orchestrator.run(task_desc)
             except Exception as e:
                 print(f"Exception during orchestrator run: {e}")
                 raise e
                 
             # Since orchestrator.run returns the report path, let's load it
             json_path = str(report_path).replace(".md", ".json")
             with open(json_path, "r") as f:
                 report_dict = json.load(f)
             
             # Re-inject the task_id into the task description for tracking
             report_dict["task"] = task_id
             report = Report(**report_dict)
             return report
             
    def _analyze_reports(self, reports: List[Report]) -> ArbitrationEffectivenessMetrics:
        metrics = ArbitrationEffectivenessMetrics(
            total_tasks=len(reports),
            claude_override_count=0,
            reflection_override_count=0,
            arbitration_triggered_count=0,
            claude_false_approval_count=0,
            decision_distribution={
                ArbitrationDecision.APPROVE_VALIDATED.value: 0,
                ArbitrationDecision.APPROVE_CLAUDE_OVERRIDDEN.value: 0,
                ArbitrationDecision.REJECT_VALIDATION_FAILED.value: 0
            },
            decision_percentages={},
            tasks_claude_overridden=[],
            tasks_reflection_overridden=[],
            tasks_validation_authoritative=[],
            tasks_claude_false_approval=[],
            tasks_fail_closed=[],
            coverage=CoverageResults(all_states_observed=False, missing_states=[]),
            authority=AuthorityVerification(
                validation_remains_authoritative=True,
                claude_never_merges_failing_code=True,
                claude_never_blocks_validated_code=True
            )
        )
        
        for r in reports:
            task_id = r.task
            
            if r.execution_outcome == ExecutionOutcome.FAIL_CLOSED:
                metrics.tasks_fail_closed.append(task_id)
                continue
                
            metrics.claude_override_count += r.claude_override_count
            metrics.reflection_override_count += r.reflection_override_count
            metrics.arbitration_triggered_count += r.arbitration_triggered_count
            metrics.claude_false_approval_count += r.claude_false_approval_count
            
            if r.arbitration_report:
                decision = r.arbitration_report.decision.value
                metrics.decision_distribution[decision] += 1
                
                if decision == ArbitrationDecision.APPROVE_CLAUDE_OVERRIDDEN.value or "CLAUDE" in r.arbitration_report.overridden_systems:
                    metrics.tasks_claude_overridden.append(task_id)
                    
                if "REFLECTION" in r.arbitration_report.overridden_systems:
                    metrics.tasks_reflection_overridden.append(task_id)
                    
                if decision == ArbitrationDecision.REJECT_VALIDATION_FAILED.value:
                    metrics.tasks_validation_authoritative.append(task_id)
                    
                if decision == ArbitrationDecision.REJECT_VALIDATION_FAILED.value and "CLAUDE" in r.arbitration_report.overridden_systems:
                    metrics.tasks_claude_false_approval.append(task_id)
                    
            # Authority Check
            val_failed = False
            if r.validation_report and r.validation_report.test_result and not r.validation_report.test_result.success:
                val_failed = True
                
            if val_failed and r.execution_outcome == ExecutionOutcome.SUCCESS:
                metrics.authority.validation_remains_authoritative = False
                metrics.authority.claude_never_merges_failing_code = False
                
            # claude_never_blocks_validated_code is only violated if:
            # - validation PASSED (val_failed is False)
            # - outcome is FAILURE
            # - it is NOT a FAIL_CLOSED case (those are system limit failures, not arbitration)
            # - the arbitration report exists and its decision is REJECT_VALIDATION_FAILED
            #   (meaning arbitration chose to reject despite validation passing)
            _system_limit_causes = (RoutingCause.BUDGET_EXHAUSTED, RoutingCause.PAYLOAD_OVERFLOW)
            if not val_failed \
                    and r.execution_outcome == ExecutionOutcome.FAILURE \
                    and r.routing_cause not in _system_limit_causes \
                    and r.arbitration_report is not None \
                    and r.arbitration_report.decision == ArbitrationDecision.REJECT_VALIDATION_FAILED:
                metrics.authority.claude_never_blocks_validated_code = False
                
        # Calculate percentages
        total_decisions = sum(metrics.decision_distribution.values())
        if total_decisions > 0:
            for k, v in metrics.decision_distribution.items():
                metrics.decision_percentages[k] = (v / total_decisions) * 100.0
                
        # Coverage
        required_decisions = [
            ArbitrationDecision.APPROVE_VALIDATED.value,
            ArbitrationDecision.APPROVE_CLAUDE_OVERRIDDEN.value,
            ArbitrationDecision.REJECT_VALIDATION_FAILED.value
        ]
        missing = []
        for d in required_decisions:
            if metrics.decision_distribution[d] == 0:
                missing.append(d)
                
        if not metrics.tasks_fail_closed:
            missing.append("FAIL_CLOSED")
        if not metrics.tasks_reflection_overridden:
            missing.append("REFLECTION_OVERRIDE")
            
        metrics.coverage.missing_states = missing
        metrics.coverage.all_states_observed = len(missing) == 0
        
        return metrics
