from typing import Dict, Any
from agent.review.schemas import ConfidenceReport, ReviewDecision

class ConfidenceEngine:
    def evaluate(
        self,
        build_success: bool,
        test_success: bool,
        lint_success: bool,
        repair_attempts: int,
        memory_count: int,
        files_modified_count: int,
        plan_step_count: int,
        constraint_violations: int
    ) -> ConfidenceReport:
        
        # Build Status: 30%
        build_score = 100.0 if build_success else 0.0
        
        # Test Status: 30%
        test_score = 100.0 if test_success else 0.0
        
        # Lint Status: 10%
        lint_score = 100.0 if lint_success else 0.0
        
        # Repair Attempts: 10%
        repair_score = max(0.0, 100.0 - (repair_attempts * 10.0))
        
        # Memory Usage: 5%
        if memory_count == 0:
            memory_score = 100.0
        elif memory_count == 1:
            memory_score = 90.0
        elif memory_count == 2:
            memory_score = 80.0
        else:
            memory_score = 70.0
            
        # Files Modified: 5%
        if files_modified_count <= 2:
            file_scope_score = 100.0
        elif files_modified_count <= 5:
            file_scope_score = 80.0
        elif files_modified_count <= 10:
            file_scope_score = 60.0
        else:
            file_scope_score = 40.0
            
        # Planner Complexity: 10%
        if plan_step_count <= 3:
            planner_complexity_score = 100.0
        elif plan_step_count <= 6:
            planner_complexity_score = 80.0
        elif plan_step_count <= 10:
            planner_complexity_score = 60.0
        else:
            planner_complexity_score = 40.0
            
        # Calculate weighted confidence
        weighted_score = (
            (build_score * 0.30) +
            (test_score * 0.30) +
            (lint_score * 0.10) +
            (repair_score * 0.10) +
            (memory_score * 0.05) +
            (file_scope_score * 0.05) +
            (planner_complexity_score * 0.10)
        )
        
        # Apply Constraint Violation Penalty
        weighted_score -= (constraint_violations * 20.0)
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, weighted_score))
        
        # Phase 6.4 Invariant Protection: Validation failures cap confidence at 94
        if not (build_success and test_success and lint_success):
            final_score = min(final_score, 94.0)
            
        review_required = final_score < 95.0
        
        # Dummy ReviewDecision to be filled by Router later, or we can just leave it APPROVE for now. 
        # The schema requires review_decision, we can initialize it to APPROVE or whatever.
        
        return ConfidenceReport(
            confidence_score=final_score,
            review_required=review_required,
            review_decision=ReviewDecision.APPROVE, # Temporary, Router overrides
            contributing_factors={
                "build": build_score * 0.30,
                "test": test_score * 0.30,
                "lint": lint_score * 0.10,
                "repair": repair_score * 0.10,
                "memory": memory_score * 0.05,
                "file_scope": file_scope_score * 0.05,
                "planner_complexity": planner_complexity_score * 0.10,
                "constraint_penalty": -(constraint_violations * 20.0)
            },
            build_score=build_score,
            test_score=test_score,
            lint_score=lint_score,
            repair_score=repair_score,
            memory_score=memory_score,
            file_scope_score=file_scope_score,
            planner_complexity_score=planner_complexity_score,
            memory_count_used=memory_count,
            files_modified_count=files_modified_count,
            plan_step_count=plan_step_count,
            constraint_violations=constraint_violations
        )
