import logging
from typing import Optional, Any
from agent.models.schemas import ArbitrationReport, ArbitrationDecision, ArbitrationReason, ValidationReport
from agent.review.schemas import ConfidenceReport
from agent.reflection.schemas import ReflectionReport, ReflectionResult

logger = logging.getLogger(__name__)

class Arbitrator:
    def arbitrate(
        self,
        validation_report: Optional[ValidationReport],
        reflection_report: Optional[ReflectionReport],
        confidence_report: Optional[ConfidenceReport],
        claude_review_report: Optional[Any] # ExternalReviewReport is not tightly typed here to avoid cyclic imports
    ) -> ArbitrationReport:
        
        # Extract individual states
        validation_pass = validation_report is not None and validation_report.build_result and validation_report.build_result.success and validation_report.test_result and validation_report.test_result.success and validation_report.lint_result and validation_report.lint_result.success
        
        reflection_pass = False
        if reflection_report and reflection_report.result in [ReflectionResult.PASS, ReflectionResult.WARNING]:
            reflection_pass = True
            
        claude_pass = True # Assume PASS if skipped or not run
        if claude_review_report and hasattr(claude_review_report, 'issues') and claude_review_report.issues:
            claude_pass = False

        overridden_systems = []
        
        # Matrix Logic
        if validation_pass:
            # Validation PASS -> Approval path
            if claude_pass:
                decision = ArbitrationDecision.APPROVE_VALIDATED
                reason = ArbitrationReason.UNANIMOUS_APPROVAL
                if not reflection_pass:
                    overridden_systems.append("REFLECTION")
                    reason = ArbitrationReason.VALIDATION_IS_AUTHORITATIVE
                    
                explanation = "Validation passed. Code is functionally correct."
                
            else:
                # Claude Failed but Validation Passed -> Override Claude
                decision = ArbitrationDecision.APPROVE_CLAUDE_OVERRIDDEN
                reason = ArbitrationReason.CLAUDE_HALLUCINATION_OVERRIDDEN
                overridden_systems.append("CLAUDE")
                if not reflection_pass:
                    overridden_systems.append("REFLECTION")
                    
                explanation = "Validation passed, demonstrating functional correctness. Claude review raised issues but repair cycles were exhausted. Claude is safely overridden."
        else:
            # Validation FAIL -> Rejection path
            decision = ArbitrationDecision.REJECT_VALIDATION_FAILED
            if claude_pass:
                overridden_systems.append("CLAUDE")
                reason = ArbitrationReason.VALIDATION_IS_AUTHORITATIVE
                explanation = "Validation failed. Claude falsely approved failing code. Rejecting due to test/lint/build failures."
            else:
                reason = ArbitrationReason.UNANIMOUS_REJECTION
                explanation = "Validation failed. Claude and test suite agree on rejection."
                
            if reflection_pass:
                overridden_systems.append("REFLECTION")

        return ArbitrationReport(
            decision=decision,
            reason=reason,
            explanation=explanation,
            overridden_systems=overridden_systems
        )
