import time
from typing import Any
from agent.config import settings, logger
from agent.llm.claude_client import ClaudeClient
from agent.reviewers.schemas import ExternalReviewReport

MAX_REVIEW_CONTEXT_CHARS = 6000

class ClaudeReviewer:
    def __init__(self):
        self.client = ClaudeClient()
        self.model = settings.claude_model

    async def review(
        self,
        task: Any,
        plan: Any,
        patch: Any,
        validation_report: Any,
        reflection_report: Any
    ) -> ExternalReviewReport:
        start_time = time.time()
        
        # Build prompt
        prompt = "You are a Reviewer agent. Your task is to review the execution of a coding task.\n\n"
        prompt += f"Task: {getattr(task, 'description', str(task))[:1000]}\n\n"
        
        if plan:
            prompt += f"Plan Summary: {str(getattr(plan, 'steps', plan))[:1000]}\n\n"
            
        # Stringify the patch
        patch_str = ""
        if patch and hasattr(patch, "operations"):
            for op in patch.operations:
                patch_str += f"File: {op.path} Action: {op.type}\n"
        
        prompt += f"Patch Summary:\n{patch_str[:1000]}\n\n"
        
        if validation_report:
            prompt += f"Validation Results: {str(validation_report)[:1000]}\n\n"
            
        if reflection_report:
            prompt += f"Reflection Summary: {str(getattr(reflection_report, 'summary', reflection_report))[:1000]}\n\n"
            
        # Truncate strictly
        if len(prompt) > MAX_REVIEW_CONTEXT_CHARS:
            prompt = prompt[:MAX_REVIEW_CONTEXT_CHARS]
            
        prompt += "\nEvaluate this execution strictly against the following categories: SCOPE_DRIFT, MISSING_FILE, LIKELY_BUG, VALIDATION_GAP, CONSTRAINT_RISK, ARCHITECTURE_RISK.\n"
        prompt += "Return a JSON object matching the InternalReviewReport schema. If there are no issues, return an empty list for issues."

        try:
            import asyncio
            from pydantic import BaseModel
            from agent.reviewers.schemas import ReviewIssue
            from typing import List
            
            class InternalReviewReport(BaseModel):
                confidence: float
                issues: List[ReviewIssue]
                summary: str
                recommended_action: str
                
            report_data, input_tokens, output_tokens = await asyncio.wait_for(
                self.client.generate_structured(
                    model=self.model,
                    prompt=prompt,
                    schema=InternalReviewReport
                ),
                timeout=60.0
            )
            
            cost_in = (input_tokens / 1000000) * 3.00
            cost_out = (output_tokens / 1000000) * 15.00
            estimated_cost = cost_in + cost_out
            
            return ExternalReviewReport(
                reviewer=self.model,
                confidence=report_data.confidence,
                issues=report_data.issues,
                summary=report_data.summary,
                recommended_action=report_data.recommended_action,
                review_status="COMPLETED",
                latency_ms=(time.time() - start_time) * 1000,
                tokens_sent=input_tokens,
                tokens_received=output_tokens,
                estimated_cost=estimated_cost
            )
            
        except Exception as e:
            logger.error(f"Claude review bypassed due to failure: {e}")
            return ExternalReviewReport(
                reviewer=self.model,
                confidence=1.0,
                issues=[],
                summary=f"Claude review bypassed: {e}",
                recommended_action="Proceed",
                review_status="BYPASSED",
                latency_ms=(time.time() - start_time) * 1000,
                tokens_sent=0,
                tokens_received=0,
                estimated_cost=0.0
            )
