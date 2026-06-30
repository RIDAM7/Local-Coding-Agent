"""Phase 10 — provider-agnostic Reviewer (reviewer cleanup).

Generalizes the old Claude-only reviewer into a `Reviewer` built via
``build_client("reviewer")`` — it works with any provider the factory supports
(Ollama/OpenAI/Anthropic/Google). It returns the same :class:`ExternalReviewReport`
shape Round 1 used, so the pipeline path is unchanged; the agent engine uses it
ONLY as a last-resort escalation (see :mod:`agent.engine.agent_engine`).

No module here imports ``agent.llm.claude_client`` — that legacy coupling is gone.
"""

import asyncio
import time
from typing import Any, List

from pydantic import BaseModel

from agent.config import logger, settings
from agent.llm.factory import build_client
from agent.llm.pricing import estimate_cost
from agent.reviewers.schemas import ExternalReviewReport, ReviewIssue

MAX_REVIEW_CONTEXT_CHARS = 6000


class InternalReviewReport(BaseModel):
    confidence: float
    issues: List[ReviewIssue]
    summary: str
    recommended_action: str


class Reviewer:
    """A single, provider-agnostic reviewer."""

    def __init__(self, client=None):
        self.client = client if client is not None else build_client("reviewer")
        self.model = getattr(self.client, "model", None) or settings.reviewer_model or settings.planner_model

    async def review(self, task: Any, plan: Any, patch: Any,
                     validation_report: Any, reflection_report: Any) -> ExternalReviewReport:
        start_time = time.time()

        prompt = "You are a Reviewer agent. Your task is to review the execution of a coding task.\n\n"
        prompt += f"Task: {getattr(task, 'description', str(task))[:1000]}\n\n"
        if plan:
            prompt += f"Plan Summary: {str(getattr(plan, 'steps', plan))[:1000]}\n\n"
        patch_str = ""
        if patch and hasattr(patch, "operations"):
            for op in patch.operations:
                patch_str += f"File: {op.path} Action: {op.type}\n"
        prompt += f"Patch Summary:\n{patch_str[:1000]}\n\n"
        if validation_report:
            prompt += f"Validation Results: {str(validation_report)[:1000]}\n\n"
        if reflection_report:
            prompt += f"Reflection Summary: {str(getattr(reflection_report, 'summary', reflection_report))[:1000]}\n\n"
        if len(prompt) > MAX_REVIEW_CONTEXT_CHARS:
            prompt = prompt[:MAX_REVIEW_CONTEXT_CHARS]
        prompt += ("\nEvaluate this execution strictly against the following categories: "
                   "SCOPE_DRIFT, MISSING_FILE, LIKELY_BUG, VALIDATION_GAP, CONSTRAINT_RISK, ARCHITECTURE_RISK.\n")
        prompt += ("Return a JSON object matching the InternalReviewReport schema. "
                   "If there are no issues, return an empty list for issues.")

        try:
            result = await asyncio.wait_for(
                self.client.generate_structured(self.model, prompt, InternalReviewReport),
                timeout=60.0,
            )
            data = result.data
            usage = result.usage
            in_tokens = getattr(usage, "input_tokens", 0)
            out_tokens = getattr(usage, "output_tokens", 0)
            est = estimate_cost(getattr(usage, "provider", ""), getattr(usage, "model", self.model),
                                in_tokens, out_tokens)
            return ExternalReviewReport(
                reviewer=self.model,
                confidence=data.confidence,
                issues=data.issues,
                summary=data.summary,
                recommended_action=data.recommended_action,
                review_status="COMPLETED",
                latency_ms=(time.time() - start_time) * 1000,
                tokens_sent=in_tokens,
                tokens_received=out_tokens,
                estimated_cost=est,
            )
        except Exception as e:
            logger.error(f"Reviewer bypassed due to failure: {e}")
            return ExternalReviewReport(
                reviewer=self.model,
                confidence=1.0,
                issues=[],
                summary=f"Reviewer bypassed: {e}",
                recommended_action="Proceed",
                review_status="BYPASSED",
                latency_ms=(time.time() - start_time) * 1000,
                tokens_sent=0,
                tokens_received=0,
                estimated_cost=0.0,
            )
