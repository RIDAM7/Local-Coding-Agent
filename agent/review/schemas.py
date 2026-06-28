from enum import Enum
from typing import Dict
from pydantic import BaseModel

class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    MANDATORY_REVIEW = "MANDATORY_REVIEW"

class ConfidenceReport(BaseModel):
    confidence_score: float
    review_required: bool
    review_decision: ReviewDecision
    contributing_factors: Dict[str, float]

    build_score: float
    test_score: float
    lint_score: float
    repair_score: float
    memory_score: float
    file_scope_score: float
    planner_complexity_score: float

    memory_count_used: int
    files_modified_count: int
    plan_step_count: int
    constraint_violations: int
