from enum import Enum
from typing import List
from pydantic import BaseModel

class ReviewSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class ReviewCategory(str, Enum):
    SCOPE_DRIFT = "SCOPE_DRIFT"
    MISSING_FILE = "MISSING_FILE"
    LIKELY_BUG = "LIKELY_BUG"
    VALIDATION_GAP = "VALIDATION_GAP"
    CONSTRAINT_RISK = "CONSTRAINT_RISK"
    ARCHITECTURE_RISK = "ARCHITECTURE_RISK"

class ReviewIssue(BaseModel):
    severity: ReviewSeverity
    category: ReviewCategory
    description: str
    issue_confidence: float

class ExternalReviewReport(BaseModel):
    reviewer: str
    confidence: float
    issues: List[ReviewIssue]
    summary: str
    recommended_action: str
    review_status: str
    latency_ms: float
    tokens_sent: int
    tokens_received: int
    estimated_cost: float
