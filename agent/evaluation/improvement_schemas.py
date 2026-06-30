from typing import List, Dict, Literal
from pydantic import BaseModel

TaxonomyClassification = Literal[
    "PLANNER_SCHEMA_ERROR",
    "CONSTRAINT_EXTRACTION_FAILURE",
    "CONTEXT_TRUNCATION",
    "MISSING_EVIDENCE",
    "EMPTY_PATCH",
    "DUPLICATE_PATCH",
    "SYNTAX_ERROR",
    "PATCH_VALIDATION_FAILURE",
    "CONFLICTING_OPERATIONS",
    "BUILD_FAILURE",
    "LINT_FAILURE",
    "TEST_FAILURE",
    "CONSTRAINT_VIOLATION",
    "ROLLBACK_TRIGGERED",
    "TIMEOUT",
    "UNHANDLED_EXCEPTION",
    "UNKNOWN"
]

RootCauseAttribution = Literal[
    "Planning",
    "Retrieval",
    "Coding",
    "Validation",
    "Safety",
    "Repair",
    "Unknown"
]

class FailureAnalysis(BaseModel):
    task_id: str
    taxonomies: List[TaxonomyClassification]
    root_cause: RootCauseAttribution
    explanation: str

class Recommendation(BaseModel):
    issue: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    suggestion: str

class HealthScore(BaseModel):
    raw_metrics: Dict[str, float]
    normalized_metrics: Dict[str, float]
    final_score: float

class AnalysisResult(BaseModel):
    benchmark_run: str
    timestamp: str
    health_score: HealthScore
    failure_analyses: List[FailureAnalysis]
    taxonomy_counts: Dict[TaxonomyClassification, int]
    root_cause_counts: Dict[RootCauseAttribution, int]
    recommendations: List[Recommendation]

class HistoricalRun(BaseModel):
    timestamp: str
    benchmark_run: str
    suite_name: str
    final_score: float
    taxonomy_counts: Dict[str, int]
    root_cause_counts: Dict[str, int]
