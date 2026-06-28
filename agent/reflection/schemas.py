from enum import Enum
from typing import List
from pydantic import BaseModel

class ReflectionResult(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"

class CritiqueCategory(str, Enum):
    CONSTRAINT_VIOLATION_RISK = "CONSTRAINT_VIOLATION_RISK"
    SCOPE_DRIFT_RISK = "SCOPE_DRIFT_RISK"
    MISSING_FILE_RISK = "MISSING_FILE_RISK"
    TEST_FAILURE_RISK = "TEST_FAILURE_RISK"
    SYNTAX_RISK = "SYNTAX_RISK"
    DUPLICATE_PATCH_RISK = "DUPLICATE_PATCH_RISK"

class ReflectionCritique(BaseModel):
    category: CritiqueCategory
    explanation: str
    severity: str

class ReflectionReport(BaseModel):
    result: ReflectionResult
    critiques: List[ReflectionCritique]
    summary: str
    execution_time_ms: float
    model_name: str
