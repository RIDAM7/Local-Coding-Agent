from typing import List
from pydantic import BaseModel

class ReflectionOutcome(BaseModel):
    reflection_result: str
    validation_failed: bool
    validation_categories: List[str]
    was_correct: bool

class ReflectionAccuracyReport(BaseModel):
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    precision: float
    recall: float
    f1_score: float

class CategoryAccuracy(BaseModel):
    category: str
    predictions: int
    correct: int
    accuracy: float
    status: str
