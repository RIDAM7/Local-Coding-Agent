from pydantic import BaseModel
from typing import List, Dict

class CoverageResults(BaseModel):
    all_states_observed: bool
    missing_states: List[str]

class AuthorityVerification(BaseModel):
    validation_remains_authoritative: bool
    claude_never_merges_failing_code: bool
    claude_never_blocks_validated_code: bool

class ArbitrationEffectivenessMetrics(BaseModel):
    # Aggregates
    total_tasks: int
    claude_override_count: int
    reflection_override_count: int
    arbitration_triggered_count: int
    claude_false_approval_count: int
    
    # Decision Distributions & Percentages
    decision_distribution: Dict[str, int]
    decision_percentages: Dict[str, float]
    
    # Traceability Lists
    tasks_claude_overridden: List[str]
    tasks_reflection_overridden: List[str]
    tasks_validation_authoritative: List[str]
    tasks_claude_false_approval: List[str]
    tasks_fail_closed: List[str]
    
    # Validation
    coverage: CoverageResults
    authority: AuthorityVerification
