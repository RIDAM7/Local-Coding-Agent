from typing import List, Optional
from pydantic import BaseModel, Field

class BenchmarkTask(BaseModel):
    id: str = Field(..., description="Unique identifier for the benchmark")
    name: str = Field(..., description="Human-readable name")
    task: str = Field(..., description="The exact task prompt for the agent")
    expected_status: str = Field("SUCCESS", description="Expected final status")
    setup_script: Optional[str] = Field(None, description="Optional script to run before the benchmark")
    expected_files_modified: Optional[List[str]] = Field(None, description="List of expected files to be modified")
    max_runtime_seconds: Optional[int] = Field(120, description="Max runtime before timeout")

class BenchmarkResult(BaseModel):
    task_id: str
    success: bool
    repair_triggered: bool
    repair_attempts: int
    rollback_triggered: bool
    constraint_violations: int
    execution_time: float
    patch_count: int
    validation_failures: int
    planner_failures: int
    coder_failures: int
    report_path: Optional[str] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    review_decision: Optional[str] = None
    reflection_triggered: bool = False
    reflection_retry_used: bool = False
    reflection_result: str = "UNKNOWN"
    
    routing_cause: Optional[str] = None
    execution_outcome: Optional[str] = None
    
    claude_review_triggered: bool = False
    claude_review_latency_ms: float = 0.0
    claude_issue_count: int = 0

class BenchmarkSuiteResult(BaseModel):
    suite_name: str
    timestamp: str
    total_tasks: int
    success_rate: float
    repair_trigger_rate: float
    avg_repair_attempts: float
    rollback_rate: float
    total_constraint_violations: int
    avg_execution_time: float
    results: List[BenchmarkResult]

class BenchmarkComparison(BaseModel):
    base_run: str
    compare_run: str
    delta_success_rate: float
    delta_repair_trigger_rate: float
    delta_avg_repair_attempts: float
    delta_rollback_rate: float
    delta_total_constraint_violations: int
    delta_avg_execution_time: float
