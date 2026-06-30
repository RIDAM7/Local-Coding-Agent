from enum import Enum
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field
from agent.review.schemas import ConfidenceReport
from agent.reflection.schemas import ReflectionReport

# Core Task
class Task(BaseModel):
    description: str = Field(..., description="The natural language task description.")

# Planner Schemas
class PlanStep(BaseModel):
    id: int = Field(..., description="Step number")
    description: str = Field(..., description="Detailed description of the step")
    expected_output: str = Field(..., description="What the expected output or state is after this step")

class Plan(BaseModel):
    goal: str = Field(..., description="The overall goal of the task")
    summary: str = Field(..., description="A brief summary of the plan")
    steps: List[PlanStep] = Field(..., description="List of steps to execute the plan")

# Coder Schemas
class FileOperation(BaseModel):
    type: Literal["create_file", "update_file", "delete_file", "search_replace"] = Field(..., description="Type of operation")
    path: str = Field(..., description="Path to the file relative to the workspace root")
    content: Optional[str] = Field(None, description="The full content to write to the file (required for create/update)")
    # Phase 7A: diff-based editing. For type == "search_replace", `search` must
    # match the current file content EXACTLY ONCE; it is replaced by `replace`.
    search: Optional[str] = Field(None, description="For search_replace: the exact text block to find (must match exactly once)")
    replace: Optional[str] = Field(None, description="For search_replace: the text that replaces the matched search block")

class Patch(BaseModel):
    operations: List[FileOperation] = Field(default_factory=list, description="List of file operations to perform")
    commands: List[str] = Field(default_factory=list, description="List of shell commands to run after applying operations")

# Execution Schemas
class CommandExecution(BaseModel):
    command: str = Field(..., description="The command executed")
    stdout: str = Field(..., description="Standard output")
    stderr: str = Field(..., description="Standard error")
    exit_code: int = Field(..., description="Exit code of the command")
    duration: float = Field(..., description="Duration in seconds")

# Retrieval Schemas
class Symbol(BaseModel):
    name: str
    type: Literal["class", "function", "method", "interface", "export", "struct", "enum", "trait"]
    file: str
    line_start: int
    line_end: int
    signature: str = ""

class RepositoryComponent(BaseModel):
    type: str
    file: str
    framework_pattern: str

class RepositoryMapData(BaseModel):
    files: List[str]
    file_types: Dict[str, int]
    components: List[RepositoryComponent]

class IndexMetadata(BaseModel):
    last_indexed: str
    total_files_indexed: int
    file_hashes: Dict[str, str]

class RetrievalResult(BaseModel):
    file: str
    score: float
    evidence: List[str]
    matched_symbols: List[Symbol]

class RetrievedContext(BaseModel):
    results: List[RetrievalResult]
    total_files: int
    total_chars: int

# Validation Schemas
class PatchValidationResult(BaseModel):
    is_valid: bool
    modified_patch: Patch
    errors: List[str]
    warnings: List[str]

class ValidationDiagnostic(BaseModel):
    stage: Literal["BUILD", "LINT", "TEST"]
    command: str
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration: float

class ValidationReport(BaseModel):
    patch_validation: Optional[PatchValidationResult] = None
    build_result: Optional[ValidationDiagnostic] = None
    lint_result: Optional[ValidationDiagnostic] = None
    test_result: Optional[ValidationDiagnostic] = None

# Constraint Schemas
class Constraint(BaseModel):
    type: Literal["PROTECTED_PATH", "ALLOWLIST_PATH", "NO_DELETE"]
    patterns: Optional[List[str]] = None

class ConstraintExtractionResult(BaseModel):
    success: bool
    constraints: List[Constraint]

class RepairScope(BaseModel):
    allowed_paths: List[str]

class ConstraintValidationResult(BaseModel):
    is_valid: bool
    violations: List[str]

# Repair Schemas
class StructuredDiagnostics(BaseModel):
    build_errors: str
    lint_errors: str
    test_errors: str
    failed_files: List[str]

class NormalizedDiagnostic(BaseModel):
    classification: Literal["BUILD_FAILURE", "LINT_FAILURE", "TEST_FAILURE", "MIXED_FAILURE", "NONE"]
    primary_error_message: str
    suspected_files: List[str]

class RepairContext(BaseModel):
    original_task: str
    diagnostics: StructuredDiagnostics
    normalized_diagnostic: NormalizedDiagnostic
    retrieved_context: RetrievedContext
    constraints: List[Constraint] = []
    repair_scope: Optional[RepairScope] = None

class RepairPatch(Patch):
    explanation: str = Field(..., description="Explanation of why this repair should work.")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")

class RepairResult(BaseModel):
    attempt_number: int
    classification: str
    patch_applied: Optional[RepairPatch] = None
    validation_result: Optional[ValidationReport] = None
    success: bool
    rollback_triggered: bool = False

class RepairMetrics(BaseModel):
    total_attempts: int
    resolved_in_attempt: Optional[int] = None
    rollback_triggered: bool = False

# Routing and Execution Schemas
class RoutingCause(str, Enum):
    CLAUDE_REVIEWED = "CLAUDE_REVIEWED"
    SKIPPED_HIGH_CONFIDENCE = "SKIPPED_HIGH_CONFIDENCE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    PAYLOAD_OVERFLOW = "PAYLOAD_OVERFLOW"

class ExecutionOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    FAIL_CLOSED = "FAIL_CLOSED"

class ArbitrationDecision(str, Enum):
    APPROVE_VALIDATED = "APPROVE_VALIDATED"
    REJECT_VALIDATION_FAILED = "REJECT_VALIDATION_FAILED"
    APPROVE_CLAUDE_OVERRIDDEN = "APPROVE_CLAUDE_OVERRIDDEN"

class ArbitrationReason(str, Enum):
    VALIDATION_IS_AUTHORITATIVE = "VALIDATION_IS_AUTHORITATIVE"
    CLAUDE_HALLUCINATION_OVERRIDDEN = "CLAUDE_HALLUCINATION_OVERRIDDEN"
    UNANIMOUS_APPROVAL = "UNANIMOUS_APPROVAL"
    UNANIMOUS_REJECTION = "UNANIMOUS_REJECTION"

class ArbitrationReport(BaseModel):
    decision: ArbitrationDecision
    reason: ArbitrationReason
    explanation: str
    overridden_systems: List[str]

# Reporting Schemas
# Cost / token telemetry (Phase 7C)
class RoleUsage(BaseModel):
    role: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost: float = 0.0

class CostSummary(BaseModel):
    per_role: List[RoleUsage] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_est_cost: float = 0.0

class Report(BaseModel):
    task: str = Field(..., description="The original task")
    plan: Optional[Plan] = Field(None, description="The plan generated by the planner")
    retrieved_files: List[str] = Field(default_factory=list, description="List of files retrieved from index")
    retrieved_symbols: List[str] = Field(default_factory=list, description="List of symbols retrieved from index")
    validation_report: Optional[ValidationReport] = Field(None, description="Detailed validation results")
    repair_history: List[RepairResult] = Field(default_factory=list, description="History of repair attempts")
    repair_metrics: Optional[RepairMetrics] = Field(None, description="Metrics summarizing repair loop")
    files_modified: List[str] = Field(default_factory=list, description="List of files that were modified/created/deleted")
    commands_executed: List[CommandExecution] = Field(default_factory=list, description="List of commands that were actually executed")
    proposed_commands: List[str] = Field(default_factory=list, description="Commands the coder proposed (shown as not-executed when EXECUTE_COMMANDS is off)")
    blocked_commands: List[str] = Field(default_factory=list, description="Commands refused by the safety denylist (never executed)")
    cost_summary: Optional[CostSummary] = Field(None, description="Per-role token usage + estimated cloud cost (Phase 7C)")
    git_branch: Optional[str] = Field(None, description="Git task branch created for this run, if GIT_INTEGRATION is on")
    git_commit: Optional[str] = Field(None, description="Git commit hash created on a successful run, if any")
    execution_results: str = Field(..., description="Summary of execution results")
    final_status: Optional[str] = Field(None, description="Legacy final status")
    routing_cause: Optional[RoutingCause] = Field(None, description="The deterministic reason for routing")
    execution_outcome: Optional[ExecutionOutcome] = Field(None, description="The final outcome of the execution")
    timestamp: str = Field(..., description="ISO formatted timestamp of the report")
    confidence_report: Optional[ConfidenceReport] = Field(None, description="Detailed confidence scoring results")
    review_decision: Optional[str] = Field(None, description="Review decision outcome")
    reflection_report: Optional[ReflectionReport] = Field(None, description="Reflection report payload")
    reflection_triggered: bool = Field(False, description="Whether reflection was executed")
    reflection_retry_used: bool = Field(False, description="Whether a single reflection retry was used")
    reflection_result: Optional[str] = Field(None, description="Final reflection result (PASS/WARNING/FAIL)")
    external_review_report: Optional[Any] = Field(None, description="External review report payload")
    
    # Arbitration Telemetry
    arbitration_report: Optional[ArbitrationReport] = Field(None, description="The final arbitration decision")
    claude_override_count: int = Field(0, description="Times Claude was overridden")
    reflection_override_count: int = Field(0, description="Times Reflection was overridden")
    arbitration_triggered_count: int = Field(0, description="Times any system was overridden")
    claude_false_approval_count: int = Field(0, description="Times Claude approved but Validation failed")
