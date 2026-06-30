"""Phase 10 — AgentState: the single shared runtime object.

Created once per task by the execution engine and passed **by reference** to every
component. Each component reads what it needs and appends to its own slice — no
component owns the whole object. AgentState *aggregates* the existing typed Round 1
objects (Plan, ContextBundle, Usage, …); it does not replace them.

Design rules honored here:
- **Serializable** (JSON) with **redaction applied on serialize** — a secret can
  never land in a checkpoint. Round-trips via :meth:`to_json` / :meth:`from_json`.
- Both the ``pipeline`` and ``agent`` strategies operate on the *same* object.
- Slices owned by later phases exist now as minimal typed placeholders so those
  phases (P11 plan, P12 memory_refs, P13 evidence, P14 timeline) fill them in
  without a redesign.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent.context.schemas import ContextBundle
from agent.safety.redact import redact


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- task / plan placeholders ------------------------------------------------

class TaskMetadata(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = Field(default_factory=_utcnow)
    description: str = ""


class StepRef(BaseModel):
    """A reference to the step in flight. Filled out by Phase 11."""
    index: int = -1
    description: str = ""


class StepResult(BaseModel):
    index: int
    description: str = ""
    status: str = "pending"          # pending | running | done | failed
    summary: str = ""
    # Phase 11: each step carries its own acceptance check (derived from the
    # planner's expected_output). The Observer evaluates a step against this.
    acceptance: str = ""


class PlanRevision(BaseModel):
    """One replanning event — captures the plan evolution (Phase 11).

    Records what the remaining steps looked like before vs after a replan and
    why, so the report can show ``original plan -> revisions``. Completed steps
    are never part of a revision (the replanner only ever rewrites the tail)."""
    reason: str = ""
    remaining_before: List[str] = Field(default_factory=list)
    remaining_after: List[str] = Field(default_factory=list)
    at: str = Field(default_factory=_utcnow)


class PlanState(BaseModel):
    """Active ordered plan + status. Phase 11 owns this — it is the single
    source of truth for the plan and lives inside :class:`AgentState`, which is
    what makes Phase 15 step-level persistence come for free."""
    goal: str = ""
    summary: str = ""
    objective: str = ""
    steps: List[StepResult] = Field(default_factory=list)
    # The descriptions of the FIRST plan, kept verbatim so the report can show
    # the original plan alongside every later revision.
    original_steps: List[str] = Field(default_factory=list)
    # Append-only history of replanning events (plan evolution).
    revisions: List[PlanRevision] = Field(default_factory=list)
    replans_used: int = 0
    # The original Round 1 Plan object is stored verbatim when available so the
    # planner's typed output is never lost.
    raw_plan: Optional[Dict[str, Any]] = None


# --- memory / evidence placeholders (P12 / P13) ------------------------------

class MemoryRefs(BaseModel):
    vector_ids: List[str] = Field(default_factory=list)
    markdown_files: List[str] = Field(default_factory=list)
    summaries: List[str] = Field(default_factory=list)


class Evidence(BaseModel):
    kind: str = ""                   # search_hit | symbol | graph_impact
    detail: str = ""
    source: str = ""


# --- runtime slices ----------------------------------------------------------

class FileChange(BaseModel):
    path: str
    op: str                          # create_file | update_file | delete_file | search_replace
    summary: str = ""


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)   # redacted before storing
    status: str = "ok"               # ok | error | blocked | skipped
    result_summary: str = ""
    started_at: str = Field(default_factory=_utcnow)
    duration: float = 0.0


class Observation(BaseModel):
    note: str
    at: str = Field(default_factory=_utcnow)


class ValidationResult(BaseModel):
    stage: str                       # BUILD | LINT | TEST | CONSTRAINT | PATCH
    success: bool
    detail: str = ""


class RepairAttempt(BaseModel):
    attempt: int
    classification: str = ""
    success: bool = False


class TimelineEvent(BaseModel):
    """The live execution timeline (Phase 14 reads this)."""
    kind: str                        # tool | observation | governor | engine | validation
    message: str
    at: str = Field(default_factory=_utcnow)


class GovernorState(BaseModel):
    """Steps/tool-calls/cost/elapsed budgets + the stop reason (the leash)."""
    max_steps: int = 25
    steps_used: int = 0
    tool_call_budget: int = 0        # 0 == disabled
    tool_calls_used: int = 0
    run_budget_usd: float = 0.0      # 0 == disabled
    cost_used_usd: float = 0.0
    step_timeout_seconds: int = 120
    run_timeout_seconds: int = 0     # 0 == disabled
    # Phase 11: replanning is bounded by the same leash. 0 == disabled.
    max_replans: int = 3
    replans_used: int = 0
    started_at: str = Field(default_factory=_utcnow)
    stopped: bool = False
    stop_reason: Optional[str] = None
    # Oscillation / no-progress detection (reuses repeated-action fingerprints).
    progress_fingerprints: List[str] = Field(default_factory=list)


class FinalOutputs(BaseModel):
    status: str = "PENDING"          # SUCCESS | FAILURE | PENDING
    summary: str = ""
    report_path: Optional[str] = None
    applied_changes: List[str] = Field(default_factory=list)


# --- the central object ------------------------------------------------------

class AgentState(BaseModel):
    """The whole run, on one shared object."""

    task: TaskMetadata = Field(default_factory=TaskMetadata)
    user_request: str = ""
    execution_mode: str = "pipeline"
    objective: str = ""

    plan: PlanState = Field(default_factory=PlanState)
    current_step: StepRef = Field(default_factory=StepRef)
    completed_steps: List[StepResult] = Field(default_factory=list)

    loaded_context: Optional[ContextBundle] = None   # from Phase 9
    memory_refs: MemoryRefs = Field(default_factory=MemoryRefs)

    files_read: List[str] = Field(default_factory=list)
    files_modified: List[FileChange] = Field(default_factory=list)
    tool_history: List[ToolCall] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    repair_attempts: List[RepairAttempt] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)

    governor: GovernorState = Field(default_factory=GovernorState)
    confidence: float = 0.0
    final_outputs: FinalOutputs = Field(default_factory=FinalOutputs)

    # Capability detection result for the model running this task (recorded here
    # per the Capability Detector spec). Stored as a plain dict to stay decoupled.
    capabilities: Dict[str, Any] = Field(default_factory=dict)

    # --- convenience mutators (components append to their own slice) ---------

    def add_observation(self, note: str) -> None:
        self.observations.append(Observation(note=note))
        self.add_timeline("observation", note)

    def add_timeline(self, kind: str, message: str) -> None:
        try:
            from agent.config import settings
            if not settings.observability_enabled:
                return
        except Exception:
            pass
        self.timeline.append(TimelineEvent(kind=kind, message=message))

    def record_file_change(self, path: str, op: str, summary: str = "") -> None:
        self.files_modified.append(FileChange(path=path, op=op, summary=summary))

    # --- serialization (redaction on serialize) ------------------------------

    def to_json(self, *, redacted: bool = True, indent: int | None = 2) -> str:
        """Serialize to JSON. With ``redacted`` (default), every secret is scrubbed
        before the string leaves memory — checkpoints (Phase 15) are always safe."""
        raw = self.model_dump_json(indent=indent)
        return redact(raw) if redacted else raw

    @classmethod
    def from_json(cls, data: str) -> "AgentState":
        return cls.model_validate_json(data)
