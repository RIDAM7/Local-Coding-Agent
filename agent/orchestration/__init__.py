"""Phase 16 — Agent Orchestration Layer (MVP 16A).

The Coordinator owns all orchestration for a multi-part task: task decomposition,
worker assignment, sequential scheduling, dependency handling, result merging,
and validation. Workers are stateless and specialized — they receive a scoped
unit of work, execute it via the existing governed engine, and return a result.
"""

from agent.orchestration.coordinator import Coordinator
from agent.orchestration.decomposer import decompose
from agent.orchestration.worker import Worker, WorkerResult, WorkerSpec
from agent.orchestration.scheduler import Scheduler
from agent.orchestration.merger import Merger

__all__ = [
    "Coordinator",
    "decompose",
    "Worker",
    "WorkerResult",
    "WorkerSpec",
    "Scheduler",
    "Merger",
]
