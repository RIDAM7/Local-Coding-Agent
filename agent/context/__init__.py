"""Phase 9 — Context Engine (Repository Understanding).

A proactive understanding layer that sits ON TOP of ``agent/retrieval`` (it reuses
the tree-sitter indexer, symbol index, and repository map — it does not reimplement
indexing). It produces a durable, structured :class:`ContextBundle` describing the
whole repository (tech stack, frameworks, entry points, conventions, dependency
map, architecture summary) before the agent plans.

100% local; never a network call. The bundle is returned as a standalone artifact
so Phase 10 can later assign it to ``AgentState.loaded_context`` without redesign.
"""

from agent.context.engine import ContextEngine, build_context_bundle
from agent.context.schemas import (
    ArchitectureSummary,
    ContextBundle,
    Conventions,
    EntryPoint,
    ScanResult,
    TechStack,
)

__all__ = [
    "ContextEngine",
    "build_context_bundle",
    "ContextBundle",
    "ScanResult",
    "TechStack",
    "EntryPoint",
    "Conventions",
    "ArchitectureSummary",
]
