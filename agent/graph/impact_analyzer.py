"""Phase 13 - impact queries over the repository graph."""

from __future__ import annotations

from agent.graph.schemas import RepositoryGraph
from agent.state.agent_state import AgentState, Evidence


class ImpactAnalyzer:
    def __init__(self, graph: RepositoryGraph):
        self.graph = graph

    def dependents(self, path: str) -> list[str]:
        return self.graph.dependents_of(path)

    def dependencies(self, path: str) -> list[str]:
        return self.graph.dependencies_of(path)

    def record_impact(self, state: AgentState, path: str) -> list[str]:
        dependents = self.dependents(path)
        detail = f"{path}: {len(dependents)} dependent file(s)"
        if dependents:
            detail += " - " + ", ".join(dependents[:10])
        state.evidence.append(Evidence(kind="graph_impact", detail=detail, source=path))
        state.add_observation(f"graph impact: {detail}")
        return dependents
