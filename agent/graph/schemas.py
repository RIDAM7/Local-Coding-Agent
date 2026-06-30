"""Phase 13 - lightweight repository graph schemas."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: str = "imports"
    import_name: str = ""


class GraphModule(BaseModel):
    path: str
    language: str = ""
    imports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    symbols: List[str] = Field(default_factory=list)


class RepositoryGraph(BaseModel):
    root: str
    fingerprint: str
    modules: Dict[str, GraphModule] = Field(default_factory=dict)
    edges: List[GraphEdge] = Field(default_factory=list)
    symbol_owners: Dict[str, List[str]] = Field(default_factory=dict)

    def dependents_of(self, path: str) -> List[str]:
        normalized = path.replace("\\", "/")
        return sorted({edge.source for edge in self.edges if edge.target == normalized})

    def dependencies_of(self, path: str) -> List[str]:
        normalized = path.replace("\\", "/")
        module = self.modules.get(normalized)
        return sorted(module.dependencies) if module else []
