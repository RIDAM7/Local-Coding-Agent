"""Phase 13 - graph query helpers."""

from __future__ import annotations

from pathlib import Path

from agent.config import settings
from agent.graph.builder import GraphBuilder
from agent.graph.impact_analyzer import ImpactAnalyzer
from agent.graph.schemas import RepositoryGraph


def build_or_load_graph(workspace: str | Path | None = None, *, force: bool = False) -> RepositoryGraph | None:
    if not settings.repo_graph_enabled:
        return None
    return GraphBuilder(workspace).build(force=force)


def impact(workspace: str | Path | None, path: str, *, force: bool = False) -> list[str]:
    graph = build_or_load_graph(workspace, force=force)
    if graph is None:
        return []
    return ImpactAnalyzer(graph).dependents(path)
