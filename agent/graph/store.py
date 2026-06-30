"""Phase 13 - local graph store."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent.config import logger, settings
from agent.graph.schemas import RepositoryGraph
from agent.safety.redact import redact

GRAPH_JSON = "graph.json"


def resolve_graph_dir(workspace: Path, graph_dir: str | Path | None = None) -> Path:
    raw = Path(graph_dir or settings.graph_dir)
    return raw if raw.is_absolute() else workspace / raw


class GraphStore:
    def __init__(self, graph_dir: str | Path):
        self.dir = Path(graph_dir)

    @property
    def path(self) -> Path:
        return self.dir / GRAPH_JSON

    def load(self) -> Optional[RepositoryGraph]:
        if not self.path.exists():
            return None
        try:
            return RepositoryGraph.model_validate_json(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Repository graph: failed to load cached graph: {e}")
            return None

    def save(self, graph: RepositoryGraph) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(redact(graph.model_dump_json(indent=2)), encoding="utf-8")

    def is_fresh(self, fingerprint: str) -> bool:
        cached = self.load()
        return cached is not None and cached.fingerprint == fingerprint
