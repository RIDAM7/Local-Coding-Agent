"""Phase 9 — context cache store.

Reads/writes/refreshes the cached Context Bundle under ``CONTEXT_DIR`` (default
``.localcli/context``, resolved relative to the scanned workspace). Outputs:

  repo_context.json     — the full machine-readable bundle (source of truth)
  architecture.md       — human-readable architecture overview
  conventions.md        — inferred conventions
  dependency_graph.json — module-level import edges

Everything written to disk is passed through the Phase 5 redaction layer so a
secret can never leak into a cached artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agent.config import logger
from agent.context.schemas import ContextBundle
from agent.context.summarizer import render_architecture_md, render_conventions_md
from agent.safety.redact import redact

REPO_CONTEXT_JSON = "repo_context.json"
ARCHITECTURE_MD = "architecture.md"
CONVENTIONS_MD = "conventions.md"
DEPENDENCY_GRAPH_JSON = "dependency_graph.json"


class ContextStore:
    """Filesystem-backed cache for the Context Bundle."""

    def __init__(self, context_dir: str | Path):
        self.dir = Path(context_dir)

    # --- read ---------------------------------------------------------------

    def load(self) -> Optional[ContextBundle]:
        target = self.dir / REPO_CONTEXT_JSON
        if not target.exists():
            return None
        try:
            return ContextBundle.model_validate_json(target.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Context engine: failed to load cached bundle: {e}")
            return None

    def is_fresh(self, fingerprint: str) -> bool:
        """True when a cached bundle exists with a matching fingerprint."""
        cached = self.load()
        return cached is not None and cached.fingerprint == fingerprint

    # --- write --------------------------------------------------------------

    def save(self, bundle: ContextBundle) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

        # Machine-readable bundle (redacted JSON body).
        self._write(REPO_CONTEXT_JSON, bundle.model_dump_json(indent=2))

        # Human-readable docs.
        self._write(ARCHITECTURE_MD, render_architecture_md(bundle))
        self._write(CONVENTIONS_MD, render_conventions_md(bundle))

        # Dependency graph (separate artifact for tooling).
        self._write(DEPENDENCY_GRAPH_JSON,
                    json.dumps(bundle.dependency_graph, indent=2, sort_keys=True))

        logger.info(f"Context engine: cached bundle written to {self.dir}")

    def _write(self, name: str, content: str) -> None:
        path = self.dir / name
        try:
            path.write_text(redact(content), encoding="utf-8")
        except Exception as e:
            logger.error(f"Context engine: failed to write {name}: {e}")
