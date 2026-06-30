"""Phase 9 — Context Engine (the pipeline that produces the Context Bundle).

Pipeline:
    Repository Scan (reuse existing indexer)
      -> Tech / Framework Detection
      -> Entry-Point Discovery
      -> Convention Inference
      -> Lightweight Dependency Map
      -> Architecture Summary (planner model; skipped/falls back if no LLM)
      -> Context Bundle  ->  returned to caller

The engine is a standalone artifact producer (no global state): it returns a
:class:`ContextBundle`. Phase 10 will assign that return value to
``state.loaded_context`` — by design this needs no change here.

100% local. The only model touched is the injected planner client (optional);
with ``llm_client=None`` it still yields a fully usable machine bundle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.config import logger, settings
from agent.context import detectors
from agent.context.context_store import ContextStore
from agent.context.scanner import RepositoryScanner
from agent.context.schemas import ContextBundle
from agent.context.summarizer import summarize_architecture


def _resolve_context_dir(workspace: Path) -> Path:
    """Resolve CONTEXT_DIR. Relative paths live under the scanned workspace."""
    raw = Path(settings.context_dir)
    return raw if raw.is_absolute() else (workspace / raw)


class ContextEngine:
    """Builds (and caches) the repository Context Bundle."""

    def __init__(self, workspace_path: str | Path | None = None,
                 context_dir: str | Path | None = None,
                 scanner: RepositoryScanner | None = None):
        self.workspace = Path(workspace_path).resolve() if workspace_path \
            else settings.get_workspace_path()
        self.context_dir = Path(context_dir) if context_dir \
            else _resolve_context_dir(self.workspace)
        self.scanner = scanner or RepositoryScanner()
        self.store = ContextStore(self.context_dir)

    async def build(self, *, use_cache: bool = True, force: bool = False,
                    llm_client=None) -> ContextBundle:
        """Build or load the Context Bundle.

        - ``use_cache``/``force`` + ``CONTEXT_CACHE`` govern cache reuse. A cache
          hit returns the cached bundle WITHOUT a full rescan (only a cheap
          fingerprint walk runs).
        - ``llm_client`` (optional) is the planner client used for the prose
          architecture summary; ``None`` yields a machine-only bundle.
        """
        fingerprint = self.scanner.fingerprint(self.workspace)

        if use_cache and not force and settings.context_cache:
            cached = self.store.load()
            if cached is not None and cached.fingerprint == fingerprint:
                logger.info("Context engine: cache hit (no rescan).")
                return cached

        logger.info("Context engine: building fresh context bundle.")
        scan = self.scanner.scan(self.workspace)
        dependency_graph = scan.dependency_graph
        repository_graph = {}
        if settings.repo_graph_enabled:
            try:
                from agent.graph.builder import GraphBuilder
                graph_builder = GraphBuilder(self.workspace, scanner=self.scanner)
                cached_graph = graph_builder.store.load()
                if cached_graph is not None and cached_graph.fingerprint == scan.fingerprint:
                    graph = cached_graph
                else:
                    graph = graph_builder.build_from_scan(scan)
                dependency_graph = dict(scan.dependency_graph)
                dependency_graph.update({
                    path: module.dependencies
                    for path, module in graph.modules.items()
                    if module.dependencies
                })
                repository_graph = {
                    "path": str(graph_builder.store.path),
                    "modules": len(graph.modules),
                    "edges": len(graph.edges),
                }
            except Exception as e:
                logger.warning(f"Context engine: repository graph unavailable, using scan map: {e}")

        tech_stack = detectors.detect_tech_stack(scan.manifests)
        frameworks = detectors.detect_frameworks(tech_stack)
        entry_points = detectors.detect_entry_points(scan.files, scan.manifests)
        conventions = detectors.infer_conventions(
            scan.files, scan.file_types, scan.manifests)

        bundle = ContextBundle(
            root=scan.root,
            generated_at=datetime.now(timezone.utc).isoformat(),
            fingerprint=scan.fingerprint,
            file_count=len(scan.files),
            languages=scan.file_types,
            frameworks=frameworks,
            tech_stack=tech_stack,
            entry_points=entry_points,
            conventions=conventions,
            dependency_graph=dependency_graph,
            repository_graph=repository_graph,
            symbol_count=scan.symbol_count,
        )

        # Architecture summary — optional, local, fail-open.
        bundle.architecture_summary = await summarize_architecture(bundle, llm_client)

        if settings.context_cache:
            self.store.save(bundle)
        return bundle


async def build_context_bundle(workspace_path: str | Path | None = None,
                               *, use_cache: bool = True, force: bool = False,
                               llm_client=None) -> Optional[ContextBundle]:
    """Convenience wrapper. Returns ``None`` when the engine is disabled.

    This is the single entry point callers (the orchestrator, the CLI) use, so
    the ``CONTEXT_ENGINE_ENABLED`` gate lives in exactly one place.
    """
    if not settings.context_engine_enabled:
        return None
    engine = ContextEngine(workspace_path)
    return await engine.build(use_cache=use_cache, force=force, llm_client=llm_client)
