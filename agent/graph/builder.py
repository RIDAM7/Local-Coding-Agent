"""Phase 13 - MVP repository graph builder.

Builds import/module edges for Python and JS/TS using the existing Phase 9
repository scanner and symbol index. This is intentionally not a language server
and does not attempt call graph analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from agent.config import logger, settings
from agent.context.scanner import RepositoryScanner
from agent.graph.schemas import GraphEdge, GraphModule, RepositoryGraph
from agent.graph.store import GraphStore, resolve_graph_dir
from agent.models.schemas import Symbol


PY_EXT = ".py"
JS_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")


class GraphBuilder:
    def __init__(
        self,
        workspace_path: str | Path | None = None,
        *,
        graph_dir: str | Path | None = None,
        scanner: RepositoryScanner | None = None,
    ):
        self.workspace = Path(workspace_path).resolve() if workspace_path else settings.get_workspace_path()
        self.graph_dir = resolve_graph_dir(self.workspace, graph_dir)
        self.scanner = scanner or RepositoryScanner()
        self.store = GraphStore(self.graph_dir)

    def build(self, *, use_cache: bool = True, force: bool = False) -> RepositoryGraph:
        fingerprint = self.scanner.fingerprint(self.workspace)
        if use_cache and not force:
            cached = self.store.load()
            if cached is not None and cached.fingerprint == fingerprint:
                logger.info("Repository graph: cache hit.")
                return cached

        scan = self.scanner.scan(self.workspace)
        return self.build_from_scan(scan)

    def build_from_scan(self, scan) -> RepositoryGraph:
        symbols = self.scanner.sym_idx.load(str(self.workspace / "index")) or []
        graph = self._from_scan(scan.files, scan.fingerprint, symbols)
        self.store.save(graph)
        return graph

    def _from_scan(self, files: Iterable[str], fingerprint: str,
                   symbols: List[Symbol]) -> RepositoryGraph:
        files = [p.replace("\\", "/") for p in files]
        file_set = set(files)
        py_modules = self._python_module_index(files)
        symbol_owners = self._symbol_owners(symbols)
        modules: Dict[str, GraphModule] = {}
        edges: List[GraphEdge] = []

        for rel in sorted(files):
            language = self._language(rel)
            if not language:
                continue
            imports = self._imports(rel)
            dependencies: List[str] = []
            for raw in imports:
                target = self._resolve_import(rel, raw, file_set, py_modules)
                if target and target != rel:
                    dependencies.append(target)
                    edges.append(GraphEdge(source=rel, target=target, import_name=raw))
            modules[rel] = GraphModule(
                path=rel,
                language=language,
                imports=sorted(set(imports)),
                dependencies=sorted(set(dependencies)),
                symbols=sorted(symbol_owners.get(rel, [])),
            )

        return RepositoryGraph(
            root=str(self.workspace),
            fingerprint=fingerprint,
            modules=modules,
            edges=sorted(edges, key=lambda e: (e.source, e.target, e.import_name)),
            symbol_owners=symbol_owners,
        )

    @staticmethod
    def _language(rel: str) -> str:
        if rel.endswith(PY_EXT):
            return "python"
        if rel.endswith(JS_EXTS):
            return "javascript"
        return ""

    _PY_IMPORT = re.compile(
        r"^\s*(?:from\s+([\w\.]+)\s+import\s+[\w\*,\s]+|import\s+([\w\.]+))",
        re.M,
    )
    _JS_IMPORT = re.compile(
        r"""(?:import\s+(?:[^'"]+?\s+from\s+)?|export\s+[^'"]+?\s+from\s+|require\(\s*)['"]([^'"]+)['"]"""
    )

    def _imports(self, rel: str) -> List[str]:
        path = self.workspace / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        if rel.endswith(PY_EXT):
            return [m.group(1) or m.group(2) for m in self._PY_IMPORT.finditer(text)
                    if (m.group(1) or m.group(2))]
        if rel.endswith(JS_EXTS):
            return [m.group(1) for m in self._JS_IMPORT.finditer(text) if m.group(1)]
        return []

    def _resolve_import(
        self,
        source: str,
        raw: str,
        file_set: set[str],
        py_modules: Dict[str, str],
    ) -> Optional[str]:
        if source.endswith(PY_EXT):
            return self._resolve_py(raw, py_modules)
        if source.endswith(JS_EXTS):
            return self._resolve_js(source, raw, file_set)
        return None

    @staticmethod
    def _python_module_index(files: Iterable[str]) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for rel in files:
            if not rel.endswith(PY_EXT):
                continue
            stem = rel[:-3].replace("/", ".")
            if stem.endswith(".__init__"):
                index[stem[: -len(".__init__")]] = rel
            index[stem] = rel
            index.setdefault(stem.rsplit(".", 1)[-1], rel)
        return index

    @staticmethod
    def _resolve_py(raw: str, py_modules: Dict[str, str]) -> Optional[str]:
        parts = raw.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in py_modules:
                return py_modules[candidate]
        return None

    @staticmethod
    def _resolve_js(source: str, raw: str, file_set: set[str]) -> Optional[str]:
        if not raw.startswith("."):
            return None
        base = (Path(source).parent / raw).as_posix()
        candidates = [base]
        candidates.extend(base + ext for ext in JS_EXTS)
        candidates.extend(f"{base}/index{ext}" for ext in JS_EXTS)
        for candidate in candidates:
            normalized = candidate.replace("\\", "/")
            if normalized in file_set:
                return normalized
        return None

    @staticmethod
    def _symbol_owners(symbols: Iterable[Symbol]) -> Dict[str, List[str]]:
        owners: Dict[str, List[str]] = {}
        for sym in symbols:
            owners.setdefault(sym.file, []).append(sym.name)
        return {path: sorted(set(names)) for path, names in owners.items()}
