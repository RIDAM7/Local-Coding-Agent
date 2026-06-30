"""Phase 9 — Repository scanner.

Drives the repo scan by *reusing* the existing retrieval indexers — it never
reimplements indexing. It leans on :class:`RepositoryMap` for the file listing /
file-type histogram and on :class:`SymbolIndex` for the tree-sitter symbol count,
then layers on the two things the retrieval layer does not provide: reading the
ecosystem manifests and a lightweight module-level dependency map.

100% local: nothing here makes a network call.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from agent.config import logger
from agent.context.schemas import ScanResult
from agent.retrieval.repository_map import RepositoryMap
from agent.retrieval.symbol_index import SymbolIndex
from agent.retrieval.tree_sitter_indexer import TreeSitterIndexer

# Mirrors the ignore set used by the retrieval indexers, plus the context cache dir.
IGNORE_DIRS = {
    ".git", "venv", ".venv", "node_modules", "__pycache__",
    "index", "reports", ".localcli", "dist", "build", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "target",
}

# Manifest files we read for tech/framework detection (the five ecosystems).
MANIFEST_NAMES = {
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
}

# Lint/format config files used for convention inference.
LINT_CONFIG_NAMES = {
    ".ruff.toml", "ruff.toml", ".flake8", ".pylintrc", "setup.cfg",
    ".eslintrc", ".eslintrc.json", ".eslintrc.js", ".eslintrc.cjs",
    ".prettierrc", "rustfmt.toml", ".golangci.yml", ".pre-commit-config.yaml",
}

# Cap how many source files we crack open for the dependency map so a huge repo
# stays fast (MVP — incremental/whole-repo dependency analysis is an enhancement).
_MAX_DEP_FILES = 400


class RepositoryScanner:
    """Reusing-the-indexers repository scanner."""

    def __init__(self, repo_map: RepositoryMap | None = None,
                 sym_idx: SymbolIndex | None = None):
        self.repo_map = repo_map or RepositoryMap()
        self.sym_idx = sym_idx or SymbolIndex(TreeSitterIndexer())

    # --- cache fingerprint (cheap; no parsing) -------------------------------

    def fingerprint(self, workspace: str | Path) -> str:
        """A cheap content fingerprint over the file set (path + size + mtime).

        Used purely as a cache key: if it is unchanged we can serve the cached
        bundle without a full rescan. Stats only — it never reads file bodies.
        """
        ws = Path(workspace)
        entries: List[Tuple[str, int, int]] = []
        for root, dirs, files in os.walk(ws):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for name in files:
                if name.startswith("."):
                    # config dotfiles still matter for conventions; include them
                    # but keep the common churny ones out via IGNORE_DIRS above.
                    pass
                full = Path(root) / name
                try:
                    st = full.stat()
                except OSError:
                    continue
                rel = str(full.relative_to(ws)).replace("\\", "/")
                entries.append((rel, st.st_size, int(st.st_mtime)))
        entries.sort()
        h = hashlib.sha256()
        for rel, size, mtime in entries:
            h.update(f"{rel}:{size}:{mtime}\n".encode("utf-8"))
        return h.hexdigest()

    # --- full scan -----------------------------------------------------------

    def scan(self, workspace: str | Path, index_dir: str | None = None) -> ScanResult:
        """Run the full scan, reusing the retrieval indexers.

        ``index_dir`` is where :class:`SymbolIndex` caches its symbols; defaults
        to ``<workspace>/index`` (the same place the orchestrator uses), so the
        context scan and the per-task retrieval share one symbol index.
        """
        ws = Path(workspace).resolve()
        index_dir = index_dir or str(ws / "index")
        logger.info(f"Context engine: scanning repository at {ws} ...")

        # 1. File listing + histogram + heuristic components (reused indexer).
        repo_data = self.repo_map.generate(str(ws))

        # 2. Symbols (reused symbol index, incremental when possible).
        try:
            symbols = self.sym_idx.incremental_update(str(ws), index_dir)
        except Exception as e:
            logger.warning(f"Context engine: symbol indexing failed, continuing: {e}")
            symbols = []

        # 3. Manifests + lint configs (read once, fed to detectors).
        manifests = self._read_manifests(ws, repo_data.files)

        # 4. Lightweight module-level dependency map.
        dep_graph = self._build_dependency_graph(ws, repo_data.files)

        return ScanResult(
            root=str(ws),
            fingerprint=self.fingerprint(ws),
            files=repo_data.files,
            file_types=repo_data.file_types,
            manifests=manifests,
            dependency_graph=dep_graph,
            symbol_count=len(symbols),
        )

    def _read_manifests(self, ws: Path, files: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        wanted = MANIFEST_NAMES | LINT_CONFIG_NAMES
        for rel in files:
            base = rel.rsplit("/", 1)[-1]
            if base in wanted:
                full = ws / rel
                try:
                    out[rel] = full.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    logger.debug(f"Context engine: could not read manifest {rel}: {e}")
        return out

    # --- dependency map (lightweight, local) ---------------------------------

    _PY_IMPORT = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.M)
    _JS_IMPORT = re.compile(r"""(?:import\s[^'"]*from\s*|require\(\s*)['"]([^'"]+)['"]""")

    def _build_dependency_graph(self, ws: Path, files: List[str]) -> Dict[str, List[str]]:
        """Module-level import edges. MVP: Python + JS/TS, intra-repo focused."""
        graph: Dict[str, List[str]] = {}
        count = 0
        for rel in files:
            if count >= _MAX_DEP_FILES:
                break
            if rel.endswith(".py"):
                deps = self._py_deps(ws / rel)
            elif rel.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
                deps = self._js_deps(ws / rel)
            else:
                continue
            count += 1
            if deps:
                graph[rel] = sorted(set(deps))
        return graph

    def _py_deps(self, path: Path) -> List[str]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        deps = []
        for m in self._PY_IMPORT.finditer(text):
            mod = m.group(1) or m.group(2)
            if mod:
                deps.append(mod.split(".")[0])
        return deps

    def _js_deps(self, path: Path) -> List[str]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        return [m.group(1) for m in self._JS_IMPORT.finditer(text)]
