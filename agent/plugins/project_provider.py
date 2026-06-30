"""Phase 17 — Project-local plugin provider.

Discovers plugins shipped inside the workspace under ``.localcli/plugins/``.
Each ``*.py`` module may export ``get_tools() -> list[Tool]``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

from agent.config import logger
from agent.plugins.base import PluginMetadata, PluginSource, ToolProvider
from agent.plugins.permissions import PermissionChecker
from agent.tools.base import Tool


def _load_module(path: Path):
    name = f"localcli_project_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class ProjectPluginProvider(ToolProvider):
    """Repo-local plugins living under ``<workspace>/.localcli/plugins/``."""

    def __init__(self, workspace: Path, permission_checker: PermissionChecker | None = None):
        self._workspace = Path(workspace)
        self._permissions = permission_checker or PermissionChecker()
        self._plugin_dir = self._workspace / ".localcli" / "plugins"

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="project",
            version="0.1.0",
            description="Workspace-local plugins (.localcli/plugins/)",
            source=PluginSource.PROJECT,
            requires_network=False,
        )

    async def discover(self) -> List[Tool]:
        tools: List[Tool] = []
        if not self._plugin_dir.is_dir():
            return tools

        for path in sorted(self._plugin_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                mod = _load_module(path)
                if mod is None or not hasattr(mod, "get_tools"):
                    continue
                discovered = mod.get_tools(permission_checker=self._permissions)
                if discovered:
                    tools.extend(discovered)
            except Exception as e:
                logger.warning("ProjectPluginProvider: failed to load %s: %s", path.name, e)
        return tools
