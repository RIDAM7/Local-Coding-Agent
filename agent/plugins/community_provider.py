"""Phase 17 — Community plugin provider.

Loads third-party plugins from ``PLUGIN_DIR`` and optional setuptools entry
points (``localcli.plugins``). Gated on ``PLUGINS_ENABLED`` (default off).
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import List

from agent.config import logger, settings
from agent.plugins.base import PluginMetadata, PluginSource, ToolProvider
from agent.plugins.permissions import PermissionChecker
from agent.tools.base import Tool


def _load_module(path: Path):
    name = f"localcli_community_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class CommunityPluginProvider(ToolProvider):
    """Third-party plugins from PLUGIN_DIR and entry points."""

    def __init__(self, permission_checker: PermissionChecker | None = None, plugin_dir: Path | None = None):
        self._permissions = permission_checker or PermissionChecker()
        raw = plugin_dir or Path(settings.plugin_dir or ".localcli/community_plugins")
        self._plugin_dir = raw if raw.is_absolute() else settings.get_workspace_path() / raw

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="community",
            version="0.1.0",
            description="Community plugins (PLUGIN_DIR / entry points)",
            source=PluginSource.COMMUNITY,
            requires_network=False,
        )

    async def discover(self) -> List[Tool]:
        if not settings.plugins_enabled:
            return []

        tools: List[Tool] = []
        tools.extend(await self._discover_from_dir())
        tools.extend(await self._discover_from_entry_points())
        return tools

    async def _discover_from_dir(self) -> List[Tool]:
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
                logger.warning("CommunityPluginProvider: failed to load %s: %s", path.name, e)
        return tools

    async def _discover_from_entry_points(self) -> List[Tool]:
        tools: List[Tool] = []
        try:
            eps = entry_points(group="localcli.plugins")
        except TypeError:
            eps = entry_points().get("localcli.plugins", [])
        for ep in eps:
            try:
                factory = ep.load()
                discovered = factory(permission_checker=self._permissions)
                if discovered:
                    tools.extend(discovered)
            except Exception as e:
                logger.warning("CommunityPluginProvider: entry point %s failed: %s", ep.name, e)
        return tools
