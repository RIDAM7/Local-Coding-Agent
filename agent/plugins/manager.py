"""Phase 17 — Plugin Manager.

Owns discovery, registration, lifecycle, permissions, and capability reporting.
Every provider implements :class:`ToolProvider`; discovered tools register into
the **same** Phase 10 :class:`ToolRegistry` — no duplicate registry or
execution path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from agent.config import logger, settings
from agent.plugins.base import PluginInfo, PluginState, ToolProvider
from agent.plugins.builtin_provider import BuiltinPluginProvider
from agent.plugins.community_provider import CommunityPluginProvider
from agent.plugins.mcp_provider import MCPProvider
from agent.plugins.permissions import PermissionChecker
from agent.plugins.project_provider import ProjectPluginProvider
from agent.tools.registry import ToolRegistry


class PluginManager:
    """Discover, register, and manage plugin tool providers."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        workspace: Path | None = None,
        permission_checker: PermissionChecker | None = None,
        providers: List[ToolProvider] | None = None,
    ):
        self.registry = registry
        self._workspace = workspace or settings.get_workspace_path()
        self._permissions = permission_checker or PermissionChecker()
        self._providers: Dict[str, ToolProvider] = {}
        self._info: Dict[str, PluginInfo] = {}
        self._tool_sources: Dict[str, str] = {}  # tool_name -> provider_name
        self._loaded = False

        if providers is not None:
            for p in providers:
                self.register_provider(p)
        else:
            self.register_provider(BuiltinPluginProvider(self._permissions))
            self.register_provider(ProjectPluginProvider(self._workspace, self._permissions))
            self.register_provider(CommunityPluginProvider(self._permissions))
            self.register_provider(MCPProvider(permission_checker=self._permissions))

    def register_provider(self, provider: ToolProvider) -> None:
        name = provider.metadata.name
        self._providers[name] = provider
        self._info[name] = PluginInfo(metadata=provider.metadata, state=PluginState.UNLOADED)

    async def load_all(self) -> int:
        """Discover and register tools from every enabled provider. Returns count."""
        if self._loaded:
            return len(self._tool_sources)

        count = 0
        for name, provider in self._providers.items():
            count += await self._load_provider(name, provider)
        self._loaded = True
        return count

    async def _load_provider(self, name: str, provider: ToolProvider) -> int:
        info = self._info[name]

        if provider.requires_network() and settings.offline_only:
            info.state = PluginState.DISABLED
            logger.info("PluginManager: disabled network provider '%s' (OFFLINE_ONLY)", name)
            return 0

        if name == "community" and not settings.plugins_enabled:
            info.state = PluginState.DISABLED
            return 0

        if name == "mcp" and not settings.mcp_enabled:
            info.state = PluginState.DISABLED
            return 0

        try:
            tools = await provider.discover()
        except Exception as e:
            logger.warning("PluginManager: discover failed for '%s': %s", name, e)
            info.state = PluginState.DISABLED
            return 0

        registered = 0
        for tool in tools:
            if self.registry.has(tool.name):
                logger.warning("PluginManager: skipping duplicate tool '%s' from '%s'", tool.name, name)
                continue
            self.registry.register(tool)
            self._tool_sources[tool.name] = name
            registered += 1

        info.tool_names = sorted(
            t for t, src in self._tool_sources.items() if src == name
        )
        info.state = PluginState.ENABLED if registered else PluginState.LOADED
        logger.info("PluginManager: provider '%s' registered %d tool(s)", name, registered)
        return registered

    async def enable(self, provider_name: str) -> int:
        """Re-enable a disabled provider and register its tools."""
        provider = self._providers.get(provider_name)
        if provider is None:
            return 0
        return await self._load_provider(provider_name, provider)

    def disable(self, provider_name: str) -> None:
        """Disable a provider and unregister its tools from the registry."""
        info = self._info.get(provider_name)
        if info is None:
            return
        for tool_name in list(info.tool_names):
            self.registry.unregister(tool_name)
            self._tool_sources.pop(tool_name, None)
        info.tool_names = []
        info.state = PluginState.DISABLED

    def unload(self, provider_name: str) -> None:
        """Disable and mark provider as unloaded."""
        self.disable(provider_name)
        info = self._info.get(provider_name)
        if info:
            info.state = PluginState.UNLOADED

    def list_plugins(self) -> List[PluginInfo]:
        return list(self._info.values())

    def capabilities(self) -> Dict[str, object]:
        """Structured capability report for observability / debugging."""
        plugins = []
        for info in self._info.values():
            plugins.append({
                "name": info.metadata.name,
                "source": info.metadata.source.value,
                "state": info.state.value,
                "tools": list(info.tool_names),
                "requires_network": info.metadata.requires_network,
            })
        return {
            "loaded": self._loaded,
            "plugin_count": len(self._info),
            "plugin_tool_count": len(self._tool_sources),
            "plugins": plugins,
        }

    def is_plugin_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_sources

    def provider_for(self, tool_name: str) -> Optional[str]:
        return self._tool_sources.get(tool_name)
