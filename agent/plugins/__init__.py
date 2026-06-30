"""Phase 17 — Plugin Architecture & Extensibility (MVP).

The Plugin Manager loads tools from builtin, project, community, and optional
MCP sources through one :class:`ToolProvider` interface. All tools register into
the shared Phase 10 :class:`ToolRegistry`.
"""

from agent.plugins.base import (
    PluginInfo,
    PluginMetadata,
    PluginSource,
    PluginState,
    PluginTool,
    ToolProvider,
)
from agent.plugins.manager import PluginManager
from agent.plugins.permissions import PermissionChecker
from agent.plugins.builtin_provider import BuiltinPluginProvider
from agent.plugins.project_provider import ProjectPluginProvider
from agent.plugins.community_provider import CommunityPluginProvider
from agent.plugins.mcp_provider import MCPProvider, MCPClient, NullMCPClient, parse_mcp_servers

__all__ = [
    "PluginManager",
    "PluginInfo",
    "PluginMetadata",
    "PluginSource",
    "PluginState",
    "PluginTool",
    "ToolProvider",
    "PermissionChecker",
    "BuiltinPluginProvider",
    "ProjectPluginProvider",
    "CommunityPluginProvider",
    "MCPProvider",
    "MCPClient",
    "NullMCPClient",
    "parse_mcp_servers",
]
