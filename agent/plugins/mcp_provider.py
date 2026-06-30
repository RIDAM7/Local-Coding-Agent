"""Phase 17 — MCP plugin provider (one transport, not the architecture).

MCP tools are discovered through the same :class:`ToolProvider` interface as
builtin/project/community plugins and register into the shared ToolRegistry.
Full stdio/HTTP transport is an enhancement hook — MVP uses an injectable client
so tests can mock MCP servers without network I/O.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from agent.config import logger, settings
from agent.plugins.base import PluginMetadata, PluginSource, PluginTool, ToolProvider
from agent.plugins.permissions import PermissionChecker
from agent.state.agent_state import AgentState
from agent.tools.base import Tool, ToolResult


class MCPClient(Protocol):
    """Transport seam for MCP server communication (mockable in tests)."""

    async def list_tools(self, server: str) -> List[Dict[str, Any]]:
        ...

    async def call_tool(
        self, server: str, tool_name: str, args: Dict[str, Any], state: AgentState
    ) -> ToolResult:
        ...


class NullMCPClient:
    """Default client — returns no tools (real transport is enhancement)."""

    async def list_tools(self, server: str) -> List[Dict[str, Any]]:
        return []

    async def call_tool(
        self, server: str, tool_name: str, args: Dict[str, Any], state: AgentState
    ) -> ToolResult:
        return ToolResult(ok=False, status="error", summary="MCP transport not configured")


def parse_mcp_servers(raw: str) -> List[str]:
    if not raw or not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _server_requires_network(server: str) -> bool:
    lower = server.lower()
    return lower.startswith("http://") or lower.startswith("https://")


class MCPTool(PluginTool):
    """A tool proxying a single MCP server tool."""

    def __init__(
        self,
        *,
        spec: Dict[str, Any],
        server: str,
        client: MCPClient,
        permission_checker: PermissionChecker | None,
    ):
        name = spec.get("name", "mcp_tool")
        super().__init__(
            name=f"mcp_{name}",
            description=spec.get("description", f"MCP tool from {server}"),
            parameters=spec.get(
                "parameters",
                {"type": "object", "properties": {}, "required": []},
            ),
            handler=self._make_handler(server, name, client),
            provider_name="mcp",
            permission_checker=permission_checker,
        )
        self._mcp_server = server
        self._mcp_tool_name = name

    @staticmethod
    def _make_handler(server: str, tool_name: str, client: MCPClient):
        async def _handler(state: AgentState, **kwargs) -> ToolResult:
            return await client.call_tool(server, tool_name, kwargs, state)

        return _handler


class MCPProvider(ToolProvider):
    """Optional MCP transport — disabled when ``MCP_ENABLED=false``."""

    def __init__(
        self,
        client: MCPClient | None = None,
        permission_checker: PermissionChecker | None = None,
    ):
        self._client = client or NullMCPClient()
        self._permissions = permission_checker or PermissionChecker()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="mcp",
            version="0.1.0",
            description="MCP server tools (optional transport)",
            source=PluginSource.MCP,
            requires_network=False,
        )

    def requires_network(self) -> bool:
        """True only when at least one configured server is remote."""
        return any(_server_requires_network(s) for s in parse_mcp_servers(settings.mcp_servers))

    async def discover(self) -> List[Tool]:
        if not settings.mcp_enabled:
            return []

        tools: List[Tool] = []
        for server in parse_mcp_servers(settings.mcp_servers):
            if settings.offline_only and _server_requires_network(server):
                logger.info("MCPProvider: skipping remote server %s (OFFLINE_ONLY)", server)
                continue
            try:
                specs = await self._client.list_tools(server)
            except Exception as e:
                logger.warning("MCPProvider: list_tools failed for %s: %s", server, e)
                continue
            for spec in specs:
                tools.append(
                    MCPTool(
                        spec=spec,
                        server=server,
                        client=self._client,
                        permission_checker=self._permissions,
                    )
                )
        return tools
