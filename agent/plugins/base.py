"""Phase 17 — Plugin base types and ToolProvider interface.

Every plugin source implements :class:`ToolProvider` and returns standard
:class:`Tool` instances that register into the Phase 10 :class:`ToolRegistry`.
MCP is one provider implementation — not the architecture itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolResult


class PluginSource(str, Enum):
    BUILTIN = "builtin"
    PROJECT = "project"
    COMMUNITY = "community"
    MCP = "mcp"


class PluginState(str, Enum):
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"


class PluginMetadata(BaseModel):
    """Describes a plugin provider (not an individual tool)."""
    name: str
    version: str = "0.1.0"
    description: str = ""
    source: PluginSource = PluginSource.BUILTIN
    requires_network: bool = False


class PluginInfo(BaseModel):
    """Runtime view of a loaded plugin provider."""
    metadata: PluginMetadata
    state: PluginState = PluginState.UNLOADED
    tool_names: List[str] = Field(default_factory=list)


PluginHandler = Callable[..., Awaitable[ToolResult]]


class PluginTool(Tool):
    """A tool supplied by a plugin provider.

    Wraps a handler callable and enforces the shared permission model before
    execution. Still uses the base :class:`Tool.run` bookkeeping (tool_history,
    timeline, governor).
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: PluginHandler,
        provider_name: str,
        permission_checker=None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._handler = handler
        self._provider_name = provider_name
        self._permission_checker = permission_checker

    async def _execute(self, state, **kwargs) -> ToolResult:
        if self._permission_checker is not None:
            allowed, reason = self._permission_checker.check(self.name, self._provider_name)
            if not allowed:
                return ToolResult(ok=False, status="blocked", summary=reason)
        return await self._handler(state, **kwargs)


class ToolProvider(ABC):
    """Common interface for every plugin source (builtin, project, community, MCP)."""

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        ...

    def requires_network(self) -> bool:
        return self.metadata.requires_network

    @abstractmethod
    async def discover(self) -> List[Tool]:
        """Return tools this provider exposes. Called during load/enable."""
