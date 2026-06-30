"""Phase 17 — Built-in plugin provider.

Ships bundled first-party plugin tools with the agent. No flag required —
always discovered when the Plugin Manager loads.
"""

from __future__ import annotations

from typing import List

from agent.plugins.base import PluginMetadata, PluginSource, PluginTool, ToolProvider
from agent.plugins.permissions import PermissionChecker
from agent.state.agent_state import AgentState
from agent.tools.base import Tool, ToolResult


async def _echo_handler(state: AgentState, *, message: str = "") -> ToolResult:
    text = message or "(empty)"
    state.add_observation(f"plugin_echo: {text}")
    return ToolResult(ok=True, summary=f"echo: {text}", data={"message": text})


class BuiltinPluginProvider(ToolProvider):
    """Bundled local plugins that ship with localcli."""

    def __init__(self, permission_checker: PermissionChecker | None = None):
        self._permissions = permission_checker or PermissionChecker()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="builtin",
            version="0.1.0",
            description="Bundled first-party plugin tools",
            source=PluginSource.BUILTIN,
            requires_network=False,
        )

    async def discover(self) -> List[Tool]:
        return [
            PluginTool(
                name="plugin_echo",
                description="Echo a message into AgentState observations (builtin plugin demo).",
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string", "description": "Text to echo"}},
                    "required": [],
                },
                handler=_echo_handler,
                provider_name=self.metadata.name,
                permission_checker=self._permissions,
            ),
        ]
