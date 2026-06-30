"""Phase 17 — MCP provider tests (mocked transport). Offline and deterministic."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agent.config import settings
from agent.plugins import MCPProvider, PluginManager, PluginState
from agent.plugins.mcp_provider import parse_mcp_servers
from agent.state.agent_state import AgentState
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry


class MockMCPClient:
    """Simulates an MCP server for tests — no network."""

    def __init__(self, tools: List[Dict[str, Any]] | None = None):
        self._tools = tools or [{
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        }]
        self.calls: List[tuple] = []

    async def list_tools(self, server: str) -> List[Dict[str, Any]]:
        return list(self._tools)

    async def call_tool(self, server: str, tool_name: str, args: Dict[str, Any], state) -> ToolResult:
        self.calls.append((server, tool_name, args))
        total = args.get("a", 0) + args.get("b", 0)
        state.add_observation(f"mcp add={total}")
        return ToolResult(ok=True, summary=str(total), data={"result": total})


class TestMCPProvider:
    def test_parse_mcp_servers(self):
        assert parse_mcp_servers("") == []
        assert parse_mcp_servers("stdio://local, http://x") == ["stdio://local", "http://x"]

    @pytest.mark.asyncio
    async def test_mocked_mcp_tools_register(self, monkeypatch):
        monkeypatch.setattr(settings, "mcp_enabled", True)
        monkeypatch.setattr(settings, "mcp_servers", "stdio://test-server")
        monkeypatch.setattr(settings, "offline_only", False)

        client = MockMCPClient()
        reg = ToolRegistry()
        mgr = PluginManager(reg, providers=[MCPProvider(client=client)])
        count = await mgr.load_all()

        assert count == 1
        assert reg.has("mcp_add")
        tool = reg.get("mcp_add")
        assert "Add two numbers" in tool.description

    @pytest.mark.asyncio
    async def test_mcp_tool_receives_agentstate(self, monkeypatch):
        monkeypatch.setattr(settings, "mcp_enabled", True)
        monkeypatch.setattr(settings, "mcp_servers", "stdio://test-server")

        client = MockMCPClient()
        reg = ToolRegistry()
        mgr = PluginManager(reg, providers=[MCPProvider(client=client)])
        await mgr.load_all()

        state = AgentState(user_request="math")
        result = await reg.get("mcp_add").run(state, a=2, b=3)
        assert result.ok is True
        assert result.summary == "5"
        assert client.calls == [("stdio://test-server", "add", {"a": 2, "b": 3})]
        assert any("mcp add=5" in o.note for o in state.observations)

    @pytest.mark.asyncio
    async def test_mcp_disabled_system_unaffected(self, monkeypatch):
        monkeypatch.setattr(settings, "mcp_enabled", False)
        monkeypatch.setattr(settings, "mcp_servers", "stdio://test-server")

        client = MockMCPClient()
        reg = ToolRegistry()
        mgr = PluginManager(reg, providers=[MCPProvider(client=client)])
        count = await mgr.load_all()

        assert count == 0
        assert not reg.has("mcp_add")
        info = mgr.list_plugins()[0]
        assert info.state == PluginState.DISABLED

    @pytest.mark.asyncio
    async def test_remote_mcp_skipped_when_offline(self, monkeypatch):
        monkeypatch.setattr(settings, "mcp_enabled", True)
        monkeypatch.setattr(settings, "mcp_servers", "https://remote.example/mcp")
        monkeypatch.setattr(settings, "offline_only", True)

        client = MockMCPClient()
        reg = ToolRegistry()
        mgr = PluginManager(reg, providers=[MCPProvider(client=client)])
        count = await mgr.load_all()

        assert count == 0

    @pytest.mark.asyncio
    async def test_local_stdio_mcp_allowed_when_offline(self, monkeypatch):
        monkeypatch.setattr(settings, "mcp_enabled", True)
        monkeypatch.setattr(settings, "mcp_servers", "stdio://local-mcp")
        monkeypatch.setattr(settings, "offline_only", True)

        client = MockMCPClient()
        reg = ToolRegistry()
        mgr = PluginManager(reg, providers=[MCPProvider(client=client)])
        count = await mgr.load_all()

        assert count == 1
        assert reg.has("mcp_add")
