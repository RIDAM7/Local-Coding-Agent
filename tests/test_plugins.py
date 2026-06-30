"""Phase 17 — Plugin Architecture tests. Offline and deterministic."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from agent.config import settings
from agent.engine.agent_engine import AgentEngine, AgentAction
from agent.llm.providers.base import LLMResult, Usage
from agent.plugins import (
    BuiltinPluginProvider,
    CommunityPluginProvider,
    MCPProvider,
    PluginManager,
    PluginState,
    ProjectPluginProvider,
)
from agent.plugins.permissions import PermissionChecker
from agent.state.agent_state import AgentState
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry


# --- helpers ----------------------------------------------------------------


def _registry() -> ToolRegistry:
    return ToolRegistry()


def _write_project_plugin(workspace: Path, name: str = "demo.py") -> None:
    plugin_dir = workspace / ".localcli" / "plugins"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / name).write_text(
        '''
from agent.plugins.base import PluginTool
from agent.state.agent_state import AgentState
from agent.tools.base import ToolResult

async def _handler(state: AgentState, **kwargs) -> ToolResult:
    state.add_observation("project plugin ran")
    return ToolResult(ok=True, summary="project ok")

def get_tools(permission_checker=None):
    return [
        PluginTool(
            name="project_ping",
            description="project plugin",
            parameters={"type": "object", "properties": {}},
            handler=_handler,
            provider_name="project",
            permission_checker=permission_checker,
        )
    ]
''',
        encoding="utf-8",
    )


def _write_community_plugin(plugin_dir: Path, name: str = "community_demo.py") -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / name).write_text(
        '''
from agent.plugins.base import PluginTool
from agent.state.agent_state import AgentState
from agent.tools.base import ToolResult

async def _handler(state: AgentState, **kwargs) -> ToolResult:
    state.add_observation("community plugin ran")
    return ToolResult(ok=True, summary="community ok")

def get_tools(permission_checker=None):
    return [
        PluginTool(
            name="community_ping",
            description="community plugin",
            parameters={"type": "object", "properties": {}},
            handler=_handler,
            provider_name="community",
            permission_checker=permission_checker,
        )
    ]
''',
        encoding="utf-8",
    )


class _MockMCPClient:
    async def list_tools(self, server: str) -> List[dict]:
        return [{
            "name": "greet",
            "description": "MCP greet tool",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }]

    async def call_tool(self, server, tool_name, args, state):
        state.add_observation(f"mcp:{tool_name}")
        return ToolResult(ok=True, summary=f"hello {args.get('name', 'world')}")


# --- PluginManager ----------------------------------------------------------


class TestPluginManager:
    @pytest.mark.asyncio
    async def test_builtin_discovery_and_registration(self):
        reg = _registry()
        mgr = PluginManager(reg, providers=[BuiltinPluginProvider()])
        count = await mgr.load_all()
        assert count == 1
        assert reg.has("plugin_echo")
        caps = mgr.capabilities()
        assert caps["plugin_tool_count"] == 1
        assert mgr.list_plugins()[0].state == PluginState.ENABLED

    @pytest.mark.asyncio
    async def test_project_plugin_discovery(self, tmp_path):
        _write_project_plugin(tmp_path)
        reg = _registry()
        mgr = PluginManager(
            reg,
            workspace=tmp_path,
            providers=[ProjectPluginProvider(tmp_path)],
        )
        count = await mgr.load_all()
        assert count == 1
        assert reg.has("project_ping")

    @pytest.mark.asyncio
    async def test_community_disabled_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "plugins_enabled", False)
        plugin_dir = tmp_path / "community_plugins"
        _write_community_plugin(plugin_dir)
        reg = _registry()
        mgr = PluginManager(
            reg,
            providers=[CommunityPluginProvider(plugin_dir=plugin_dir)],
        )
        count = await mgr.load_all()
        assert count == 0
        assert not reg.has("community_ping")

    @pytest.mark.asyncio
    async def test_community_enabled_loads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "plugins_enabled", True)
        plugin_dir = tmp_path / "community_plugins"
        _write_community_plugin(plugin_dir)
        reg = _registry()
        mgr = PluginManager(
            reg,
            providers=[CommunityPluginProvider(plugin_dir=plugin_dir)],
        )
        count = await mgr.load_all()
        assert count == 1
        assert reg.has("community_ping")

    @pytest.mark.asyncio
    async def test_lifecycle_disable_unload(self):
        reg = _registry()
        mgr = PluginManager(reg, providers=[BuiltinPluginProvider()])
        await mgr.load_all()
        assert reg.has("plugin_echo")
        mgr.disable("builtin")
        assert not reg.has("plugin_echo")
        assert mgr.list_plugins()[0].state == PluginState.DISABLED
        mgr.unload("builtin")
        assert mgr.list_plugins()[0].state == PluginState.UNLOADED

    @pytest.mark.asyncio
    async def test_agentstate_integration_via_dispatch(self):
        reg = _registry()
        mgr = PluginManager(reg, providers=[BuiltinPluginProvider()])
        await mgr.load_all()
        state = AgentState(user_request="test")
        tool = reg.get("plugin_echo")
        result = await tool.run(state, message="hi")
        assert result.ok is True
        assert any("plugin_echo" in o.note for o in state.observations)
        assert len(state.tool_history) == 1

    @pytest.mark.asyncio
    async def test_permission_denial(self, monkeypatch):
        monkeypatch.setattr(settings, "plugin_tool_deny", "plugin_echo")
        reg = _registry()
        mgr = PluginManager(
            reg,
            providers=[BuiltinPluginProvider(PermissionChecker())],
        )
        await mgr.load_all()
        state = AgentState(user_request="test")
        result = await reg.get("plugin_echo").run(state, message="blocked")
        assert result.ok is False
        assert result.status == "blocked"

    @pytest.mark.asyncio
    async def test_offline_lock_disables_network_mcp(self, monkeypatch):
        monkeypatch.setattr(settings, "mcp_enabled", True)
        monkeypatch.setattr(settings, "mcp_servers", "https://remote.example/mcp")
        monkeypatch.setattr(settings, "offline_only", True)

        reg = _registry()
        mgr = PluginManager(
            reg,
            providers=[MCPProvider(client=_MockMCPClient())],
        )
        count = await mgr.load_all()
        assert count == 0
        assert mgr.list_plugins()[0].state == PluginState.DISABLED

    @pytest.mark.asyncio
    async def test_all_sources_same_registry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "plugins_enabled", True)
        monkeypatch.setattr(settings, "mcp_enabled", True)
        monkeypatch.setattr(settings, "mcp_servers", "stdio://local")
        _write_project_plugin(tmp_path)
        community_dir = tmp_path / "community_plugins"
        _write_community_plugin(community_dir)

        reg = _registry()
        mgr = PluginManager(
            reg,
            workspace=tmp_path,
            providers=[
                BuiltinPluginProvider(),
                ProjectPluginProvider(tmp_path),
                CommunityPluginProvider(plugin_dir=community_dir),
                MCPProvider(client=_MockMCPClient()),
            ],
        )
        count = await mgr.load_all()
        assert count == 4
        for name in ("plugin_echo", "project_ping", "community_ping", "mcp_greet"):
            assert reg.has(name), f"missing {name}"


class TestAgentEnginePluginIntegration:
    @pytest.mark.asyncio
    async def test_engine_loads_plugins_on_execute(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "plugins_enabled", False)
        monkeypatch.setattr(settings, "mcp_enabled", False)

        class _ScriptedPolicy:
            model = "scripted"
            actions = [
                AgentAction(thought="echo", tool="plugin_echo", args={"message": "from agent"}),
                AgentAction(done=True, final_summary="done"),
            ]
            i = 0

            async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
                a = self.actions[min(self.i, len(self.actions) - 1)]
                self.i += 1
                return LLMResult(data=a, usage=Usage(provider="ollama", model=model))

        engine = AgentEngine(tmp_path, policy_client=_ScriptedPolicy())
        state = await engine.execute(AgentState(user_request="plugin test"))
        assert engine.registry.has("plugin_echo")
        assert any("plugin_echo" in o.note for o in state.observations)

    @pytest.mark.asyncio
    async def test_disabled_mcp_leaves_other_plugins_functional(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "mcp_enabled", False)
        reg = _registry()
        mgr = PluginManager(
            reg,
            workspace=tmp_path,
            providers=[BuiltinPluginProvider(), MCPProvider(client=_MockMCPClient())],
        )
        await mgr.load_all()
        assert reg.has("plugin_echo")
        assert not reg.has("mcp_greet")
        mcp_info = next(p for p in mgr.list_plugins() if p.metadata.name == "mcp")
        assert mcp_info.state == PluginState.DISABLED
