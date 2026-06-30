"""Phase 10 — tool registry.

Holds the available tools and advertises their JSON schemas to a tool-calling
model. :func:`build_default_registry` wires the built-in tools to the existing
Round 1 modules. The optional ``web``/``api`` tool is **only registered when
``OFFLINE_ONLY`` is false** — the Offline Lock removes remote tools entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from agent.config import settings
from agent.tools.base import Tool, ToolSpec
from agent.tools.builtin import (
    ApplyPatchTool,
    BuildTool,
    GitOpsTool,
    LintTool,
    ListDirTool,
    ListSymbolsTool,
    MemoryReadTool,
    MemoryWriteTool,
    ReadFileTool,
    RunCommandTool,
    RunTestsTool,
    SearchReplaceTool,
    SearchTool,
    WebTool,
)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return sorted(self._tools)

    def specs(self) -> List[ToolSpec]:
        return [ToolSpec(**t.spec()) for t in self._tools.values()]


def build_default_registry(*, file_manager, executor, safety, rg, sym_idx,
                           index_dir: str, workspace: Path, git_manager,
                           build_validator, lint_validator, test_validator,
                           memory_manager=None) -> ToolRegistry:
    """Wire built-in tools to existing modules. Honors the Offline Lock."""
    reg = ToolRegistry()
    reg.register(ReadFileTool(file_manager))
    reg.register(ListDirTool(file_manager))
    reg.register(SearchTool(rg, workspace))
    reg.register(ListSymbolsTool(sym_idx, index_dir))
    reg.register(BuildTool(build_validator))
    reg.register(LintTool(lint_validator))
    reg.register(RunTestsTool(test_validator))
    reg.register(GitOpsTool(git_manager))
    reg.register(ApplyPatchTool(file_manager, safety))
    reg.register(SearchReplaceTool(file_manager, safety))
    reg.register(RunCommandTool(executor, safety))
    reg.register(MemoryReadTool(memory_manager))
    reg.register(MemoryWriteTool(memory_manager))

    # Offline Lock: remote tools exist only when explicitly allowed to be online.
    if not settings.offline_only:
        reg.register(WebTool())

    return reg
