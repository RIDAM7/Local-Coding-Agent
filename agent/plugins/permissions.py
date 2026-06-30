"""Phase 17 — Plugin permission model.

Uniform allow/deny checks applied to every plugin tool before execution.
Native (Phase 10) built-in tools are unaffected — only plugin-registered tools
pass through this gate.
"""

from __future__ import annotations

from typing import List, Set, Tuple

from agent.config import settings


class PermissionChecker:
    """Per-tool allow/deny policy for plugin tools.

    Configuration (comma-separated tool names):
      PLUGIN_TOOL_ALLOW — if non-empty, only listed tools are permitted
      PLUGIN_TOOL_DENY  — listed tools are always blocked
    """

    def __init__(
        self,
        *,
        allow: List[str] | None = None,
        deny: List[str] | None = None,
    ):
        self._allow = self._parse(allow if allow is not None else settings.plugin_tool_allow)
        self._deny = self._parse(deny if deny is not None else settings.plugin_tool_deny)

    @staticmethod
    def _parse(raw: str) -> Set[str]:
        if not raw or not raw.strip():
            return set()
        return {part.strip() for part in raw.split(",") if part.strip()}

    def check(self, tool_name: str, provider_name: str = "") -> Tuple[bool, str]:
        """Return (allowed, reason). ``provider_name`` reserved for future scoping."""
        _ = provider_name
        if tool_name in self._deny:
            return False, f"tool '{tool_name}' denied by PLUGIN_TOOL_DENY"
        if self._allow and tool_name not in self._allow:
            return False, f"tool '{tool_name}' not in PLUGIN_TOOL_ALLOW"
        return True, ""
