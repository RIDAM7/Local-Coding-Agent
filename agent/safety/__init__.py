"""Phase 5 safety + interactive trust controls.

This package makes it safe to point the agent at a real repo and safe to publish:

- ``commands``   — a hard, non-bypassable denylist of destructive shell patterns.
- ``diff``       — unified-diff rendering for file ops (preview before apply).
- ``jail``       — workspace jail: every file op must resolve inside ``workspace/``.
- ``redact``     — secret scrubbing for logs AND reports.
- ``controller`` — ``SafetyMode`` + ``SafetyController`` (confirmation, --yes, --dry-run).
"""

from agent.safety.controller import SafetyController, SafetyMode, CommandVerdict
from agent.safety.commands import find_denied
from agent.safety.diff import render_diff
from agent.safety.jail import assert_within_workspace
from agent.safety.redact import redact, RedactionFilter

__all__ = [
    "SafetyController",
    "SafetyMode",
    "CommandVerdict",
    "find_denied",
    "render_diff",
    "assert_within_workspace",
    "redact",
    "RedactionFilter",
]
