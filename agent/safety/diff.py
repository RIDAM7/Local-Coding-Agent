"""Unified-diff rendering for file ops (Phase 5).

Before applying a create/update/delete, the controller renders the old-vs-new
content as a unified diff so the user can preview exactly what will change.
"""

import difflib

from agent.safety.redact import redact

_LABELS = {
    "create_file": "new file",
    "update_file": "modified",
    "delete_file": "deleted",
}


def render_diff(path, old_content, new_content, op_type="update_file"):
    """Return a redacted unified diff string for a single file operation."""
    if op_type == "delete_file":
        new_content = ""
    old_lines = (old_content or "").splitlines()
    new_lines = (new_content or "").splitlines()

    label = _LABELS.get(op_type, op_type)
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}",
        lineterm="",
    ))
    body = "\n".join(diff_lines) if diff_lines else "(no textual changes)"
    return redact(f"--- diff: {path} ({label}) ---\n{body}")
