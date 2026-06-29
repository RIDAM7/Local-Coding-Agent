"""Workspace jail (Phase 5).

Every file operation path must resolve (after following symlinks and ``..``) to a
location strictly inside the workspace directory. Path traversal and absolute
escapes are *rejected*, not clamped — the operation is refused with a clear error.
"""

from pathlib import Path

from agent.exceptions.errors import FileOperationError


def assert_within_workspace(workspace, relative_path):
    """Resolve ``relative_path`` under ``workspace`` and assert it stays inside.

    Returns the resolved absolute :class:`Path`. Raises :class:`FileOperationError`
    if the path escapes the workspace (``../`` traversal, absolute path, symlink
    pointing outside, etc.).
    """
    workspace = Path(workspace).resolve()
    try:
        target = (workspace / relative_path).resolve()
    except Exception as e:
        raise FileOperationError(f"Invalid path provided: {relative_path} - {e}")

    if target != workspace and not target.is_relative_to(workspace):
        raise FileOperationError(
            f"Path traversal rejected: '{relative_path}' resolves outside the "
            f"workspace jail ({workspace}). The operation was refused."
        )
    return target
