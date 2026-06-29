"""Git integration helper (Phase 7B).

Only active when ``GIT_INTEGRATION`` is true AND the workspace is a git repo AND
the run is not a ``--dry-run``. It creates a task branch at run start, commits the
applied changes on a successful run with a redacted message, and exposes a
git-based rollback that *complements* (does not replace) the snapshot rollback.

All git calls are best-effort: any failure is logged and surfaced to the caller,
never crashing the run. No secret is ever placed in a commit message.
"""

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from agent.config import settings, logger
from agent.safety.redact import redact


def _slugify(text: str, max_len: int = 32) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "task"


class GitManager:
    def __init__(self, workspace_path: Path = None):
        self.workspace = Path(workspace_path) if workspace_path else settings.get_workspace_path()

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        """Run a git command in the workspace, capturing output (no exception)."""
        return subprocess.run(
            ["git", *args],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
        )

    def git_available(self) -> bool:
        return shutil.which("git") is not None

    def is_git_repo(self) -> bool:
        if not self.git_available():
            return False
        try:
            result = self._git("rev-parse", "--is-inside-work-tree")
        except Exception:
            return False
        return result.returncode == 0 and result.stdout.strip() == "true"

    def create_task_branch(self, task_description: str) -> str:
        """Create and switch to a task branch; returns the branch name."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch = f"localcli/{_slugify(task_description)}-{timestamp}"
        result = self._git("checkout", "-b", branch)
        if result.returncode != 0:
            raise RuntimeError(f"git checkout -b failed: {result.stderr.strip()}")
        return branch

    def commit_all(self, message: str) -> str | None:
        """Stage everything and commit with a REDACTED message. Returns the commit
        hash, or None if there was nothing to commit."""
        safe_message = redact(message or "localcli automated change")
        self._git("add", "-A")
        # Nothing staged? Then there is nothing to commit.
        if self._git("diff", "--cached", "--quiet").returncode == 0:
            logger.info("Git: no changes to commit.")
            return None
        result = self._git("commit", "-m", safe_message)
        if result.returncode != 0:
            raise RuntimeError(f"git commit failed: {result.stderr.strip()}")
        rev = self._git("rev-parse", "HEAD")
        return rev.stdout.strip() if rev.returncode == 0 else None

    def rollback(self) -> bool:
        """Git-based rollback that complements the snapshot rollback: discard any
        staged/working-tree changes back to the current branch HEAD."""
        result = self._git("reset", "--hard", "HEAD")
        if result.returncode != 0:
            logger.warning(f"Git: rollback (reset --hard) failed: {result.stderr.strip()}")
            return False
        self._git("clean", "-fd")
        return True
