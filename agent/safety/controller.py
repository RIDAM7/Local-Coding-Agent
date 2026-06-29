"""Interactive trust controls (Phase 5).

:class:`SafetyMode` captures the three CLI knobs (``--yes/--auto`` and
``--dry-run``); :class:`SafetyController` enforces them for both shell commands
and file operations:

- **denylist** — checked first for commands, ALWAYS, never bypassable (not even
  with ``--yes``). A denied command is reported as blocked and never run.
- **--dry-run** — previews diffs + planned commands and approves nothing; the
  caller writes nothing and runs nothing.
- **--yes/--auto** — auto-approves the per-op / per-command confirmation prompt
  (for CI); the denylist still applies.
- **default** — interactive: prompt the user to approve each command (when
  ``EXECUTE_COMMANDS`` is on) and each file op before it is applied.
"""

import logging
from dataclasses import dataclass

from agent.safety.commands import find_denied
from agent.safety.diff import render_diff
from agent.safety.redact import redact

# Use the already-configured "agent" logger without importing agent.config at
# module load time (keeps this package free of an import cycle with logging setup).
logger = logging.getLogger("agent")


@dataclass
class SafetyMode:
    """How much to trust the agent for this run."""
    auto_approve: bool = False  # --yes / --auto: skip confirmation prompts
    dry_run: bool = False       # --dry-run: preview only, write/run nothing


@dataclass
class CommandVerdict:
    command: str
    allowed: bool
    status: str          # "approved" | "blocked" | "skipped" | "dry_run"
    reason: str = ""


class SafetyController:
    def __init__(self, mode=None, input_fn=input, output_fn=print):
        self.mode = mode or SafetyMode()
        self._input = input_fn
        self._output = output_fn

    def _confirm(self, question):
        """Prompt for y/N. Auto-approved under --yes; default/denied/EOF => False."""
        if self.mode.auto_approve:
            return True
        try:
            answer = self._input(f"{question} [y/N]: ")
        except EOFError:
            return False
        return str(answer).strip().lower() in ("y", "yes")

    def check_command(self, command):
        """Decide whether a shell command may run. Denylist is non-bypassable."""
        reason = find_denied(command)
        if reason:
            message = f"BLOCKED by safety denylist ({reason}): {command}"
            logger.warning(redact(message))
            self._output(f"⛔ {redact(message)}")
            return CommandVerdict(command, False, "blocked", reason)

        if self.mode.dry_run:
            self._output(f"[dry-run] would execute: {command}")
            return CommandVerdict(command, False, "dry_run", "dry-run preview")

        self._output(f"Proposed command: {command}")
        if self._confirm("Execute this command?"):
            return CommandVerdict(command, True, "approved")
        return CommandVerdict(command, False, "skipped", "declined by user")

    def confirm_file_op(self, op_type, path, old_content, new_content):
        """Preview a file op as a unified diff and decide whether to apply it.

        Returns True if the op should be applied. Under --dry-run it previews and
        returns False (writes nothing); under --yes it auto-approves.
        """
        self._output(render_diff(path, old_content, new_content, op_type))
        if self.mode.dry_run:
            self._output(f"[dry-run] would {op_type}: {path}")
            return False
        return self._confirm(f"Apply {op_type} to {path}?")
