"""Hard command denylist (Phase 5).

A small set of obviously destructive shell patterns that are **ALWAYS** blocked,
regardless of ``--yes``/``--auto`` or any setting. This is the non-bypassable
floor of command safety: the orchestrator checks it before ever calling the
executor, and :class:`~agent.execution.core.Executor` re-checks it as a backstop,
so a denylisted command can never reach a subprocess.
"""

import re

# (compiled pattern, human-readable reason). Patterns are case-insensitive.
DENYLIST = [
    # rm with both recursive and force, any flag order / long flags.
    (re.compile(r"\brm\b\s+(?:-\S*\s+)*-\S*r\S*f", re.I), "recursive force file deletion (rm -rf)"),
    (re.compile(r"\brm\b\s+(?:-\S*\s+)*-\S*f\S*r", re.I), "recursive force file deletion (rm -fr)"),
    (re.compile(r"\brm\b(?=.*\s-r\b)(?=.*\s-f\b)", re.I), "recursive force file deletion (rm -r -f)"),
    (re.compile(r"\brm\b(?=.*--recursive)(?=.*--force)", re.I), "recursive force file deletion (rm --recursive --force)"),
    (re.compile(r"\bsudo\b\s+rm\b", re.I), "privileged file deletion (sudo rm)"),
    # Pipe a downloaded script straight into a shell.
    (re.compile(r"\b(?:curl|wget)\b.+\|\s*(?:sudo\s+)?(?:sh|bash|zsh|dash)\b", re.I), "pipe-to-shell of remote script (curl|wget ... | sh)"),
    # Filesystem / raw disk destruction.
    (re.compile(r"\bmkfs(?:\.\w+)?\b", re.I), "filesystem format (mkfs)"),
    (re.compile(r"\bdd\b.+\bif=", re.I), "raw disk write (dd if=)"),
    (re.compile(r">\s*/dev/(?:sd|nvme|hd|disk|vd)\w*", re.I), "overwrite block device (> /dev/sdX)"),
    (re.compile(r"\bmv\b\s+\S+\s+/dev/null\b", re.I), "destructive move to /dev/null"),
    # Fork bomb.
    (re.compile(r":\(\)\s*\{.*\|\s*:.*&\s*\}\s*;?\s*:", re.S), "fork bomb"),
    # System power state.
    (re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", re.I), "system shutdown/reboot"),
    (re.compile(r"\binit\s+[06]\b", re.I), "system runlevel change (init 0/6)"),
    # Over-permissive recursive chmod on root.
    (re.compile(r"\bchmod\b\s+-R\s+0*777\s+/(?:\s|$)", re.I), "recursive chmod 777 on /"),
    # Windows destructive equivalents.
    (re.compile(r"\bformat\b\s+[a-zA-Z]:", re.I), "drive format (format C:)"),
    (re.compile(r"\bdel\b\s+/[sfq]\b", re.I), "Windows recursive delete (del /s)"),
    (re.compile(r"\brd\b\s+/s\b|\brmdir\b\s+/s\b", re.I), "Windows recursive rmdir (rd /s)"),
]


def find_denied(command):
    """Return the reason a command is denylisted, or ``None`` if it is allowed.

    Note: ``None`` means *not on the hard denylist* — it does NOT mean approved.
    Confirmation / allowlist logic still applies (see :class:`SafetyController`).
    """
    cmd = command or ""
    for pattern, reason in DENYLIST:
        if pattern.search(cmd):
            return reason
    return None
