"""Phase 10 — built-in tools. Each wraps an existing Round 1 module; no logic is
copied. Every write routes through the SafetyController + workspace jail.
"""

from __future__ import annotations

from pathlib import Path

from agent.config import logger
from agent.execution.core import Executor
from agent.files.core import FileManager, apply_search_replace_text
from agent.git.core import GitManager
from agent.safety.controller import SafetyController
from agent.state.agent_state import AgentState, Evidence, RepairAttempt, ValidationResult
from agent.tools.base import Tool, ToolResult


# --- read / inspect ----------------------------------------------------------

class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file inside the workspace."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def __init__(self, file_manager: FileManager):
        self.fm = file_manager

    async def _execute(self, state: AgentState, *, path: str) -> ToolResult:
        content = await self.fm.read_file(path)   # FileManager enforces the jail
        if path not in state.files_read:
            state.files_read.append(path)
        return ToolResult(summary=f"read {path} ({len(content)} chars)", data=content)


class ListDirTool(Tool):
    name = "list_dir"
    description = "List entries of a directory inside the workspace."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}}

    def __init__(self, file_manager: FileManager):
        self.fm = file_manager

    async def _execute(self, state: AgentState, *, path: str = ".") -> ToolResult:
        items = await self.fm.list_directory(path)
        return ToolResult(summary=f"{len(items)} entries in {path}", data=items)


class SearchTool(Tool):
    name = "search"
    description = "Ripgrep the workspace for a literal string."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def __init__(self, rg, workspace: Path):
        self.rg = rg
        self.workspace = str(workspace)

    async def _execute(self, state: AgentState, *, query: str) -> ToolResult:
        hits = await self.rg.exact_search(query, self.workspace)
        for h in hits[:20]:
            path = (h.get("path") or {}).get("text", "")
            if path:
                state.evidence.append(Evidence(kind="search_hit", detail=query, source=path))
        return ToolResult(summary=f"{len(hits)} matches for '{query}'", data=hits)


class ListSymbolsTool(Tool):
    name = "list_symbols"
    description = "List indexed tree-sitter symbols (optionally filtered by name)."
    parameters = {"type": "object", "properties": {"contains": {"type": "string"}}}

    def __init__(self, sym_idx, index_dir: str):
        self.sym_idx = sym_idx
        self.index_dir = index_dir

    async def _execute(self, state: AgentState, *, contains: str = "") -> ToolResult:
        symbols = self.sym_idx.load(self.index_dir) or []
        if contains:
            symbols = [s for s in symbols if contains.lower() in s.name.lower()]
        for s in symbols[:30]:
            state.evidence.append(Evidence(kind="symbol", detail=f"{s.type} {s.name}", source=s.file))
        return ToolResult(summary=f"{len(symbols)} symbols", data=[s.name for s in symbols[:50]])


# --- validation --------------------------------------------------------------

class _ValidatorTool(Tool):
    stage = "BUILD"

    def __init__(self, validator):
        self.validator = validator

    async def _execute(self, state: AgentState, **_) -> ToolResult:
        diag = await self.validator.validate()
        state.validation_results.append(ValidationResult(
            stage=self.stage, success=diag.success,
            detail=(diag.stderr or diag.stdout or "")[:300]))
        return ToolResult(ok=diag.success, status="ok" if diag.success else "error",
                          summary=f"{self.stage} {'passed' if diag.success else 'failed'}", data=diag)


class BuildTool(_ValidatorTool):
    name = "build"
    description = "Run the configured build command."
    stage = "BUILD"


class LintTool(_ValidatorTool):
    name = "lint"
    description = "Run the configured lint command."
    stage = "LINT"


class RunTestsTool(_ValidatorTool):
    name = "run_tests"
    description = "Run the configured test command."
    stage = "TEST"


# --- git ---------------------------------------------------------------------

class GitOpsTool(Tool):
    name = "git_ops"
    description = "Git operations: status | commit | rollback | branch."
    parameters = {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["status", "commit", "rollback", "branch"]},
        "message": {"type": "string"},
    }, "required": ["op"]}

    def __init__(self, git_manager: GitManager):
        self.git = git_manager

    async def _execute(self, state: AgentState, *, op: str, message: str = "") -> ToolResult:
        if not self.git.is_git_repo():
            return ToolResult(ok=False, status="skipped", summary="workspace is not a git repo")
        if op == "status":
            r = self.git._git("status", "--porcelain")
            return ToolResult(summary="git status", data=r.stdout)
        if op == "branch":
            branch = self.git.create_task_branch(message or state.user_request)
            return ToolResult(summary=f"created branch {branch}", data=branch)
        if op == "commit":
            sha = self.git.commit_all(message or "localcli automated change")
            return ToolResult(summary=f"committed {sha}", data=sha)
        if op == "rollback":
            ok = self.git.rollback()
            return ToolResult(ok=ok, summary="git rollback")
        return ToolResult(ok=False, status="error", summary=f"unknown git op '{op}'")


# --- writes (route through SafetyController + jail) --------------------------

class ApplyPatchTool(Tool):
    name = "apply_patch"
    description = "Create, update, or delete a file. Routes through safety + jail."
    parameters = {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["create_file", "update_file", "delete_file"]},
        "path": {"type": "string"},
        "content": {"type": "string"},
    }, "required": ["op", "path"]}

    def __init__(self, file_manager: FileManager, safety: SafetyController):
        self.fm = file_manager
        self.safety = safety

    async def _execute(self, state: AgentState, *, op: str, path: str, content: str = "") -> ToolResult:
        old = ""
        if op in ("update_file", "delete_file"):
            try:
                old = await self.fm.read_file(path)
            except Exception:
                old = ""
        new = "" if op == "delete_file" else (content or "")

        # Every write is previewed + gated by the SafetyController (jail is enforced
        # inside FileManager). Under --dry-run / a decline this returns False.
        if not self.safety.confirm_file_op(op, path, old, new):
            return ToolResult(ok=False, status="skipped", summary=f"{op} {path} not applied (safety)")

        if op == "create_file":
            await self.fm.create_file(path, new)
        elif op == "update_file":
            await self.fm.update_file(path, new)
        elif op == "delete_file":
            await self.fm.delete_file(path)
        else:
            return ToolResult(ok=False, status="error", summary=f"unknown op '{op}'")

        state.record_file_change(path, op)
        return ToolResult(summary=f"{op} {path}")


class SearchReplaceTool(Tool):
    name = "search_replace"
    description = "Replace an exactly-once text block in a file. Routes through safety + jail."
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"},
        "search": {"type": "string"},
        "replace": {"type": "string"},
    }, "required": ["path", "search", "replace"]}

    def __init__(self, file_manager: FileManager, safety: SafetyController):
        self.fm = file_manager
        self.safety = safety

    async def _execute(self, state: AgentState, *, path: str, search: str, replace: str = "") -> ToolResult:
        old = await self.fm.read_file(path)
        new = apply_search_replace_text(old, search, replace)   # exactly-once invariant
        if not self.safety.confirm_file_op("search_replace", path, old, new):
            return ToolResult(ok=False, status="skipped", summary=f"search_replace {path} not applied (safety)")
        await self.fm.update_file(path, new)
        state.record_file_change(path, "search_replace")
        return ToolResult(summary=f"search_replace {path}")


class RunCommandTool(Tool):
    name = "run_command"
    description = "Run a shell command. Routed through the safety denylist + confirmation."
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

    def __init__(self, executor: Executor, safety: SafetyController):
        self.executor = executor
        self.safety = safety

    async def _execute(self, state: AgentState, *, command: str) -> ToolResult:
        verdict = self.safety.check_command(command)
        if verdict.status == "blocked":
            # A denylisted command never reaches the executor — even via the tool path.
            return ToolResult(ok=False, status="blocked", summary=f"blocked: {verdict.reason}")
        if not verdict.allowed:
            return ToolResult(ok=False, status="skipped", summary=f"not executed ({verdict.status})")
        result = await self.executor.run_command(command)
        ok = result.exit_code == 0
        if not ok:
            state.repair_attempts.append(RepairAttempt(
                attempt=len(state.repair_attempts) + 1,
                classification="COMMAND_FAILURE", success=False))
        return ToolResult(ok=ok, status="ok" if ok else "error",
                          summary=f"exit={result.exit_code}", data=result)


# --- memory ------------------------------------------------------------------

class MemoryReadTool(Tool):
    name = "memory_read"
    description = "Recall similar past runs from project memory (degrades to empty)."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def __init__(self, memory_manager):
        self.mm = memory_manager

    async def _execute(self, state: AgentState, *, query: str) -> ToolResult:
        if not self.mm or not getattr(self.mm, "enabled", False):
            return ToolResult(ok=False, status="skipped", summary="memory disabled")
        try:
            hits = await self.mm.successful_repairs(query, [], limit=3)
        except Exception as e:
            return ToolResult(ok=False, status="error", summary=f"memory read failed: {e}")
        for h in hits:
            state.memory_refs.summaries.append(str(getattr(h, "content", ""))[:200])
        return ToolResult(summary=f"{len(hits)} memory hits", data=len(hits))


class MemoryWriteTool(Tool):
    name = "memory_write"
    description = "Persist a learning to project memory (path-scrubbed; degrades to no-op)."
    parameters = {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}

    def __init__(self, memory_manager):
        self.mm = memory_manager

    async def _execute(self, state: AgentState, *, content: str) -> ToolResult:
        if not self.mm or not getattr(self.mm, "enabled", False):
            return ToolResult(ok=False, status="skipped", summary="memory disabled")
        # MVP: record the intent on the state; the P12 updater owns durable writes.
        state.memory_refs.summaries.append(content[:200])
        logger.debug("memory_write tool recorded a learning on the state (P12 will persist).")
        return ToolResult(summary="learning recorded")


# --- optional / remote (DISABLED under OFFLINE_ONLY) -------------------------

class WebTool(Tool):
    name = "web"
    description = "Fetch a URL (remote). DISABLED under OFFLINE_ONLY."
    parameters = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}

    async def _execute(self, state: AgentState, *, url: str) -> ToolResult:  # pragma: no cover
        # Only ever registered when OFFLINE_ONLY is false (see registry). Kept a
        # clean extension point for 10B; not exercised in the offline default path.
        return ToolResult(ok=False, status="error", summary="web tool not implemented (10B stub)")
