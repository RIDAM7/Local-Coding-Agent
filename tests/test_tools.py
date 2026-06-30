"""Phase 10 — tools wrap the right Round 1 modules; every write routes through
safety + jail; a denylisted command is blocked through the tool path. Offline."""

import pytest
from unittest.mock import patch

from agent.state.agent_state import AgentState
from agent.tools import build_default_registry, ToolExecutor
from agent.execution.core import Executor
from agent.files.core import FileManager
from agent.git.core import GitManager
from agent.safety.controller import SafetyController, SafetyMode
from agent.retrieval import RipgrepSearch, SymbolIndex, TreeSitterIndexer
from agent.validation import BuildValidator, LintValidator
from agent.validation import TestValidator as _TestValidator


def _build(tmp_path, safety=None):
    fm = FileManager(tmp_path)
    ex = Executor(tmp_path)
    safety = safety or SafetyController(SafetyMode(auto_approve=True))
    reg = build_default_registry(
        file_manager=fm, executor=ex, safety=safety, rg=RipgrepSearch(),
        sym_idx=SymbolIndex(TreeSitterIndexer()), index_dir=str(tmp_path / "index"),
        workspace=tmp_path, git_manager=GitManager(tmp_path),
        build_validator=BuildValidator(ex), lint_validator=LintValidator(ex),
        test_validator=_TestValidator(ex), memory_manager=None,
    )
    return reg, ToolExecutor(reg), ex, safety


@pytest.mark.asyncio
async def test_read_file_tool_wraps_file_manager(tmp_path):
    (tmp_path / "a.txt").write_text("content here", encoding="utf-8")
    _, ex, _, _ = _build(tmp_path)
    state = AgentState()
    res = await ex.dispatch(state, "read_file", {"path": "a.txt"})
    assert res.ok and res.data == "content here"
    assert "a.txt" in state.files_read


@pytest.mark.asyncio
async def test_apply_patch_routes_through_safety_and_jail(tmp_path):
    _, ex, _, safety = _build(tmp_path)
    state = AgentState()
    with patch.object(safety, "confirm_file_op", wraps=safety.confirm_file_op) as spy:
        res = await ex.dispatch(state, "apply_patch",
                                {"op": "create_file", "path": "made.py", "content": "z=3\n"})
    assert res.ok
    assert spy.called                         # the write went through the SafetyController
    assert (tmp_path / "made.py").exists()


@pytest.mark.asyncio
async def test_jail_rejects_traversal_through_tool(tmp_path):
    _, ex, _, _ = _build(tmp_path)
    state = AgentState()
    res = await ex.dispatch(state, "read_file", {"path": "../escape.txt"})
    # FileManager's jail raises; the tool surfaces it as an error, never escapes.
    assert not res.ok
    assert res.status == "error"


@pytest.mark.asyncio
async def test_denylisted_command_blocked_through_tool_path(tmp_path):
    _, ex, executor, _ = _build(tmp_path)
    state = AgentState()
    with patch.object(executor, "run_command", wraps=executor.run_command) as spy:
        res = await ex.dispatch(state, "run_command", {"command": "rm -rf /"})
    assert res.status == "blocked"
    spy.assert_not_called()                    # never reached the executor


@pytest.mark.asyncio
async def test_run_command_tool_executes_allowed_command(tmp_path):
    _, ex, _, _ = _build(tmp_path)
    state = AgentState()
    res = await ex.dispatch(state, "run_command", {"command": "echo hello"})
    assert res.ok and res.status == "ok"


@pytest.mark.asyncio
async def test_unknown_tool_is_graceful(tmp_path):
    _, ex, _, _ = _build(tmp_path)
    state = AgentState()
    res = await ex.dispatch(state, "does_not_exist", {})
    assert not res.ok and res.status == "error"


@pytest.mark.asyncio
async def test_validation_tools_append_results(tmp_path):
    _, ex, _, _ = _build(tmp_path)
    state = AgentState()
    await ex.dispatch(state, "build", {})
    await ex.dispatch(state, "lint", {})
    stages = {v.stage for v in state.validation_results}
    assert {"BUILD", "LINT"} <= stages
