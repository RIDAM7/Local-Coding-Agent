"""Phase 10 — AgentState: threading through both strategies, serialization
round-trip + redaction-clean, and per-tool slice updates. Fully offline."""

import pytest
from unittest.mock import AsyncMock

from agent.state.agent_state import AgentState, TaskMetadata
from agent.llm.providers.base import LLMResult, Usage
from agent.engine.agent_engine import AgentEngine, AgentAction
from agent.engine.pipeline_engine import PipelineEngine
from agent.tools import build_default_registry, ToolExecutor
from agent.execution.core import Executor
from agent.files.core import FileManager
from agent.git.core import GitManager
from agent.safety.controller import SafetyController, SafetyMode
from agent.retrieval import RipgrepSearch, SymbolIndex, TreeSitterIndexer
from agent.validation import BuildValidator, LintValidator
from agent.validation import TestValidator as _TestValidator


class _ScriptedPolicy:
    model = "scripted:latest"

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
        if self.i < len(self.actions):
            a = self.actions[self.i]
            self.i += 1
        else:
            a = AgentAction(done=True, final_summary="done")
        return LLMResult(data=a, usage=Usage(provider="ollama", model=model))


def _registry(tmp_path):
    fm = FileManager(tmp_path)
    ex = Executor(tmp_path)
    safety = SafetyController(SafetyMode(auto_approve=True))
    return build_default_registry(
        file_manager=fm, executor=ex, safety=safety, rg=RipgrepSearch(),
        sym_idx=SymbolIndex(TreeSitterIndexer()), index_dir=str(tmp_path / "index"),
        workspace=tmp_path, git_manager=GitManager(tmp_path),
        build_validator=BuildValidator(ex), lint_validator=LintValidator(ex),
        test_validator=_TestValidator(ex), memory_manager=None,
    )


def test_serialization_roundtrip():
    state = AgentState(user_request="do a thing", task=TaskMetadata(description="x"))
    state.add_observation("hello")
    state.record_file_change("a.py", "create_file")
    restored = AgentState.from_json(state.to_json())
    assert restored.user_request == "do a thing"
    assert restored.files_modified[0].path == "a.py"
    assert any(o.note == "hello" for o in restored.observations)


def test_serialization_is_redaction_clean():
    state = AgentState(user_request="task")
    # A secret-looking token must never survive serialization.
    state.add_observation("my key is sk-ABCDEF1234567890 do not leak")
    out = state.to_json()
    assert "sk-ABCDEF1234567890" not in out
    assert "***REDACTED***" in out


@pytest.mark.asyncio
async def test_each_tool_appends_to_right_slice(tmp_path):
    (tmp_path / "hello.txt").write_text("hi there", encoding="utf-8")
    reg = _registry(tmp_path)
    ex = ToolExecutor(reg)
    state = AgentState(user_request="t")

    await ex.dispatch(state, "read_file", {"path": "hello.txt"})
    assert "hello.txt" in state.files_read

    await ex.dispatch(state, "apply_patch", {"op": "create_file", "path": "new.py", "content": "x=1\n"})
    assert any(f.path == "new.py" for f in state.files_modified)
    assert (tmp_path / "new.py").exists()

    await ex.dispatch(state, "run_tests", {})
    assert any(v.stage == "TEST" for v in state.validation_results)

    # Every dispatch records a ToolCall + a timeline event.
    assert len(state.tool_history) == 3
    assert len(state.timeline) >= 3


@pytest.mark.asyncio
async def test_state_threads_through_agent_strategy(tmp_path):
    policy = _ScriptedPolicy([
        AgentAction(thought="create", tool="apply_patch",
                    args={"op": "create_file", "path": "f.py", "content": "y=2\n"}),
        AgentAction(done=True, final_summary="finished"),
    ])
    engine = AgentEngine(tmp_path, policy_client=policy)
    state = AgentState(user_request="make f.py")
    out = await engine.execute(state)

    assert out is state                      # same object, mutated by reference
    assert out.execution_mode == "agent"
    assert (tmp_path / "f.py").exists()
    assert out.final_outputs.status == "SUCCESS"
    assert out.governor.stop_reason == "done"
    assert out.tool_history                  # tool history populated


@pytest.mark.asyncio
async def test_state_threads_through_pipeline_strategy():
    orch = AsyncMock()
    orch.run = AsyncMock(return_value="reports/r.md")
    engine = PipelineEngine(orchestrator=orch)
    state = AgentState(user_request="the request")
    out = await engine.execute(state)

    orch.run.assert_awaited_once_with("the request")
    assert out.execution_mode == "pipeline"
    assert out.final_outputs.report_path == "reports/r.md"
