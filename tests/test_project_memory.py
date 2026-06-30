"""Phase 12 - local markdown project memory. Offline and deterministic."""

from pathlib import Path

from agent import cli
from agent.context.schemas import ContextBundle
from agent.memory.project_memory import MEMORY_VERSION, ProjectMemoryManager
from agent.state.agent_state import AgentState, StepResult, ToolCall, ValidationResult


def _context(root: Path) -> ContextBundle:
    return ContextBundle(
        root=str(root),
        generated_at="2026-01-01T00:00:00+00:00",
        fingerprint="fp",
        architecture_summary="The app has a CLI layer and a service layer.",
    )


def _successful_state(root: Path) -> AgentState:
    state = AgentState(user_request="add feature")
    state.loaded_context = _context(root)
    state.final_outputs.status = "SUCCESS"
    state.completed_steps.append(
        StepResult(
            index=0,
            description="Use the service helper",
            status="done",
            summary=f"Followed the helper pattern in {root / 'src' / 'service.py'}",
        )
    )
    state.validation_results.append(
        ValidationResult(stage="TEST", success=False, detail="pytest failed before fix")
    )
    state.tool_history.append(
        ToolCall(name="run_command", args={"command": "pytest -q"}, status="ok")
    )
    state.memory_refs.summaries.append("Prefer small focused patches and keep tests nearby.")
    return state


def test_project_memory_creation_update_dedupe_and_scrub(tmp_path):
    manager = ProjectMemoryManager(tmp_path, enabled=True)
    state = _successful_state(tmp_path)

    first = manager.update_from_state(state)
    second = manager.update_from_state(state)

    assert first.entries_added > 0
    assert second.entries_added == 0
    assert (tmp_path / ".localcli" / "memory" / "learned_patterns.md").exists()
    learned = (tmp_path / ".localcli" / "memory" / "learned_patterns.md").read_text(encoding="utf-8")
    commands = (tmp_path / ".localcli" / "memory" / "commands.md").read_text(encoding="utf-8")
    preferences = (tmp_path / ".localcli" / "memory" / "developer_preferences.md").read_text(encoding="utf-8")
    mistakes = (tmp_path / ".localcli" / "memory" / "mistakes.md").read_text(encoding="utf-8")

    assert MEMORY_VERSION in learned
    assert "Use the service helper" in learned
    assert str(tmp_path) not in learned
    assert "[WORKSPACE]" in learned or "[SCRUBBED_PATH]" in learned
    assert "pytest -q" in commands
    assert "Prefer small focused patches" in preferences
    assert "pytest failed before fix" in mistakes


def test_project_memory_loading_and_agentstate_context_integration(tmp_path):
    manager = ProjectMemoryManager(tmp_path, enabled=True)
    manager.update({"learned_patterns": ["Use FastAPI dependency injection for request state."]})
    state = AgentState(user_request="change API")
    state.loaded_context = _context(tmp_path)

    bundle = manager.load_into_state(state)

    assert bundle.used_files
    assert ".localcli/memory/learned_patterns.md" in state.memory_refs.markdown_files
    assert any("FastAPI dependency injection" in s for s in state.memory_refs.summaries)
    planner_block = state.loaded_context.to_planner_block()
    assert "Project Memory" in planner_block
    assert "FastAPI dependency injection" in planner_block


def test_project_memory_disabled_mode(tmp_path):
    manager = ProjectMemoryManager(tmp_path, enabled=False)
    state = _successful_state(tmp_path)

    bundle = manager.load_into_state(state)
    result = manager.update_from_state(state)

    assert bundle.files == {}
    assert result.entries_added == 0
    assert not (tmp_path / ".localcli" / "memory").exists()
    assert state.memory_refs.markdown_files == []


def test_project_memory_recovers_hand_edited_file_without_header(tmp_path):
    memory_dir = tmp_path / ".localcli" / "memory"
    memory_dir.mkdir(parents=True)
    path = memory_dir / "learned_patterns.md"
    path.write_text("Remember to keep schema changes backwards compatible.\n", encoding="utf-8")

    manager = ProjectMemoryManager(tmp_path, enabled=True)
    bundle = manager.load()
    recovered = path.read_text(encoding="utf-8")

    assert ".localcli/memory/learned_patterns.md" in bundle.recovered_files
    assert MEMORY_VERSION in recovered
    assert "backwards compatible" in recovered


def test_memory_cli_views_local_markdown(tmp_path, monkeypatch, capsys):
    manager = ProjectMemoryManager(tmp_path, enabled=True)
    manager.update({"commands": ["Useful command: `pytest -q`"]})
    monkeypatch.setattr(cli.settings, "workspace_dir", str(tmp_path))
    monkeypatch.setattr(cli.settings, "project_memory_enabled", True)
    monkeypatch.setattr(cli.settings, "project_memory_dir", ".localcli/memory")

    rc = cli.main(["memory"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Project Memory" in out
    assert "commands.md" in out
    assert "pytest -q" in out
