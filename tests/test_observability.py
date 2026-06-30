"""Phase 14 - Observability over AgentState. Offline and deterministic."""

from unittest.mock import AsyncMock

from agent import cli
from agent.config import settings
from agent.models.schemas import Report
from agent.observability import (
    build_observability_snapshot,
    render_dashboard,
    render_dashboard_snapshot,
    render_report_view,
)
from agent.reporting.core import Reporter
from agent.state.agent_state import (
    AgentState,
    RepairAttempt,
    StepResult,
    ToolCall,
    ValidationResult,
)


def _state() -> AgentState:
    state = AgentState(user_request="change service")
    state.execution_mode = "agent"
    state.objective = "change service safely"
    state.plan.steps = [
        StepResult(index=0, description="inspect", status="done"),
        StepResult(index=1, description="edit", status="pending"),
    ]
    state.completed_steps.append(StepResult(index=0, description="inspect", status="done", summary="ok"))
    state.tool_history.append(ToolCall(name="read_file", status="ok", result_summary="read app.py"))
    state.tool_history.append(ToolCall(name="run_tests", status="error", result_summary="failed"))
    state.memory_refs.vector_ids.append("v1")
    state.memory_refs.markdown_files.append(".localcli/memory/learned_patterns.md")
    state.memory_refs.summaries.append("Prefer focused patches.")
    state.validation_results.append(ValidationResult(stage="TEST", success=False, detail="failed"))
    state.repair_attempts.append(RepairAttempt(attempt=1, classification="TEST_FAILURE", success=False))
    state.confidence = 0.71
    state.final_outputs.status = "FAILURE"
    state.add_timeline("engine", "started")
    state.add_timeline("validation", "TEST failed")
    state.add_timeline("failure", "run finished: FAILURE")
    return state


def test_timeline_creation_order_and_snapshot_panels(monkeypatch):
    monkeypatch.setattr(settings, "observability_enabled", True)
    state = _state()

    assert [e.message for e in state.timeline][-3:] == [
        "started",
        "TEST failed",
        "run finished: FAILURE",
    ]

    snap = build_observability_snapshot(state)
    assert snap["goal"] == "change service safely"
    assert snap["progress"]["completed"] == 1
    assert snap["progress"]["total"] == 2
    assert snap["tools"]["by_name"] == {"read_file": 1, "run_tests": 1}
    assert snap["memory"]["vector_hits"] == 1
    assert snap["memory"]["markdown_files"] == [".localcli/memory/learned_patterns.md"]
    assert snap["repairs"]["failed"] == 1
    assert snap["validation"]["failed"] == 1


def test_disabled_observability_skips_timeline(monkeypatch):
    monkeypatch.setattr(settings, "observability_enabled", False)
    state = AgentState(user_request="x")

    state.add_timeline("engine", "should not record")
    state.add_observation("observation still records on observation slice")

    assert state.timeline == []
    assert len(state.observations) == 1


def test_dashboard_and_report_view_agree_and_redact(monkeypatch):
    monkeypatch.setattr(settings, "observability_enabled", True)
    state = _state()
    state.add_timeline("tool", "used key sk-ABCDEF1234567890")
    snap = build_observability_snapshot(state)

    dashboard = render_dashboard_snapshot(snap)
    report_view = render_report_view(snap)

    assert "Execution Observability" in dashboard
    assert "change service safely" in dashboard
    assert "run_tests" in dashboard
    assert "Execution Observability" in report_view
    assert "sk-ABCDEF1234567890" not in dashboard
    assert "sk-ABCDEF1234567890" not in report_view


def test_reporting_includes_observability_markdown_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "observability_enabled", True)
    state = _state()
    snap = build_observability_snapshot(state)
    report = Report(
        task="change service",
        execution_results="failed",
        final_status="FAILURE",
        timestamp="2026-01-01T00:00:00+00:00",
        observability=snap,
    )

    path = Reporter(tmp_path).generate_report(report)
    md = tmp_path.joinpath(path.split("\\")[-1]).read_text(encoding="utf-8")
    js = tmp_path.joinpath(path.split("\\")[-1].replace(".md", ".json")).read_text(encoding="utf-8")

    assert "## Observability" in md
    assert "Execution Observability" in md
    assert '"observability"' in js
    assert "run finished: FAILURE" in js


def test_cli_observe_prints_dashboard(monkeypatch, capsys):
    import agent.engine.selector as selector
    import agent.orchestrator as orchestrator_module

    monkeypatch.setattr(settings, "observability_enabled", True)
    monkeypatch.setattr(settings, "verbosity", "normal")
    monkeypatch.setattr(settings, "incremental_planning", True)
    monkeypatch.setattr(cli, "tooling_check", lambda: None)
    monkeypatch.setattr(cli, "preflight_check", AsyncMock(return_value=None))
    monkeypatch.setattr(cli, "_ensure_index", lambda orchestrator: None)
    monkeypatch.setattr(selector, "resolve_mode", AsyncMock(return_value=("pipeline", None)))

    fake = AsyncMock()
    fake.memory_manager = None
    fake.last_state = _state()
    fake.run_incremental = AsyncMock(return_value="reports/r.md")
    monkeypatch.setattr(orchestrator_module, "Orchestrator", lambda safety_mode=None: fake)

    rc = cli.main(["run", "change service", "--yes", "--observe"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Execution Observability" in out
    assert "change service safely" in out


def test_success_and_failure_events_are_visible(monkeypatch):
    monkeypatch.setattr(settings, "observability_enabled", True)
    success = AgentState(user_request="ok")
    success.final_outputs.status = "SUCCESS"
    success.add_timeline("completion", "run finished: SUCCESS")

    failure = AgentState(user_request="bad")
    failure.final_outputs.status = "FAILURE"
    failure.add_timeline("failure", "run finished: FAILURE")

    assert "run finished: SUCCESS" in render_dashboard(success)
    assert "run finished: FAILURE" in render_dashboard(failure)
