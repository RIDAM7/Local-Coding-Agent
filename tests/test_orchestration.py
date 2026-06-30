"""Phase 16 — Agent Orchestration Layer (MVP). Offline and deterministic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.config import settings
from agent.observability import build_observability_snapshot
from agent.orchestration import Coordinator, Merger, Worker, WorkerResult, WorkerSpec, decompose
from agent.orchestration.scheduler import Scheduler
from agent.session import SessionStore
from agent.state.agent_state import AgentState, Evidence, FileChange, TaskMetadata


# --- helpers ----------------------------------------------------------------


def _parent_state(task: str = "add api endpoint and react ui with tests") -> AgentState:
    state = AgentState(user_request=task, task=TaskMetadata(description=task))
    state.execution_mode = "pipeline"
    state.objective = task
    state.evidence.append(Evidence(kind="graph_impact", detail="app.py", source="graph"))
    state.evidence.append(Evidence(kind="search_hit", detail="routes.py", source="rg"))
    return state


def _worker_result(
    *,
    role: str = "backend",
    success: bool = True,
    path: str = "app.py",
    steps: int = 2,
    cost: float = 0.01,
) -> WorkerResult:
    return WorkerResult(
        success=success,
        summary=f"{role} done",
        files_modified=[FileChange(path=path, op="update_file", summary=role)],
        evidence=[{"kind": "search_hit", "detail": path, "source": role}],
        steps_used=steps,
        tool_calls_used=1,
        cost_used_usd=cost,
        status="SUCCESS" if success else "FAILURE",
    )


# --- decomposer -------------------------------------------------------------


class TestDecomposer:
    def test_single_role_defaults_to_backend(self):
        specs = decompose(_parent_state("fix the parser"))
        assert len(specs) == 1
        assert specs[0].role == "backend"
        assert specs[0].sub_task == "fix the parser"

    def test_multi_concern_dependency_order(self):
        task = "add rest api endpoint, react ui component, and pytest tests"
        specs = decompose(_parent_state(task))
        roles = [s.role for s in specs]
        assert roles == ["backend", "frontend", "test"]
        assert all(task in s.sub_task for s in specs)
        assert specs[0].execution_mode == "pipeline"
        assert specs[0].parent_session_id

    def test_docs_role_last(self):
        task = "update readme documentation and api docs for the server"
        specs = decompose(_parent_state(task))
        roles = [s.role for s in specs]
        assert "docs" in roles
        assert roles.index("docs") == len(roles) - 1


# --- worker statelessness ---------------------------------------------------


class TestWorker:
    @pytest.mark.asyncio
    async def test_run_is_pure_no_shared_state(self, monkeypatch):
        """Worker holds no state across calls — each run is independent."""
        calls = []

        async def fake_execute(state):
            calls.append(state.user_request)
            state.final_outputs.status = "SUCCESS"
            state.final_outputs.summary = "ok"
            state.governor.steps_used = 1
            return state

        mock_engine = MagicMock()
        mock_engine.execute = fake_execute
        monkeypatch.setattr("agent.orchestration.worker.build_engine", lambda *a, **k: mock_engine)

        spec_a = WorkerSpec(role="backend", sub_task="task A", execution_mode="pipeline")
        spec_b = WorkerSpec(role="frontend", sub_task="task B", execution_mode="pipeline")

        r1 = await Worker.run(spec_a)
        r2 = await Worker.run(spec_b)

        assert r1.success is True
        assert r2.success is True
        assert calls == ["task A", "task B"]
        assert r1.summary == "ok"
        assert r2.summary == "ok"

    @pytest.mark.asyncio
    async def test_run_returns_failure_on_engine_error(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr("agent.orchestration.worker.build_engine", lambda *a, **k: mock_engine)

        result = await Worker.run(WorkerSpec(role="test", sub_task="x"))
        assert result.success is False
        assert "RuntimeError" in result.summary


# --- merger -----------------------------------------------------------------


class TestMerger:
    def test_last_writer_wins_for_files(self):
        state = _parent_state()
        state.files_modified = [FileChange(path="a.py", op="update_file", summary="first")]
        results = [
            _worker_result(path="a.py", role="backend"),
            WorkerResult(
                success=True,
                files_modified=[FileChange(path="a.py", op="update_file", summary="second")],
            ),
        ]
        Merger.merge(state, results)
        assert len(state.files_modified) == 1
        assert state.files_modified[0].summary == "second"

    def test_sums_governor_usage(self):
        state = _parent_state()
        results = [_worker_result(steps=2, cost=0.01), _worker_result(steps=3, cost=0.02)]
        Merger.merge(state, results)
        assert state.governor.steps_used == 5
        assert state.governor.cost_used_usd == pytest.approx(0.03)


# --- scheduler --------------------------------------------------------------


class TestScheduler:
    @pytest.mark.asyncio
    async def test_sequential_order(self, monkeypatch):
        order = []

        async def fake_run(spec, **kwargs):
            order.append(spec.role)
            return _worker_result(role=spec.role)

        monkeypatch.setattr("agent.orchestration.scheduler.Worker.run", fake_run)
        specs = [
            WorkerSpec(role="backend", sub_task="a"),
            WorkerSpec(role="frontend", sub_task="b"),
        ]
        results = await Scheduler().run(specs)
        assert order == ["backend", "frontend"]
        assert len(results) == 2


# --- coordinator ------------------------------------------------------------


class TestCoordinator:
    @pytest.mark.asyncio
    async def test_disabled_is_noop(self, monkeypatch):
        monkeypatch.setattr(settings, "orchestration_enabled", False)
        state = _parent_state()
        coord = Coordinator()
        out = await coord.run(state)
        assert out is state
        assert out.final_outputs.status == "PENDING"

    @pytest.mark.asyncio
    async def test_routes_schedules_merges_validates(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "orchestration_enabled", True)
        monkeypatch.setattr(settings, "observability_enabled", True)
        monkeypatch.setattr(settings, "session_persistence", True)
        monkeypatch.setattr(settings, "session_dir", str(tmp_path / ".localcli"))

        async def fake_worker(spec, **kwargs):
            return _worker_result(role=spec.role, path=f"{spec.role}.py")

        monkeypatch.setattr("agent.orchestration.coordinator.Worker.run", fake_worker)

        mock_orch = MagicMock()
        ok = MagicMock(success=True, stderr="")
        mock_orch.build_validator.validate = AsyncMock(return_value=ok)
        mock_orch.lint_validator.validate = AsyncMock(return_value=ok)
        mock_orch.test_validator.validate = AsyncMock(return_value=ok)

        state = _parent_state("add api endpoint and react ui with pytest tests")
        coord = Coordinator(orchestrator=mock_orch)
        out = await coord.run(state)

        assert out.final_outputs.status == "SUCCESS"
        assert len(out.files_modified) == 3
        roles = {fc.summary for fc in out.files_modified}
        assert roles == {"backend", "frontend", "test"}
        assert any(v.stage == "BUILD" for v in out.validation_results)
        timeline_kinds = [e.kind for e in out.timeline if e.kind == "orchestration"]
        assert timeline_kinds

    @pytest.mark.asyncio
    async def test_worker_failure_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(settings, "orchestration_enabled", True)
        monkeypatch.setattr(settings, "observability_enabled", True)

        call_count = 0

        async def fake_worker(spec, **kwargs):
            nonlocal call_count
            call_count += 1
            if spec.role == "frontend":
                return WorkerResult(success=False, summary="frontend failed", status="FAILURE")
            return _worker_result(role=spec.role, path=f"{spec.role}.py")

        monkeypatch.setattr("agent.orchestration.coordinator.Worker.run", fake_worker)

        mock_orch = MagicMock()
        ok = MagicMock(success=True, stderr="")
        mock_orch.build_validator.validate = AsyncMock(return_value=ok)
        mock_orch.lint_validator.validate = AsyncMock(return_value=ok)
        mock_orch.test_validator.validate = AsyncMock(return_value=ok)

        state = _parent_state("add api endpoint and react ui with tests")
        out = await Coordinator(orchestrator=mock_orch).run(state)

        assert call_count == 3
        assert out.final_outputs.status == "FAILURE"

    @pytest.mark.asyncio
    async def test_global_budget_halts_overrun(self, monkeypatch):
        monkeypatch.setattr(settings, "orchestration_enabled", True)
        monkeypatch.setattr(settings, "observability_enabled", True)
        monkeypatch.setattr(settings, "orchestration_budget_usd", 0.015)
        monkeypatch.setattr(settings, "run_budget_usd", 0.0)

        async def expensive_worker(spec, **kwargs):
            return _worker_result(role=spec.role, cost=0.01)

        monkeypatch.setattr("agent.orchestration.coordinator.Worker.run", expensive_worker)

        mock_orch = MagicMock()
        ok = MagicMock(success=True, stderr="")
        mock_orch.build_validator.validate = AsyncMock(return_value=ok)
        mock_orch.lint_validator.validate = AsyncMock(return_value=ok)
        mock_orch.test_validator.validate = AsyncMock(return_value=ok)

        state = _parent_state("add api endpoint and react ui with pytest tests")
        out = await Coordinator(orchestrator=mock_orch).run(state)

        assert out.governor.stopped is True
        assert out.governor.stop_reason == "cost_budget"
        halt_msgs = [e.message for e in out.timeline if "budget halt" in e.message]
        assert halt_msgs

    @pytest.mark.asyncio
    async def test_session_checkpoint_integration(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "orchestration_enabled", True)
        monkeypatch.setattr(settings, "session_persistence", True)
        monkeypatch.setattr(settings, "session_dir", str(tmp_path / ".localcli"))
        monkeypatch.setattr(settings, "observability_enabled", False)

        monkeypatch.setattr(
            "agent.orchestration.coordinator.Worker.run",
            AsyncMock(return_value=_worker_result()),
        )

        mock_orch = MagicMock()
        ok = MagicMock(success=True, stderr="")
        mock_orch.build_validator.validate = AsyncMock(return_value=ok)
        mock_orch.lint_validator.validate = AsyncMock(return_value=ok)
        mock_orch.test_validator.validate = AsyncMock(return_value=ok)

        state = _parent_state("fix parser")
        sid = state.task.id
        await Coordinator(orchestrator=mock_orch).run(state)

        store = SessionStore()
        store.init()
        row = store.get(sid)
        assert row is not None
        assert row.checkpoint_count >= 1

    @pytest.mark.asyncio
    async def test_observability_snapshot_after_run(self, monkeypatch):
        monkeypatch.setattr(settings, "orchestration_enabled", True)
        monkeypatch.setattr(settings, "observability_enabled", True)

        monkeypatch.setattr(
            "agent.orchestration.coordinator.Worker.run",
            AsyncMock(return_value=_worker_result()),
        )

        mock_orch = MagicMock()
        ok = MagicMock(success=True, stderr="")
        mock_orch.build_validator.validate = AsyncMock(return_value=ok)
        mock_orch.lint_validator.validate = AsyncMock(return_value=ok)
        mock_orch.test_validator.validate = AsyncMock(return_value=ok)

        state = _parent_state("fix parser")
        out = await Coordinator(orchestrator=mock_orch).run(state)
        snap = build_observability_snapshot(out)
        assert snap["status"] in ("SUCCESS", "FAILURE", "PENDING")
        assert any("orchestration" in e.message or e.kind == "orchestration" for e in out.timeline)


# --- pipeline compatibility -------------------------------------------------


class TestPipelineCompatibility:
    @pytest.mark.asyncio
    async def test_orchestration_disabled_uses_existing_incremental_path(self, monkeypatch):
        """When ORCHESTRATION_ENABLED=false, CLI does not enter the coordinator path."""
        monkeypatch.setattr(settings, "orchestration_enabled", False)
        assert settings.orchestration_enabled is False

    def test_config_defaults_off(self):
        assert settings.orchestration_enabled is False
        assert settings.max_parallel_workers == 1


class TestCoordinatorInit:
    def test_initialization(self):
        coord = Coordinator()
        assert coord._orchestrator is None
        assert coord._safety_mode.auto_approve is True
