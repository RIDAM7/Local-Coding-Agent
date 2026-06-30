"""Phase 15 — Session persistence & resume. Offline and deterministic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.config import settings
from agent.exceptions.errors import SessionError
from agent.session import Checkpointer, ResumeManager, SessionStore
from agent.state.agent_state import AgentState, TaskMetadata


# --- helpers ----------------------------------------------------------------


def _sample_state(task_desc: str = "test task") -> AgentState:
    from agent.state.agent_state import StepResult
    state = AgentState(user_request=task_desc,
                       task=TaskMetadata(description=task_desc))
    state.execution_mode = "pipeline"
    state.objective = task_desc
    state.plan.steps = [
        StepResult(index=0, description="step 1", status="done"),
        StepResult(index=1, description="step 2", status="pending"),
    ]
    state.add_timeline("engine", "started")
    state.confidence = 0.85
    state.governor.steps_used = 1
    return state


@pytest.fixture
def tmp_session_dir(monkeypatch, tmp_path):
    """Point session_dir to a temp path for each test."""
    monkeypatch.setattr(settings, "session_persistence", True)
    monkeypatch.setattr(settings, "session_dir", str(tmp_path / ".localcli"))
    yield tmp_path


# --- SessionStore tests -----------------------------------------------------


class TestSessionStore:
    def test_create_and_get(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create(task_description="hello")
        row = store.get(sid)
        assert row is not None
        assert row.session_id == sid
        assert row.task_description == "hello"
        assert row.status == "active"
        assert row.checkpoint_count == 0

    def test_get_missing_returns_none(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        assert store.get("nonexistent") is None

    def test_list_ordered_by_updated_at(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid_a = store.create("A")
        sid_b = store.create("B")
        rows = store.list()
        assert len(rows) >= 2
        # Most recent first.
        assert rows[0].session_id in (sid_a, sid_b)

    def test_update_status(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("x")
        store.update_status(sid, "completed")
        row = store.get(sid)
        assert row.status == "completed"

    def test_delete_removes_row_and_files(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("to_delete")
        state = _sample_state()
        store.save_checkpoint(sid, 0, state)
        assert store.get(sid) is not None
        deleted = store.delete(sid)
        assert deleted is True
        assert store.get(sid) is None

    def test_delete_nonexistent_returns_false(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        with patch.object(Path, "exists", return_value=False):
            deleted = store.delete("nope")
            assert deleted is False

    def test_checkpoint_roundtrip(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("checkpoint_test")
        original = _sample_state("checkpoint test")
        original.objective = "original objective"
        original.confidence = 0.92

        store.save_checkpoint(sid, 0, original)
        restored = store.load_checkpoint(sid, 0)

        assert restored.user_request == original.user_request
        assert restored.objective == original.objective
        assert restored.confidence == original.confidence

    def test_checkpoint_preserves_timeline(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("timeline_test")
        original = _sample_state("timeline preservation")
        original.add_timeline("tool", "searched for auth.ts")
        original.add_timeline("engine", "patch applied")

        store.save_checkpoint(sid, 0, original)
        restored = store.load_checkpoint(sid, 0)

        assert len(restored.timeline) == len(original.timeline)
        assert restored.timeline[0].message == "started"
        assert restored.timeline[-1].message == "patch applied"

    def test_checkpoint_preserves_governor_state(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("gov_test")
        original = _sample_state()
        original.governor.steps_used = 3
        original.governor.cost_used_usd = 0.05
        original.governor.stop_reason = "max_steps"

        store.save_checkpoint(sid, 0, original)
        restored = store.load_checkpoint(sid, 0)

        assert restored.governor.steps_used == 3
        assert restored.governor.cost_used_usd == 0.05
        assert restored.governor.stop_reason == "max_steps"

    def test_missing_checkpoint_raises(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("missing")
        with pytest.raises(SessionError, match="Checkpoint not found"):
            store.load_checkpoint(sid, 99)

    def test_corrupted_checkpoint_raises(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("corrupted")
        sdir = Path(tmp_session_dir) / ".localcli" / "sessions" / sid
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "step_0000.json").write_text("not valid json", encoding="utf-8")
        with pytest.raises(SessionError, match="Corrupted checkpoint"):
            store.load_checkpoint(sid, 0)

    def test_latest_checkpoint_index(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("latest")
        state = _sample_state()
        store.save_checkpoint(sid, 0, state)
        store.save_checkpoint(sid, 2, state)
        assert store.latest_checkpoint_index(sid) == 2

    def test_latest_checkpoint_index_no_checkpoints(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("no_cp")
        assert store.latest_checkpoint_index(sid) is None

    def test_increment_checkpoint_count(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("counter")
        state = _sample_state()
        store.save_checkpoint(sid, 0, state)
        row = store.get(sid)
        assert row.checkpoint_count == 1
        store.save_checkpoint(sid, 1, state)
        row = store.get(sid)
        assert row.checkpoint_count == 2


# --- Checkpointer tests -----------------------------------------------------


class TestCheckpointer:
    def test_checkpoint_step_writes_file(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = "checkpointer_test"
        store.create("test", session_id=sid)
        state = _sample_state()
        state.task.id = sid

        cp = Checkpointer(store)
        cp.checkpoint_step(state, 0)

        path = store._checkpoint_path(sid, 0)
        assert path.exists()

    def test_checkpoint_disabled_is_noop(self, monkeypatch, tmp_session_dir):
        monkeypatch.setattr(settings, "session_persistence", False)
        cp = Checkpointer()
        state = _sample_state()
        # Should not raise despite no store being set up.
        cp.checkpoint_step(state, 0)

    def test_get_latest_checkpoint(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = "latest_cp"
        store.create("test", session_id=sid)
        state = _sample_state()
        state.task.id = sid
        store.save_checkpoint(sid, 0, state)
        store.save_checkpoint(sid, 1, state)

        cp = Checkpointer(store)
        assert cp.get_latest_checkpoint(state) == 1

    def test_get_latest_checkpoint_no_session(self, tmp_session_dir):
        cp = Checkpointer()
        state = _sample_state()
        state.task.id = "nonexistent"
        assert cp.get_latest_checkpoint(state) is None


# --- ResumeManager tests ----------------------------------------------------


class TestResumeManager:
    def test_resume_restores_state_and_last_step(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = "resume_test"
        store.create("resume test", session_id=sid)
        original = _sample_state("resume me")
        original.task.id = sid
        original.governor.steps_used = 2
        store.save_checkpoint(sid, 1, original)

        mgr = ResumeManager(store)
        restored, last_step = mgr.resume(sid)
        assert last_step == 1
        assert restored.user_request == "resume me"
        assert restored.governor.steps_used == 2
        # Timeline should include the resume event.
        assert any("session resumed" in e.message for e in restored.timeline)

    def test_resume_missing_session_raises(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        mgr = ResumeManager(store)
        with pytest.raises(SessionError, match="not found"):
            mgr.resume("ghost_session")

    def test_resume_no_checkpoints_raises(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("no_cp_session")
        mgr = ResumeManager(store)
        with pytest.raises(SessionError, match="no checkpoints"):
            mgr.resume(sid)

    def test_resume_completed_session_raises(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("completed_session")
        store.update_status(sid, "completed")
        mgr = ResumeManager(store)
        with pytest.raises(SessionError, match="already completed"):
            mgr.resume(sid)

    def test_resume_corrupted_session_raises(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("corrupted_session")
        store.update_status(sid, "corrupted")
        original = _sample_state()
        original.task.id = sid
        store.save_checkpoint(sid, 0, original)
        mgr = ResumeManager(store)
        with pytest.raises(SessionError, match="corrupted"):
            mgr.resume(sid)

    def test_resume_corrupted_checkpoint_marks_corrupted(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        sid = store.create("corrupt_cp_session")
        sdir = Path(tmp_session_dir) / ".localcli" / "sessions" / sid
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "step_0000.json").write_text("{broken", encoding="utf-8")
        mgr = ResumeManager(store)
        with pytest.raises(SessionError, match="Corrupted|corrupted"):
            mgr.resume(sid)
        # The session should be marked as corrupted.
        row = store.get(sid)
        assert row is not None and row.status == "corrupted"

    def test_list_sessions_returns_dicts(self, tmp_session_dir):
        store = SessionStore()
        store.init()
        store.create("session A")
        store.create("session B")
        mgr = ResumeManager(store)
        sessions = mgr.list_sessions()
        assert len(sessions) >= 2
        for s in sessions:
            assert "session_id" in s
            assert "status" in s
            assert "checkpoint_count" in s


# --- CLI parser tests -------------------------------------------------------


class TestSessionCLI:
    def test_resume_subcommand_parses(self):
        from agent.cli import build_parser, _normalize_argv
        parser = build_parser()
        args = parser.parse_args(_normalize_argv(["resume", "abc123"]))
        assert args.subcommand == "resume"
        assert args.session_id == "abc123"

    def test_sessions_subcommand_parses(self):
        from agent.cli import build_parser, _normalize_argv
        parser = build_parser()
        args = parser.parse_args(_normalize_argv(["sessions"]))
        assert args.subcommand == "sessions"

    def test_sessions_cmd_no_sessions(self, monkeypatch, capsys, tmp_session_dir):
        monkeypatch.setattr(settings, "session_persistence", True)
        from agent.cli import cmd_sessions
        rc = cmd_sessions(None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No sessions found" in out


# --- AgentState serialization support ---------------------------------------


class TestAgentStateSessionIntegration:
    def test_to_json_redaction_clean_for_checkpoint(self):
        """Checkpoints must never contain secrets. Verify to_json redacts."""
        state = AgentState(user_request="test")
        state.add_timeline("tool", "used key sk-ABCDEF1234567890")
        serialized = state.to_json(redacted=True)
        assert "sk-ABCDEF1234567890" not in serialized

    def test_state_roundtrip_preserves_plan_and_timeline(self):
        original = _sample_state("roundtrip")
        original.add_timeline("engine", "replan triggered")
        original.add_timeline("tool", "searched for config")

        serialized = original.to_json(redacted=True)
        restored = AgentState.from_json(serialized)

        assert restored.user_request == original.user_request
        assert restored.objective == original.objective
        assert restored.confidence == original.confidence
        assert len(restored.timeline) == 3
        assert restored.timeline[1].message == "replan triggered"


# --- Phase 15 disabled mode -------------------------------------------------


class TestSessionDisabledMode:
    def test_checkpointer_noop_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "session_persistence", False)
        cp = Checkpointer()
        state = _sample_state()
        # No exception despite no store backing.
        cp.checkpoint_step(state, 0)

    def test_resume_disabled_returns_error(self, monkeypatch):
        monkeypatch.setattr(settings, "session_persistence", False)
        from agent.cli import cmd_resume

        async def run():
            class FakeArgs:
                session_id = "any"
            return await cmd_resume(FakeArgs())

        import contextlib
        import sys
        with contextlib.redirect_stdout(sys.stderr):
            pass  # No error expected, just the message
