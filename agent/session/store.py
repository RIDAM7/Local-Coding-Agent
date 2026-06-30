"""Phase 15 — SQLite-backed session store.

Maintains a lightweight session index in ``.localcli/sessions.db``.
Each row records a session's id, task description, created_at, status,
and the checkpoint count. The actual AgentState snapshots are stored
as redacted JSON files under ``.localcli/sessions/<session_id>/``.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.config import settings
from agent.exceptions.errors import SessionError
from agent.state.agent_state import AgentState


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_dir(session_id: str) -> Path:
    base = Path(settings.session_dir) if Path(settings.session_dir).is_absolute() else settings.get_workspace_path() / settings.session_dir
    return base / "sessions" / session_id


def _db_path() -> Path:
    base = Path(settings.session_dir) if Path(settings.session_dir).is_absolute() else settings.get_workspace_path() / settings.session_dir
    base.mkdir(parents=True, exist_ok=True)
    return base / "sessions.db"


# --- Schema version for forward compatibility --------------------------------

SCHEMA_VERSION = 1


class SessionRow:
    """A read-only view of a session row from the store."""

    def __init__(self, row: sqlite3.Row):
        self.session_id: str = row["session_id"]
        self.task_description: str = row["task_description"]
        self.created_at: str = row["created_at"]
        self.updated_at: str = row["updated_at"]
        self.status: str = row["status"]          # active | paused | completed | failed | corrupted
        self.checkpoint_count: int = row["checkpoint_count"]
        self.schema_version: int = row["schema_version"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_description": self.task_description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "checkpoint_count": self.checkpoint_count,
            "schema_version": self.schema_version,
        }


class SessionStore:
    """Lightweight SQLite-backed session index.

    Thread-safe for single-process use (sqlite3 in ``check_same_thread=False``
    for async-friendly access). Only metadata lives here; checkpoints are JSON
    files on disk.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    # --- connection / init ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init(self) -> None:
        """Create the sessions table if it does not exist."""
        conn = self._connect()
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id      TEXT PRIMARY KEY,
                task_description TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'active',
                checkpoint_count INTEGER NOT NULL DEFAULT 0,
                schema_version  INTEGER NOT NULL DEFAULT {SCHEMA_VERSION}
            )
        """)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- CRUD ----------------------------------------------------------------

    def create(self, task_description: str = "", session_id: Optional[str] = None) -> str:
        """Create a new session row. Returns the session_id."""
        self.init()
        sid = session_id or uuid.uuid4().hex[:12]
        now = _utcnow()
        conn = self._connect()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, task_description, created_at, updated_at, status, checkpoint_count, schema_version) VALUES (?, ?, ?, ?, 'active', 0, ?)",
            (sid, task_description, now, now, SCHEMA_VERSION),
        )
        conn.commit()
        return sid

    def get(self, session_id: str) -> Optional[SessionRow]:
        """Get a session by id. Returns None if not found."""
        self.init()
        conn = self._connect()
        cur = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        return SessionRow(row) if row else None

    def list(self, limit: int = 50) -> List[SessionRow]:
        """List sessions ordered by updated_at descending."""
        self.init()
        conn = self._connect()
        cur = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [SessionRow(r) for r in cur.fetchall()]

    def update_status(self, session_id: str, status: str) -> None:
        """Update session status (active|paused|completed|failed|corrupted)."""
        self.init()
        conn = self._connect()
        now = _utcnow()
        conn.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
            (status, now, session_id),
        )
        conn.commit()

    def increment_checkpoint(self, session_id: str) -> None:
        """Increment the checkpoint counter."""
        self.init()
        conn = self._connect()
        now = _utcnow()
        conn.execute(
            "UPDATE sessions SET checkpoint_count = checkpoint_count + 1, updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        conn.commit()

    def delete(self, session_id: str) -> bool:
        """Delete a session row and its checkpoint directory. Returns True if deleted."""
        self.init()
        conn = self._connect()
        cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        deleted = cur.rowcount > 0
        # Remove checkpoint files on disk.
        sdir = _session_dir(session_id)
        if sdir.exists():
            import shutil
            shutil.rmtree(str(sdir), ignore_errors=True)
        return deleted

    # --- checkpoint file helpers ---------------------------------------------

    def _checkpoint_path(self, session_id: str, step_index: int) -> Path:
        return _session_dir(session_id) / f"step_{step_index:04d}.json"

    def save_checkpoint(self, session_id: str, step_index: int, state: AgentState) -> Path:
        """Serialize the redacted AgentState to a checkpoint file."""
        sdir = _session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path(session_id, step_index)
        state_json = state.to_json(redacted=True)
        path.write_text(state_json, encoding="utf-8")
        self.increment_checkpoint(session_id)
        return path

    def load_checkpoint(self, session_id: str, step_index: int) -> AgentState:
        """Deserialize a checkpoint file back to AgentState."""
        path = self._checkpoint_path(session_id, step_index)
        if not path.exists():
            raise SessionError(f"Checkpoint not found: {path}")
        raw = path.read_text(encoding="utf-8")
        try:
            return AgentState.from_json(raw)
        except Exception as e:
            raise SessionError(f"Corrupted checkpoint {path}: {e}")

    def latest_checkpoint_index(self, session_id: str) -> Optional[int]:
        """Find the highest step_index among checkpoint files for a session."""
        sdir = _session_dir(session_id)
        if not sdir.exists():
            return None
        indices = []
        for f in sdir.iterdir():
            if f.name.startswith("step_") and f.suffix == ".json":
                try:
                    indices.append(int(f.stem.split("_")[1]))
                except (ValueError, IndexError):
                    continue
        return max(indices) if indices else None

    def list_checkpoints(self, session_id: str) -> List[int]:
        """Return sorted list of checkpoint step indices for a session."""
        sdir = _session_dir(session_id)
        if not sdir.exists():
            return []
        indices = []
        for f in sdir.iterdir():
            if f.name.startswith("step_") and f.suffix == ".json":
                try:
                    indices.append(int(f.stem.split("_")[1]))
                except (ValueError, IndexError):
                    continue
        return sorted(indices)
