"""Phase 7B git integration tests (use a real temp git repo)."""

import json
import shutil
import subprocess
import contextlib
from unittest.mock import patch, Mock, AsyncMock

import pytest

from agent.config import settings
from agent.git.core import GitManager
from agent.orchestrator import Orchestrator
from agent.models.schemas import Plan, Patch, FileOperation, ValidationDiagnostic
from agent.reflection.schemas import ReflectionReport, ReflectionResult
from agent.review.router import ReviewDecision
from agent.safety.controller import SafetyMode

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def _git(path, *args):
    return subprocess.run(["git", *args], cwd=str(path), capture_output=True, text=True)


def _init_repo(path):
    _git(path, "init")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "tester")
    (path / "README.md").write_text("init\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-m", "init")


def _diag(stage, success=True):
    return ValidationDiagnostic(stage=stage, command="", success=success,
                                stdout="", stderr="", exit_code=0 if success else 1, duration=0.0)


def _pipeline(modified_patch):
    return [
        patch('agent.planner.core.Planner.create_plan',
              AsyncMock(return_value=Plan(goal="g", summary="s", steps=[]))),
        patch('agent.coder.core.Coder.generate_patch', AsyncMock(return_value=modified_patch)),
        patch('agent.retrieval.retrieval_manager.RetrievalManager.search_context',
              AsyncMock(return_value=Mock(results=[]))),
        patch('agent.repair.constraints.ConstraintExtractor.extract',
              AsyncMock(return_value=Mock(success=True, constraints=[]))),
        patch('agent.validation.BuildValidator.validate', return_value=_diag("BUILD", True)),
        patch('agent.validation.LintValidator.validate', return_value=_diag("LINT", True)),
        patch('agent.validation.TestValidator.validate', return_value=_diag("TEST", True)),
        patch('agent.reflection.manager.ReflectionManager.reflect',
              AsyncMock(return_value=ReflectionReport(result=ReflectionResult.PASS, critiques=[],
                                                      summary="", execution_time_ms=0, model_name="t"))),
        patch('agent.review.router.ReviewRouter.route', return_value=ReviewDecision.APPROVE),
    ]


@contextlib.contextmanager
def _apply(patches):
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def _load(report_path):
    with open(report_path.replace('.md', '.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


# --- GitManager unit ---------------------------------------------------------

def test_is_git_repo_true_and_false(tmp_path):
    non_repo = tmp_path / "plain"
    non_repo.mkdir()
    assert GitManager(non_repo).is_git_repo() is False

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    assert GitManager(repo).is_git_repo() is True


def test_branch_and_commit_with_redacted_message(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    gm = GitManager(repo)

    branch = gm.create_task_branch("Add a feature")
    assert branch.startswith("localcli/")
    assert _git(repo, "branch", "--show-current").stdout.strip() == branch

    (repo / "f.txt").write_text("data\n", encoding="utf-8")
    commit = gm.commit_all("localcli: use key sk-SECRETKEY123456 now")
    assert commit
    body = _git(repo, "log", "-1", "--format=%B").stdout
    assert "sk-SECRETKEY123456" not in body   # redacted
    assert "REDACTED" in body


# --- Orchestrator integration ------------------------------------------------

@pytest.mark.asyncio
async def test_commit_lands_on_success(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.setattr(settings, "git_integration", True)

    op = FileOperation(type="create_file", path="new.py", content="print('hi')\n")
    with _apply(_pipeline(Patch(operations=[op]))):
        orch = Orchestrator(workspace_path=tmp_path, claude_enabled=False,
                            safety_mode=SafetyMode(auto_approve=True))
        report_path = await orch.run("add new file")

    assert _git(tmp_path, "branch", "--show-current").stdout.strip().startswith("localcli/")
    assert "localcli:" in _git(tmp_path, "log", "--oneline").stdout
    report = _load(report_path)
    assert report["git_branch"].startswith("localcli/")
    assert report["git_commit"]


@pytest.mark.asyncio
async def test_non_git_workspace_falls_back_without_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "git_integration", True)  # on, but not a git repo

    op = FileOperation(type="create_file", path="x.py", content="1\n")
    with _apply(_pipeline(Patch(operations=[op]))):
        orch = Orchestrator(workspace_path=tmp_path, claude_enabled=False,
                            safety_mode=SafetyMode(auto_approve=True))
        report_path = await orch.run("create x")

    report = _load(report_path)
    assert report["git_branch"] is None          # snapshot-only fallback
    assert (tmp_path / "x.py").exists()           # run still applied the file


@pytest.mark.asyncio
async def test_dry_run_makes_no_commit(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.setattr(settings, "git_integration", True)
    head_before = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()

    op = FileOperation(type="create_file", path="y.py", content="1\n")
    with _apply(_pipeline(Patch(operations=[op]))):
        orch = Orchestrator(workspace_path=tmp_path, claude_enabled=False,
                            safety_mode=SafetyMode(dry_run=True))
        report_path = await orch.run("create y")

    assert _git(tmp_path, "rev-parse", "HEAD").stdout.strip() == head_before  # no commit
    report = _load(report_path)
    assert report["git_branch"] is None           # git inactive under --dry-run
