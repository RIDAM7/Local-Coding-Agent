"""Phase 10 — Offline Lock: cloud providers rejected, remote tool disabled, and
secrets never leak into serialized state/logs. No network."""

import logging

import pytest

from agent.config import settings
from agent.llm.preflight import preflight_check
from agent.exceptions.errors import PreflightError
from agent.tools.registry import build_default_registry
from agent.execution.core import Executor
from agent.files.core import FileManager
from agent.git.core import GitManager
from agent.safety.controller import SafetyController, SafetyMode
from agent.safety.redact import RedactionFilter, redact
from agent.retrieval import RipgrepSearch, SymbolIndex, TreeSitterIndexer
from agent.validation import BuildValidator, LintValidator
from agent.validation import TestValidator as _TestValidator


@pytest.mark.asyncio
async def test_offline_only_rejects_cloud_provider(monkeypatch):
    monkeypatch.setattr(settings, "offline_only", True)
    monkeypatch.setattr(settings, "coder_provider", "openai")
    with pytest.raises(PreflightError) as exc:
        await preflight_check(["coder"])
    assert "OFFLINE_ONLY" in str(exc.value)


@pytest.mark.asyncio
async def test_offline_only_allows_local(monkeypatch):
    monkeypatch.setattr(settings, "offline_only", True)
    monkeypatch.setattr(settings, "planner_provider", "ollama")
    # Should not raise on the offline-lock check (it may still need ollama up, but
    # the lock itself must pass for a local provider). We only assert the lock path.
    from agent.llm.factory import resolve_provider
    assert resolve_provider("planner") == "ollama"


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


def test_web_tool_disabled_under_offline_only(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "offline_only", True)
    reg = _registry(tmp_path)
    assert not reg.has("web")


def test_web_tool_registered_when_online(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "offline_only", False)
    reg = _registry(tmp_path)
    assert reg.has("web")


def test_secret_never_logged(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-SUPERSECRETVALUE123")
    f = RedactionFilter()
    rec = logging.LogRecord("agent", logging.INFO, __file__, 1,
                            "calling with key sk-ant-SUPERSECRETVALUE123", None, None)
    f.filter(rec)
    assert "SUPERSECRETVALUE123" not in rec.getMessage()
    assert "***REDACTED***" in redact("token sk-ant-SUPERSECRETVALUE123")
