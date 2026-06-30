"""Phase 9 — Context Engine tests.

100% local and offline: no network, no live models. The architecture summary is
exercised with a mock client and with ``None`` (machine-only). All filesystem work
happens in a pytest tmp_path fixture repo.
"""

from unittest.mock import patch

import pytest

from agent.context import (
    ContextBundle,
    ContextEngine,
    build_context_bundle,
)
from agent.context.engine import _resolve_context_dir
from agent.config import settings
from agent.llm.providers.base import LLMResult, Usage
from agent.context.schemas import ArchitectureSummary
from agent.planner.core import Planner


# --- fixture repo ------------------------------------------------------------

def _make_fixture_repo(root):
    (root / "package.json").write_text(
        '{\n'
        '  "name": "demo",\n'
        '  "dependencies": {"react": "^18.0.0", "express": "^4.18.0"},\n'
        '  "devDependencies": {"jest": "^29.0.0"},\n'
        '  "scripts": {"start": "node index.js", "test": "jest"},\n'
        '  "bin": {"demo": "cli.js"}\n'
        '}\n', encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\n'
        'name = "demo"\n'
        'dependencies = ["fastapi", "pydantic"]\n\n'
        '[project.scripts]\n'
        'demo = "demo.cli:main"\n\n'
        '[tool.ruff]\n'
        'line-length = 100\n', encoding="utf-8")
    (root / "main.py").write_text(
        "import os\nfrom demo import helper\n\n"
        "def main():\n    return os.getcwd()\n", encoding="utf-8")
    (root / "index.js").write_text(
        "import React from 'react';\nconst x = require('express');\n", encoding="utf-8")
    src = root / "src"
    src.mkdir()
    (src / "app.py").write_text("def handler():\n    return 1\n", encoding="utf-8")
    tests_dir = root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text("def test_handler():\n    assert True\n", encoding="utf-8")
    return root


# --- mock LLM ----------------------------------------------------------------

class _MockPlannerClient:
    model = "mock-planner:latest"

    def __init__(self):
        self.calls = 0

    async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
        self.calls += 1
        data = ArchitectureSummary(
            overview="A Python + Node demo app.",
            key_components=["main.py", "index.js"],
            notes="",
        )
        return LLMResult(data=data, usage=Usage(provider="ollama", model=model))


# --- tests -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_framework_and_entrypoint_detection(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    engine = ContextEngine(repo)
    bundle = await engine.build(llm_client=None)

    assert isinstance(bundle, ContextBundle)
    # Frameworks from manifests across ecosystems.
    assert "React" in bundle.frameworks
    assert "Express" in bundle.frameworks
    assert "FastAPI" in bundle.frameworks
    assert "Pydantic" in bundle.frameworks

    ecos = {t.ecosystem for t in bundle.tech_stack}
    assert {"python", "node"} <= ecos

    # Entry points: well-known files + declared scripts.
    targets = {e.target for e in bundle.entry_points}
    assert "main.py" in targets
    assert "index.js" in targets
    assert any(e.kind == "script" for e in bundle.entry_points)
    assert any("demo" in e.target for e in bundle.entry_points if e.kind == "cli")

    # Conventions inferred locally.
    assert bundle.conventions.test_layout  # tests/ directory detected
    assert "ruff" in bundle.conventions.lint_tools

    # Dependency map captured module-level imports.
    assert "main.py" in bundle.dependency_graph
    assert "os" in bundle.dependency_graph["main.py"]


@pytest.mark.asyncio
async def test_machine_only_bundle_without_llm(tmp_path):
    """No network: a None LLM still yields a usable, complete machine bundle."""
    repo = _make_fixture_repo(tmp_path)
    engine = ContextEngine(repo)
    bundle = await engine.build(llm_client=None)

    assert bundle.architecture_summary is None  # gracefully skipped
    assert bundle.file_count > 0
    assert bundle.frameworks  # detection still ran

    # Cached artifacts were written and are loadable.
    ctx_dir = _resolve_context_dir(repo.resolve())
    assert (ctx_dir / "repo_context.json").exists()
    assert (ctx_dir / "architecture.md").exists()
    assert (ctx_dir / "conventions.md").exists()
    assert (ctx_dir / "dependency_graph.json").exists()


@pytest.mark.asyncio
async def test_mock_llm_produces_architecture_summary(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    engine = ContextEngine(repo)
    client = _MockPlannerClient()
    bundle = await engine.build(llm_client=client)

    assert client.calls == 1
    assert bundle.architecture_summary is not None
    assert "demo app" in bundle.architecture_summary
    # architecture.md embeds the summary.
    md = (_resolve_context_dir(repo.resolve()) / "architecture.md").read_text(encoding="utf-8")
    assert "Overview" in md
    assert "demo app" in md


@pytest.mark.asyncio
async def test_cache_hit_avoids_rescan(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    engine = ContextEngine(repo)

    with patch.object(engine.scanner, "scan", wraps=engine.scanner.scan) as spy:
        first = await engine.build(llm_client=None)
        second = await engine.build(llm_client=None)

    # Second build served from cache: scan() ran exactly once.
    assert spy.call_count == 1
    assert first.fingerprint == second.fingerprint
    assert second.file_count == first.file_count


@pytest.mark.asyncio
async def test_force_refresh_triggers_rescan(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    engine = ContextEngine(repo)

    with patch.object(engine.scanner, "scan", wraps=engine.scanner.scan) as spy:
        await engine.build(llm_client=None)
        await engine.build(llm_client=None, force=True)

    assert spy.call_count == 2


@pytest.mark.asyncio
async def test_bundle_injected_into_planner_prompt(tmp_path):
    """The bundle's text block lands in the planner prompt when provided."""
    repo = _make_fixture_repo(tmp_path)
    bundle = await ContextEngine(repo).build(llm_client=None)

    captured = {}

    class _CaptureClient:
        model = "p:latest"

        async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
            captured["prompt"] = prompt
            from agent.models.schemas import Plan
            return LLMResult(data=Plan(goal="g", summary="s", steps=[]),
                             usage=Usage(provider="ollama", model=model))

    from agent.models.schemas import Task
    planner = Planner(_CaptureClient())
    await planner.create_plan(Task(description="add a feature"), bundle)

    assert "<repository_context>" in captured["prompt"]
    assert "React" in captured["prompt"]


@pytest.mark.asyncio
async def test_planner_prompt_parity_when_no_bundle(tmp_path):
    """Pipeline parity: with no bundle the prompt has no context block."""
    captured = {}

    class _CaptureClient:
        model = "p:latest"

        async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
            captured["prompt"] = prompt
            from agent.models.schemas import Plan
            return LLMResult(data=Plan(goal="g", summary="s", steps=[]),
                             usage=Usage(provider="ollama", model=model))

    from agent.models.schemas import Task
    planner = Planner(_CaptureClient())
    await planner.create_plan(Task(description="add a feature"))

    assert "<repository_context>" not in captured["prompt"]


@pytest.mark.asyncio
async def test_build_context_bundle_disabled_returns_none(tmp_path, monkeypatch):
    """The CONTEXT_ENGINE_ENABLED gate lives in one place and returns None when off."""
    monkeypatch.setattr(settings, "context_engine_enabled", False)
    result = await build_context_bundle(tmp_path)
    assert result is None
