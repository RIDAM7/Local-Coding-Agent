"""Phase 13 - Repository Graph MVP. Offline and deterministic."""

import pytest

from agent import cli
from agent.config import settings
from agent.context import ContextEngine
from agent.graph import GraphBuilder, ImpactAnalyzer
from agent.graph.queries import build_or_load_graph
from agent.llm.providers.base import LLMResult, Usage
from agent.models.schemas import Plan
from agent.planner.core import Planner
from agent.state.agent_state import AgentState


def _make_graph_repo(root):
    pkg = root / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text("class User:\n    pass\n", encoding="utf-8")
    (pkg / "service.py").write_text(
        "from app.models import User\n\n"
        "def build_user():\n    return User()\n",
        encoding="utf-8",
    )
    (pkg / "api.py").write_text(
        "from app.service import build_user\n\n"
        "def handler():\n    return build_user()\n",
        encoding="utf-8",
    )
    web = root / "web"
    web.mkdir()
    (web / "util.ts").write_text("export function formatName(x: string) { return x }\n", encoding="utf-8")
    (web / "view.ts").write_text(
        "import { formatName } from './util'\n"
        "export const view = formatName('a')\n",
        encoding="utf-8",
    )
    return root


def test_graph_construction_import_edges_symbols_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_graph_enabled", True)
    repo = _make_graph_repo(tmp_path)
    builder = GraphBuilder(repo)
    graph = builder.build(force=True)

    assert graph.modules["app/service.py"].dependencies == ["app/models.py"]
    assert graph.modules["app/api.py"].dependencies == ["app/service.py"]
    assert graph.modules["web/view.ts"].dependencies == ["web/util.ts"]
    assert any(e.source == "app/api.py" and e.target == "app/service.py" for e in graph.edges)
    assert "User" in graph.symbol_owners.get("app/models.py", [])
    assert builder.store.path.exists()

    loaded = builder.store.load()
    assert loaded is not None
    assert loaded.fingerprint == graph.fingerprint
    assert loaded.modules["web/view.ts"].dependencies == ["web/util.ts"]


def test_impact_query_and_agentstate_evidence(tmp_path):
    repo = _make_graph_repo(tmp_path)
    graph = GraphBuilder(repo).build(force=True)
    analyzer = ImpactAnalyzer(graph)
    state = AgentState(user_request="change service")

    dependents = analyzer.record_impact(state, "app/service.py")

    assert dependents == ["app/api.py"]
    assert analyzer.dependents("missing.py") == []
    assert state.evidence[-1].kind == "graph_impact"
    assert "1 dependent" in state.evidence[-1].detail


@pytest.mark.asyncio
async def test_context_engine_reuses_repository_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_graph_enabled", True)
    repo = _make_graph_repo(tmp_path)

    bundle = await ContextEngine(repo).build(llm_client=None)

    assert bundle.repository_graph["modules"] >= 5
    assert bundle.repository_graph["edges"] >= 3
    assert bundle.dependency_graph["app/api.py"] == ["app/service.py"]
    assert (repo / ".localcli" / "graph.json").exists()


@pytest.mark.asyncio
async def test_planner_prompt_includes_graph_visibility(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_graph_enabled", True)
    repo = _make_graph_repo(tmp_path)
    bundle = await ContextEngine(repo).build(llm_client=None)
    captured = {}

    class _CaptureClient:
        model = "planner:test"

        async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
            captured["prompt"] = prompt
            return LLMResult(data=Plan(goal="g", summary="s", steps=[]),
                             usage=Usage(provider="ollama", model=model))

    from agent.models.schemas import Task
    await Planner(_CaptureClient()).create_plan(Task(description="edit service"), bundle)

    assert "Repository graph:" in captured["prompt"]
    assert "app/api.py" in captured["prompt"]
    assert "app/service.py" in captured["prompt"]


@pytest.mark.asyncio
async def test_disabled_graph_mode_leaves_context_without_graph(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings, "repo_graph_enabled", False)
    repo = _make_graph_repo(tmp_path)

    assert build_or_load_graph(repo) is None
    bundle = await ContextEngine(repo).build(llm_client=None)
    assert bundle.repository_graph == {}

    monkeypatch.setattr(cli.settings, "workspace_dir", str(repo))
    args = cli.build_parser().parse_args(["graph", "impact", "app/service.py"])
    rc = cli.cmd_graph(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "disabled" in out


def test_graph_regenerates_after_repository_change(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_graph_enabled", True)
    repo = _make_graph_repo(tmp_path)
    builder = GraphBuilder(repo)
    first = builder.build()
    assert "app/consumer.py" not in first.modules

    (repo / "app" / "consumer.py").write_text(
        "from app.service import build_user\nvalue = build_user()\n",
        encoding="utf-8",
    )
    second = builder.build()

    assert "app/consumer.py" in second.modules
    assert "app/consumer.py" in ImpactAnalyzer(second).dependents("app/service.py")


def test_graph_cli_impact_lists_dependents(tmp_path, monkeypatch, capsys):
    repo = _make_graph_repo(tmp_path)
    monkeypatch.setattr(cli.settings, "workspace_dir", str(repo))
    monkeypatch.setattr(cli.settings, "repo_graph_enabled", True)
    monkeypatch.setattr(cli.settings, "graph_dir", ".localcli")

    rc = cli.main(["graph", "impact", "app/service.py"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Impact for app/service.py: 1 dependent file(s)" in out
    assert "app/api.py" in out
