"""Phase 10 — Capability Detector: static table, mocked probe, caching, and the
unknown-model -> pipeline degrade path. No network."""

import pytest

from agent.config import settings
from agent.engine.capability_detector import CapabilityDetector, Capabilities
from agent.llm.providers.base import LLMResult, Usage
from agent.engine.capability_detector import _ProbeSchema


@pytest.mark.asyncio
async def test_static_table_known_model_is_agent_capable(tmp_path):
    det = CapabilityDetector(cache_path=tmp_path / "caps.json")
    caps = await det.detect("anthropic", "claude-3-5-sonnet-20240620")
    assert caps.source == "table"
    assert caps.is_agent_capable()


@pytest.mark.asyncio
async def test_static_table_known_local_model(tmp_path):
    det = CapabilityDetector(cache_path=tmp_path / "caps.json")
    caps = await det.detect("ollama", "qwen2.5-coder:32b")
    assert caps.source == "table"
    assert caps.tool_calling and caps.structured_output


class _ProbeClient:
    def __init__(self):
        self.calls = 0

    async def generate_structured(self, model, prompt, schema, *, max_tokens=4096):
        self.calls += 1
        return LLMResult(data=_ProbeSchema(ok=True, answer="pong"),
                         usage=Usage(provider="ollama", model=model))


@pytest.mark.asyncio
async def test_unknown_model_probe_then_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "capability_probe", True)
    det = CapabilityDetector(cache_path=tmp_path / "caps.json")
    client = _ProbeClient()

    caps = await det.detect("ollama", "mystery-model:7b", client=client)
    assert caps.source == "probe"
    assert caps.structured_output is True
    # Probe does not confirm tool calling -> degrades to pipeline.
    assert caps.is_agent_capable() is False
    assert client.calls == 1

    # Second call is served from cache (no second probe).
    again = await det.detect("ollama", "mystery-model:7b", client=client)
    assert again.structured_output is True
    assert client.calls == 1


@pytest.mark.asyncio
async def test_unknown_model_no_probe_degrades_to_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "capability_probe", False)
    det = CapabilityDetector(cache_path=tmp_path / "caps.json")
    caps = await det.detect("ollama", "totally-unknown", client=None)
    assert caps.source == "default"
    assert caps.is_agent_capable() is False


def test_is_agent_capable_requires_all_minimums():
    assert not Capabilities(structured_output=True, tool_calling=False,
                            context_window=128000).is_agent_capable()
    assert not Capabilities(structured_output=True, tool_calling=True,
                            context_window=1000).is_agent_capable()
    assert Capabilities(structured_output=True, tool_calling=True,
                        context_window=32000).is_agent_capable()
