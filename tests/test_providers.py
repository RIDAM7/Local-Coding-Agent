"""Phase 2 cloud provider tests. The HTTP layer (aiohttp) is mocked — no live
network calls are made."""

import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from agent.config import settings
from agent.llm.providers.base import LLMResult, Usage
from agent.llm.providers.openai import OpenAIClient
from agent.llm.providers.anthropic import AnthropicClient
from agent.llm.providers.google import GoogleClient


class Sample(BaseModel):
    name: str
    value: int


# --- aiohttp fakes -----------------------------------------------------------

class _FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)


class _FakeSession:
    """A single fake session reused across retries; pops responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json or {}})
        return self._responses.pop(0)


def _patch_session(responses):
    session = _FakeSession([_FakeResponse(200, p) for p in responses])
    return session, patch("aiohttp.ClientSession", lambda *a, **k: session)


# --- per-provider success payload builders -----------------------------------

def _openai_payload(content, pin=11, pout=7):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": pin, "completion_tokens": pout}}


def _anthropic_payload(content, pin=11, pout=7):
    return {"content": [{"type": "text", "text": content}],
            "usage": {"input_tokens": pin, "output_tokens": pout}}


def _google_payload(content, pin=11, pout=7):
    return {"candidates": [{"content": {"parts": [{"text": content}]}}],
            "usageMetadata": {"promptTokenCount": pin, "candidatesTokenCount": pout}}


GOOD = '{"name": "widget", "value": 42}'


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-openai")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-anthropic")
    monkeypatch.setattr(settings, "google_api_key", "sk-test-google")


@pytest.mark.asyncio
@pytest.mark.parametrize("ClientCls,provider,payload_fn,model", [
    (OpenAIClient, "openai", _openai_payload, "gpt-4o-mini"),
    (AnthropicClient, "anthropic", _anthropic_payload, "claude-3-5-sonnet-20240620"),
    (GoogleClient, "google", _google_payload, "gemini-1.5-pro"),
])
async def test_cloud_provider_parses_success(ClientCls, provider, payload_fn, model):
    session, sp = _patch_session([payload_fn(GOOD)])
    with sp:
        client = ClientCls()
        result = await client.generate_structured(model, "make a widget", Sample)

    assert isinstance(result, LLMResult)
    assert isinstance(result.data, Sample)
    assert result.data.name == "widget" and result.data.value == 42
    assert isinstance(result.usage, Usage)
    assert result.usage.provider == provider
    assert result.usage.model == model
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert len(session.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ClientCls,payload_fn", [
    (OpenAIClient, _openai_payload),
    (AnthropicClient, _anthropic_payload),
    (GoogleClient, _google_payload),
])
async def test_cloud_provider_retry_repair(ClientCls, payload_fn):
    # First response is junk (no JSON), second is valid -> should succeed via retry.
    bad = payload_fn("Sorry, I cannot comply.", pin=5, pout=2)
    good = payload_fn('{"name": "ok", "value": 1}', pin=8, pout=3)
    session, sp = _patch_session([bad, good])
    with sp:
        client = ClientCls()
        result = await client.generate_structured("some-model", "do it", Sample)

    assert result.data.name == "ok" and result.data.value == 1
    # Usage accumulates across the failed + successful attempts.
    assert result.usage.input_tokens == 13
    assert result.usage.output_tokens == 5
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_openai_uses_configured_base_url(monkeypatch):
    monkeypatch.setattr(settings, "openai_base_url", "https://openrouter.ai/api/v1")
    session, sp = _patch_session([_openai_payload(GOOD)])
    with sp:
        client = OpenAIClient()
        assert client.base_url == "https://openrouter.ai/api/v1"
        await client.generate_structured("anthropic/claude-3.5-sonnet", "hi", Sample)

    assert session.calls[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"


@pytest.mark.asyncio
async def test_missing_key_raises_before_network(monkeypatch):
    from agent.exceptions.errors import LLMError
    monkeypatch.setattr(settings, "openai_api_key", "")
    client = OpenAIClient()
    with pytest.raises(LLMError):
        await client.generate_structured("gpt-4o-mini", "hi", Sample)
