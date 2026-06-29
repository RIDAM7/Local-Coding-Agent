"""Phase 2 preflight tests. No live network — the Ollama HTTP call is mocked."""

import json
from unittest.mock import patch

import pytest

from agent.config import settings
from agent.exceptions.errors import PreflightError
from agent.llm.preflight import preflight_check


# --- aiohttp fake for the Ollama /api/tags GET -------------------------------

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
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def get(self, url):
        return _FakeResponse(200, self._payload)


def _patch_ollama(model_names):
    payload = {"models": [{"name": n} for n in model_names]}
    return patch("aiohttp.ClientSession", lambda *a, **k: _FakeSession(payload))


# --- cloud key checks (no network) -------------------------------------------

@pytest.mark.asyncio
async def test_missing_cloud_key_raises_clear_secret_free_error(monkeypatch):
    # planner OK on anthropic (key set), coder on google with NO key -> error.
    monkeypatch.setattr(settings, "planner_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "ANTHROPIC_SECRET_XYZ")
    monkeypatch.setattr(settings, "coder_provider", "google")
    monkeypatch.setattr(settings, "google_api_key", "")

    with pytest.raises(PreflightError) as ei:
        await preflight_check(["planner", "coder"])

    msg = str(ei.value)
    assert "coder" in msg
    assert "google" in msg
    assert "GOOGLE_API_KEY" in msg
    # The configured secret for the OTHER role must never leak into the message.
    assert "ANTHROPIC_SECRET_XYZ" not in msg


@pytest.mark.asyncio
async def test_passes_when_cloud_roles_configured(monkeypatch):
    monkeypatch.setattr(settings, "planner_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "x")
    monkeypatch.setattr(settings, "coder_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "y")
    monkeypatch.setattr(settings, "openai_base_url", "https://api.openai.com/v1")

    # Should not raise (no network: both roles are cloud).
    await preflight_check(["planner", "coder"])


# --- ollama checks (mocked HTTP) ---------------------------------------------

@pytest.mark.asyncio
async def test_ollama_passes_when_model_present(monkeypatch):
    monkeypatch.setattr(settings, "planner_provider", "ollama")
    monkeypatch.setattr(settings, "planner_model", "testmodel:1b")
    with _patch_ollama(["testmodel:1b", "other:7b"]):
        await preflight_check(["planner"])


@pytest.mark.asyncio
async def test_ollama_raises_when_model_missing(monkeypatch):
    monkeypatch.setattr(settings, "planner_provider", "ollama")
    monkeypatch.setattr(settings, "planner_model", "ghostmodel:99b")
    with _patch_ollama(["something-else:7b"]):
        with pytest.raises(PreflightError) as ei:
            await preflight_check(["planner"])
    msg = str(ei.value)
    assert "ghostmodel:99b" in msg
    assert "ollama pull" in msg
