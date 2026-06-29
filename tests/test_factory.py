"""Tests for the Phase 1 role-based LLM client factory."""

import pytest

from agent.config import settings
from agent.llm.factory import build_client, resolve_model, resolve_provider, ROLE_INHERITANCE
from agent.llm.providers.base import BaseLLMClient
from agent.llm.providers.ollama import OllamaClient
from agent.llm.providers.openai import OpenAIClient
from agent.llm.providers.anthropic import AnthropicClient
from agent.llm.providers.google import GoogleClient

ALL_ROLES = ["planner", "coder", "refiner", "repair", "constraint", "reflection", "reviewer"]


def test_all_roles_resolve_to_ollama_client_by_default():
    """Every role must build a local Ollama client (no cloud) in Phase 1."""
    for role in ALL_ROLES:
        client = build_client(role)
        assert isinstance(client, OllamaClient)
        assert isinstance(client, BaseLLMClient)
        assert resolve_provider(role) == "ollama"


def test_base_roles_use_their_own_model():
    assert build_client("planner").model == settings.planner_model
    assert build_client("coder").model == settings.coder_model


def test_inheritance_defaults_resolve_to_correct_model():
    """repair/reflection inherit coder; constraint inherits planner."""
    assert resolve_model("repair") == settings.coder_model
    assert resolve_model("reflection") == settings.coder_model
    assert resolve_model("constraint") == settings.planner_model
    # refiner/reviewer inherit planner.
    assert resolve_model("refiner") == settings.planner_model
    assert resolve_model("reviewer") == settings.planner_model

    # And the built clients carry that resolved model.
    assert build_client("repair").model == settings.coder_model
    assert build_client("reflection").model == settings.coder_model
    assert build_client("constraint").model == settings.planner_model


def test_explicit_override_routes_correctly(monkeypatch):
    """An explicit <role>_model override wins over inheritance."""
    monkeypatch.setattr(settings, "reflection_model", "custom-reflect:latest")
    monkeypatch.setattr(settings, "constraint_model", "custom-constraint:latest")

    assert resolve_model("reflection") == "custom-reflect:latest"
    assert build_client("reflection").model == "custom-reflect:latest"
    assert resolve_model("constraint") == "custom-constraint:latest"
    assert build_client("constraint").model == "custom-constraint:latest"

    # Unrelated roles keep inheriting from their base config.
    assert resolve_model("repair") == settings.coder_model


def test_blank_override_falls_back_to_inheritance(monkeypatch):
    monkeypatch.setattr(settings, "reflection_model", "   ")
    assert resolve_model("reflection") == settings.coder_model


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        build_client("bogus")
    with pytest.raises(ValueError):
        resolve_model("bogus")


def test_role_inheritance_table_is_complete():
    for role in ALL_ROLES:
        assert role in ROLE_INHERITANCE


# --- Phase 2: provider routing ----------------------------------------------

def test_default_providers_are_ollama():
    for role in ALL_ROLES:
        assert resolve_provider(role) == "ollama"


def test_explicit_provider_routes_to_correct_class(monkeypatch):
    """CODER_PROVIDER=openai -> OpenAI client (custom base_url/model); planner stays local."""
    monkeypatch.setattr(settings, "coder_provider", "openai")
    monkeypatch.setattr(settings, "coder_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "openai_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    coder = build_client("coder")
    assert isinstance(coder, OpenAIClient)
    assert coder.model == "gpt-4o-mini"
    assert coder.base_url == "https://openrouter.ai/api/v1"

    assert isinstance(build_client("planner"), OllamaClient)


def test_each_provider_maps_to_its_class(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "x")
    monkeypatch.setattr(settings, "google_api_key", "y")

    monkeypatch.setattr(settings, "planner_provider", "anthropic")
    assert isinstance(build_client("planner"), AnthropicClient)
    monkeypatch.setattr(settings, "planner_provider", "google")
    assert isinstance(build_client("planner"), GoogleClient)


def test_provider_inheritance(monkeypatch):
    """repair/reflection inherit coder's provider; constraint inherits planner's."""
    monkeypatch.setattr(settings, "coder_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    assert resolve_provider("repair") == "openai"
    assert resolve_provider("reflection") == "openai"
    assert isinstance(build_client("repair"), OpenAIClient)
    assert isinstance(build_client("reflection"), OpenAIClient)

    # planner untouched -> constraint (inherits planner) stays ollama.
    assert resolve_provider("constraint") == "ollama"
    assert isinstance(build_client("constraint"), OllamaClient)


def test_explicit_provider_override_beats_inheritance(monkeypatch):
    monkeypatch.setattr(settings, "coder_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "reflection_provider", "ollama")
    # reflection explicitly pinned local even though coder is cloud.
    assert resolve_provider("reflection") == "ollama"
    assert isinstance(build_client("reflection"), OllamaClient)


def test_blank_base_provider_falls_back_to_ollama(monkeypatch):
    monkeypatch.setattr(settings, "planner_provider", "")
    assert resolve_provider("planner") == "ollama"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(settings, "planner_provider", "lemurai")
    with pytest.raises(ValueError):
        build_client("planner")
