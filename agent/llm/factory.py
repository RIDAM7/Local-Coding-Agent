"""Role-based LLM client factory (Phase 1 — Provider Abstraction Layer).

``build_client(role)`` resolves a role to a concrete :class:`BaseLLMClient`,
reading the provider + model from settings. In Phase 1 every role resolves to a
local Ollama client (cloud providers arrive in Phase 2); the resolved default
model for the role is attached as ``client.model``.

Model inheritance keeps env config small:
  - ``constraint`` defaults to the ``planner`` role's model.
  - ``repair`` and ``reflection`` default to the ``coder`` role's model.
  - ``refiner`` and ``reviewer`` default to the ``planner`` role's model.
Any role may set an explicit ``<role>_model`` in settings to override.
"""

from agent.config import settings, logger
from agent.llm.providers.base import BaseLLMClient
from agent.llm.providers.ollama import OllamaClient
from agent.llm.providers.openai import OpenAIClient
from agent.llm.providers.anthropic import AnthropicClient
from agent.llm.providers.google import GoogleClient

# Provider name -> client class. OpenAI-compatible gateways (OpenRouter, Groq,
# Together, DeepSeek, vLLM, ...) all use the "openai" provider with a custom base_url.
PROVIDERS = {
    "ollama": OllamaClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "google": GoogleClient,
}

# Supported roles and the role whose model they inherit when not explicitly set.
# ``planner`` and ``coder`` are the base roles (they own a configured model).
ROLE_INHERITANCE = {
    "planner": "planner",
    "coder": "coder",
    "constraint": "planner",
    "repair": "coder",
    "reflection": "coder",
    "refiner": "planner",
    "reviewer": "planner",
}


def resolve_model(role: str) -> str:
    """Resolve the model name for a role, honoring explicit overrides then
    inheritance defaults (``planner_model`` / ``coder_model``)."""
    role = role.lower()
    if role not in ROLE_INHERITANCE:
        raise ValueError(f"Unknown LLM role: {role!r}. Valid roles: {sorted(ROLE_INHERITANCE)}")

    # Explicit per-role override, e.g. settings.reflection_model.
    explicit = (getattr(settings, f"{role}_model", "") or "").strip()
    if explicit:
        return explicit

    base = ROLE_INHERITANCE[role]
    return getattr(settings, f"{base}_model")


def resolve_provider(role: str) -> str:
    """Resolve the provider for a role.

    Reads ``<role>_provider`` from settings; an explicit value wins. Derived roles
    with no explicit provider inherit their base role's provider (the same
    inheritance used for models). Base roles default to ``ollama`` so the default
    env stays fully local and zero-config.
    """
    role = role.lower()
    if role not in ROLE_INHERITANCE:
        raise ValueError(f"Unknown LLM role: {role!r}. Valid roles: {sorted(ROLE_INHERITANCE)}")

    explicit = (getattr(settings, f"{role}_provider", "") or "").strip().lower()
    if explicit:
        return explicit

    base = ROLE_INHERITANCE[role]
    if base == role:
        # Base role with no explicit provider -> local default.
        return "ollama"
    return resolve_provider(base)


def build_client(role: str) -> BaseLLMClient:
    """Build the LLM client for the given role with its resolved provider + model."""
    provider = resolve_provider(role)
    model = resolve_model(role)

    client_cls = PROVIDERS.get(provider)
    if client_cls is None:
        raise ValueError(
            f"Unknown provider {provider!r} for role {role!r}. "
            f"Valid providers: {sorted(PROVIDERS)}"
        )

    client = client_cls()
    client.model = model
    logger.debug(f"Built '{role}' client -> provider={provider}, model={model}")
    return client
