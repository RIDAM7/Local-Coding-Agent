"""Backward-compatibility shim.

The Ollama client now lives in :mod:`agent.llm.providers.ollama` as part of the
provider abstraction layer (Phase 1). This module re-exports ``OllamaClient`` so
existing imports (``from agent.llm.client import OllamaClient``) keep working.
"""

from agent.llm.providers.ollama import OllamaClient

__all__ = ["OllamaClient"]
