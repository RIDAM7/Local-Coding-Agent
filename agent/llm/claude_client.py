"""Backward-compatibility adapter for the Anthropic provider.

The Anthropic logic now lives in :mod:`agent.llm.providers.anthropic` as a
``BaseLLMClient``. ``ClaudeReviewer`` still expects the legacy
``(obj, input_tokens, output_tokens)`` tuple, so this thin ``ClaudeClient``
delegates to :class:`AnthropicClient` and adapts the :class:`LLMResult` back to
that tuple. Nothing else changes for existing callers.
"""

from typing import Type, TypeVar

from pydantic import BaseModel

from agent.llm.providers.anthropic import AnthropicClient

T = TypeVar('T', bound=BaseModel)


class ClaudeClient:
    def __init__(self):
        self._client = AnthropicClient()
        # Preserve attribute parity with the previous implementation.
        self.api_key = self._client.api_key
        self.max_retries = self._client.max_retries

    async def generate_structured(self, model: str, prompt: str, schema: Type[T], max_tokens: int = 4096) -> tuple[T, int, int]:
        """Returns (ParsedObject, input_tokens, output_tokens) for legacy callers."""
        result = await self._client.generate_structured(model, prompt, schema, max_tokens=max_tokens)
        return result.data, result.usage.input_tokens, result.usage.output_tokens
