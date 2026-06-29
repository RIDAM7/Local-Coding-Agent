from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    """Token/cost accounting for a single structured generation call.

    Local providers (e.g. Ollama) report token counts when the backend exposes
    them, otherwise ``0`` with ``est_cost == 0.0``.
    """

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost: float = 0.0


@dataclass
class LLMResult:
    """Result of a structured generation: the validated model plus usage."""

    data: BaseModel
    usage: Usage


class BaseLLMClient(ABC):
    """Provider-agnostic interface for structured LLM generation.

    Every concrete provider returns an :class:`LLMResult` so callers can read
    the validated payload via ``result.data`` and record ``result.usage`` for
    cost/telemetry purposes regardless of which backend served the request.
    """

    @abstractmethod
    async def generate_structured(
        self,
        model: str,
        prompt: str,
        schema: Type[T],
        *,
        max_tokens: int = 4096,
    ) -> LLMResult:
        ...
