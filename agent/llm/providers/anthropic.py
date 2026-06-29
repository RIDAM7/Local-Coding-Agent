import aiohttp
from typing import Type, TypeVar

from pydantic import BaseModel

from agent.config import settings, logger
from agent.exceptions.errors import LLMError
from agent.llm.providers.base import BaseLLMClient, LLMResult
from agent.llm.providers.util import generate_structured_with_retry

T = TypeVar('T', bound=BaseModel)


class AnthropicClient(BaseLLMClient):
    """Anthropic Messages API client conforming to BaseLLMClient.

    Generalizes the original ``ClaudeClient`` logic. Anthropic has no native JSON
    mode, so reliability comes from the shared schema-aware retry/repair loop.
    """

    provider = "anthropic"

    def __init__(self):
        self.api_key = settings.anthropic_api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.anthropic_version = "2023-06-01"
        self.max_retries = settings.max_retries
        self.model = None

    async def _request(self, model: str, prompt: str, json_mode: bool, max_tokens: int) -> tuple[str, int, int]:
        # Anthropic has no response_format toggle; json_mode is accepted for
        # interface parity and the JSON instruction lives in the prompt instead.
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"Anthropic API error: {response.status} - {text}")
                        raise LLMError(f"Anthropic API returned status {response.status}: {text}")

                    data = await response.json()
                    response_text = ""
                    for content in data.get("content", []) or []:
                        if content.get("type") == "text":
                            response_text += content.get("text", "")

                    usage = data.get("usage") or {}
                    input_tokens = usage.get("input_tokens", 0) or 0
                    output_tokens = usage.get("output_tokens", 0) or 0

                    if not response_text:
                        raise LLMError("Empty response from Anthropic")

                    return response_text, input_tokens, output_tokens
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to Anthropic: {e}")
            raise LLMError(f"Connection error to Anthropic: {e}")

    async def generate_structured(self, model: str, prompt: str, schema: Type[T], *, max_tokens: int = 4096) -> LLMResult:
        if not self.api_key:
            raise LLMError("Anthropic API key is not configured.")

        async def request_fn(text: str, json_mode: bool, mt: int) -> tuple[str, int, int]:
            return await self._request(model, text, json_mode, mt)

        return await generate_structured_with_retry(
            provider=self.provider,
            model=model,
            prompt=prompt,
            schema=schema,
            max_retries=self.max_retries,
            max_tokens=max_tokens,
            request_fn=request_fn,
            logger=logger,
        )
