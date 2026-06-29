import aiohttp
from typing import Type, TypeVar

from pydantic import BaseModel

from agent.config import settings, logger
from agent.exceptions.errors import LLMError
from agent.llm.providers.base import BaseLLMClient, LLMResult
from agent.llm.providers.util import generate_structured_with_retry

T = TypeVar('T', bound=BaseModel)


class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions client.

    The configurable ``base_url`` makes this a single client for every
    OpenAI-compatible gateway (OpenRouter, Groq, Together, DeepSeek, Fireworks,
    vLLM, LM Studio, ...): just change ``OPENAI_BASE_URL`` + key + model.
    """

    provider = "openai"

    def __init__(self):
        self.api_key = settings.openai_api_key
        self.base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.max_retries = settings.max_retries
        self.model = None

    async def _request(self, model: str, prompt: str, json_mode: bool, max_tokens: int) -> tuple[str, int, int]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.chat_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {text}")
                        raise LLMError(f"OpenAI API returned status {response.status}: {text}")

                    data = await response.json()
                    choices = data.get("choices") or []
                    response_text = ""
                    if choices:
                        response_text = (choices[0].get("message") or {}).get("content", "") or ""

                    usage = data.get("usage") or {}
                    input_tokens = usage.get("prompt_tokens", 0) or 0
                    output_tokens = usage.get("completion_tokens", 0) or 0

                    if not response_text:
                        raise LLMError("Empty response from OpenAI")

                    return response_text, input_tokens, output_tokens
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to OpenAI-compatible endpoint at {self.base_url}: {e}")
            raise LLMError(f"Connection error to OpenAI-compatible endpoint: {e}")

    async def generate_structured(self, model: str, prompt: str, schema: Type[T], *, max_tokens: int = 4096) -> LLMResult:
        if not self.api_key:
            raise LLMError("OpenAI API key is not configured.")

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
