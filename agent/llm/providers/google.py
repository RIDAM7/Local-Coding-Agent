import aiohttp
from typing import Type, TypeVar

from pydantic import BaseModel

from agent.config import settings, logger
from agent.exceptions.errors import LLMError
from agent.llm.providers.base import BaseLLMClient, LLMResult
from agent.llm.providers.util import generate_structured_with_retry

T = TypeVar('T', bound=BaseModel)

# Host only — the API key is passed as a query param and must never be logged.
GOOGLE_API_HOST = "https://generativelanguage.googleapis.com"


class GoogleClient(BaseLLMClient):
    """Google Gemini client (generateContent) conforming to BaseLLMClient."""

    provider = "google"

    def __init__(self):
        self.api_key = settings.google_api_key
        self.max_retries = settings.max_retries
        self.model = None

    async def _request(self, model: str, prompt: str, json_mode: bool, max_tokens: int) -> tuple[str, int, int]:
        # Key is in the query string; do NOT log the full URL.
        url = f"{GOOGLE_API_HOST}/v1beta/models/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        generation_config = {"temperature": 0.1, "maxOutputTokens": max_tokens}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"Google API error: {response.status} - {text}")
                        raise LLMError(f"Google API returned status {response.status}: {text}")

                    data = await response.json()
                    response_text = ""
                    candidates = data.get("candidates") or []
                    if candidates:
                        parts = ((candidates[0].get("content") or {}).get("parts")) or []
                        for part in parts:
                            response_text += part.get("text", "") or ""

                    usage = data.get("usageMetadata") or {}
                    input_tokens = usage.get("promptTokenCount", 0) or 0
                    output_tokens = usage.get("candidatesTokenCount", 0) or 0

                    if not response_text:
                        raise LLMError("Empty response from Google")

                    return response_text, input_tokens, output_tokens
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to Google Gemini API: {e}")
            raise LLMError(f"Connection error to Google Gemini API: {e}")

    async def generate_structured(self, model: str, prompt: str, schema: Type[T], *, max_tokens: int = 4096) -> LLMResult:
        if not self.api_key:
            raise LLMError("Google API key is not configured.")

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
