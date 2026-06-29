import json
import aiohttp
from typing import TypeVar, Type

from pydantic import BaseModel, ValidationError

from agent.config import settings, logger
from agent.exceptions.errors import LLMError
from agent.llm.providers.base import BaseLLMClient, LLMResult, Usage

T = TypeVar('T', bound=BaseModel)

class OllamaClient(BaseLLMClient):
    provider = "ollama"

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.generate_url = f"{self.base_url}/api/generate"
        self.max_retries = settings.max_retries
        # Default model for the role this client was built for. The factory sets
        # this; callers may still pass an explicit ``model`` to generate_structured.
        self.model = None

    def _extract_json(self, text: str) -> str:
        """Attempts to extract the first valid JSON object or array from a string."""
        start_idx_obj = text.find('{')
        end_idx_obj = text.rfind('}')

        start_idx_arr = text.find('[')
        end_idx_arr = text.rfind(']')

        # Determine whether object or array appears first
        is_obj = start_idx_obj != -1 and (start_idx_arr == -1 or start_idx_obj < start_idx_arr)

        if is_obj and end_idx_obj > start_idx_obj:
            return text[start_idx_obj:end_idx_obj+1]
        elif start_idx_arr != -1 and end_idx_arr > start_idx_arr:
            return text[start_idx_arr:end_idx_arr+1]

        return text

    async def _make_request(self, payload: dict) -> tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens).

        Ollama reports ``prompt_eval_count`` / ``eval_count`` when available;
        otherwise these default to 0.
        """
        logger.debug(f"Sending payload to Ollama:\n{json.dumps(payload, indent=2)}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.generate_url, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {text}")
                        raise LLMError(f"Ollama API returned status {response.status}: {text}")

                    data = await response.json()
                    response_text = data.get("response", "")
                    input_tokens = data.get("prompt_eval_count", 0) or 0
                    output_tokens = data.get("eval_count", 0) or 0

                    logger.debug(f"Received raw response from Ollama:\n{response_text}")

                    if not response_text:
                        raise LLMError("Empty response from Ollama")

                    return response_text, input_tokens, output_tokens
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            raise LLMError(f"Connection error to Ollama: {e}")

    async def generate_structured(self, model: str, prompt: str, schema: Type[T], *, max_tokens: int = 4096) -> LLMResult:
        """
        Sends a prompt to Ollama, requesting JSON format, and validates
        the response against the provided Pydantic schema with retry logic.

        Returns an :class:`LLMResult` wrapping the validated model and token usage.
        ``max_tokens`` is accepted for interface parity with cloud providers; Ollama
        does not require it, so request payloads are unchanged.
        """

        last_error = None
        last_response_text = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Ollama generation attempt {attempt}/{self.max_retries} for model {model}")

            try:
                if attempt == 1:
                    # Attempt 1: Use JSON mode
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "options": {
                            "temperature": 0.1
                        }
                    }
                elif attempt == 2:
                    # Attempt 2: Remove JSON mode, ask explicitly for ONLY JSON
                    modified_prompt = prompt + "\n\nCRITICAL INSTRUCTION: Return ONLY valid JSON. Do not include markdown formatting or explanations."
                    payload = {
                        "model": model,
                        "prompt": modified_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1
                        }
                    }
                else:
                    # Attempt 3+: Repair prompt
                    repair_prompt = (
                        "The following JSON response was generated but failed validation or was malformed.\n"
                        f"Target Schema:\n{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
                        f"Previous Error:\n{last_error}\n\n"
                        f"Previous Invalid Output:\n{last_response_text}\n\n"
                        "Please repair the JSON so it strictly matches the Target Schema. Return ONLY valid JSON."
                    )
                    payload = {
                        "model": model,
                        "prompt": repair_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1
                        }
                    }

                response_text, input_tokens, output_tokens = await self._make_request(payload)
                last_response_text = response_text

                # Extract JSON string for attempt >= 2 or if attempt 1 returns something with markdown wrapper
                extracted_text = self._extract_json(response_text)
                if extracted_text != response_text:
                    logger.debug(f"Extracted JSON string from response:\n{extracted_text}")

                try:
                    parsed_json = json.loads(extracted_text)
                    data = schema.model_validate(parsed_json)
                    usage = Usage(
                        provider=self.provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        est_cost=0.0,
                    )
                    return LLMResult(data=data, usage=usage)
                except json.JSONDecodeError as e:
                    last_error = f"JSONDecodeError: {e}"
                    logger.warning(f"Attempt {attempt} failed JSON parsing: {e}")
                except ValidationError as e:
                    last_error = f"ValidationError: {e}"
                    logger.warning(f"Attempt {attempt} failed schema validation: {e}")

            except LLMError as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt} failed due to LLM error: {e}")

        # If all retries exhausted
        logger.error(f"All {self.max_retries} attempts failed to generate valid structured output.")
        logger.error(f"Last Error: {last_error}")
        logger.error(f"Last Invalid Output:\n{last_response_text}")
        raise LLMError(f"Failed to generate structured output after {self.max_retries} retries. Last error: {last_error}")
