import json
import aiohttp
from typing import TypeVar, Type

from pydantic import BaseModel, ValidationError

from agent.config import settings, logger
from agent.exceptions.errors import LLMError

T = TypeVar('T', bound=BaseModel)

class ClaudeClient:
    def __init__(self):
        self.api_key = settings.anthropic_api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.max_retries = settings.max_retries

    def _extract_json(self, text: str) -> str:
        """Attempts to extract the first valid JSON object or array from a string."""
        start_idx_obj = text.find('{')
        end_idx_obj = text.rfind('}')
        
        start_idx_arr = text.find('[')
        end_idx_arr = text.rfind(']')
        
        is_obj = start_idx_obj != -1 and (start_idx_arr == -1 or start_idx_obj < start_idx_arr)
        
        if is_obj and end_idx_obj > start_idx_obj:
            return text[start_idx_obj:end_idx_obj+1]
        elif start_idx_arr != -1 and end_idx_arr > start_idx_arr:
            return text[start_idx_arr:end_idx_arr+1]
            
        return text

    async def _make_request(self, payload: dict) -> tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens)"""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
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
                    for content in data.get("content", []):
                        if content.get("type") == "text":
                            response_text += content.get("text", "")
                    
                    usage = data.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    
                    if not response_text:
                        raise LLMError("Empty response from Anthropic")
                        
                    return response_text, input_tokens, output_tokens
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to Anthropic: {e}")
            raise LLMError(f"Connection error to Anthropic: {e}")

    async def generate_structured(self, model: str, prompt: str, schema: Type[T], max_tokens: int = 4096) -> tuple[T, int, int]:
        """
        Sends a prompt to Claude, requesting JSON format, and validates
        the response against the provided Pydantic schema with retry logic.
        Returns (ParsedObject, input_tokens, output_tokens).
        """
        if not self.api_key:
            raise LLMError("Anthropic API key is not configured.")
            
        last_error = None
        last_response_text = None
        
        total_input_tokens = 0
        total_output_tokens = 0
        
        base_prompt = prompt + f"\n\nCRITICAL INSTRUCTION: Return ONLY valid JSON matching this schema. Do not include markdown formatting or explanations.\nSchema:\n{json.dumps(schema.model_json_schema(), indent=2)}"
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Claude generation attempt {attempt}/{self.max_retries} for model {model}")
            
            try:
                current_prompt = base_prompt
                if attempt > 1:
                    current_prompt = (
                        "The following JSON response was generated but failed validation or was malformed.\n"
                        f"Previous Error:\n{last_error}\n\n"
                        f"Previous Invalid Output:\n{last_response_text}\n\n"
                        f"Please repair the JSON so it strictly matches the Target Schema.\n"
                        f"{base_prompt}"
                    )
                
                payload = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "user", "content": current_prompt}
                    ]
                }

                response_text, i_tok, o_tok = await self._make_request(payload)
                total_input_tokens += i_tok
                total_output_tokens += o_tok
                last_response_text = response_text
                
                extracted_text = self._extract_json(response_text)

                try:
                    parsed_json = json.loads(extracted_text)
                    return schema.model_validate(parsed_json), total_input_tokens, total_output_tokens
                except json.JSONDecodeError as e:
                    last_error = f"JSONDecodeError: {e}"
                    logger.warning(f"Attempt {attempt} failed JSON parsing: {e}")
                except ValidationError as e:
                    last_error = f"ValidationError: {e}"
                    logger.warning(f"Attempt {attempt} failed schema validation: {e}")

            except LLMError as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt} failed due to LLM error: {e}")

        logger.error(f"All {self.max_retries} attempts failed to generate valid structured output from Claude.")
        raise LLMError(f"Failed to generate structured output after {self.max_retries} retries. Last error: {last_error}")
