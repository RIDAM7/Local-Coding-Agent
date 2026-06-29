"""Shared helpers for cloud LLM providers (Phase 2).

These reuse the same JSON-extraction + 3-attempt retry/repair strategy as the
Ollama provider so structured output stays reliable regardless of backend.
"""

import json
from typing import Awaitable, Callable, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from agent.exceptions.errors import LLMError
from agent.llm.providers.base import LLMResult, Usage

T = TypeVar("T", bound=BaseModel)

# request_fn(prompt, json_mode, max_tokens) -> (response_text, input_tokens, output_tokens)
RequestFn = Callable[[str, bool, int], Awaitable[Tuple[str, int, int]]]


def extract_json(text: str) -> str:
    """Extract the first plausible JSON object or array from a string.

    Mirrors ``OllamaClient._extract_json`` so every provider tolerates models
    that wrap JSON in prose or markdown fences.
    """
    start_idx_obj = text.find('{')
    end_idx_obj = text.rfind('}')

    start_idx_arr = text.find('[')
    end_idx_arr = text.rfind(']')

    is_obj = start_idx_obj != -1 and (start_idx_arr == -1 or start_idx_obj < start_idx_arr)

    if is_obj and end_idx_obj > start_idx_obj:
        return text[start_idx_obj:end_idx_obj + 1]
    elif start_idx_arr != -1 and end_idx_arr > start_idx_arr:
        return text[start_idx_arr:end_idx_arr + 1]

    return text


async def generate_structured_with_retry(
    *,
    provider: str,
    model: str,
    prompt: str,
    schema: Type[T],
    max_retries: int,
    max_tokens: int,
    request_fn: RequestFn,
    logger,
) -> LLMResult:
    """Run the shared retry/repair loop and return an :class:`LLMResult`.

    Attempt 1 requests native JSON mode (where the provider supports it); attempt
    2 drops JSON mode but keeps the explicit JSON instruction; attempt 3+ sends a
    repair prompt carrying the schema, the last error, and the last bad output.
    Token usage is accumulated across attempts for accurate cost accounting.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    base_prompt = (
        f"{prompt}\n\nCRITICAL INSTRUCTION: Return ONLY valid JSON matching this schema. "
        f"Do not include markdown formatting or explanations.\nSchema:\n{schema_json}"
    )

    last_error = None
    last_response_text = None
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(1, max_retries + 1):
        logger.info(f"{provider} generation attempt {attempt}/{max_retries} for model {model}")

        if attempt <= 2:
            current_prompt = base_prompt
            json_mode = attempt == 1
        else:
            current_prompt = (
                "The following JSON response was generated but failed validation or was malformed.\n"
                f"Previous Error:\n{last_error}\n\n"
                f"Previous Invalid Output:\n{last_response_text}\n\n"
                "Please repair the JSON so it strictly matches the Target Schema.\n"
                f"{base_prompt}"
            )
            json_mode = False

        try:
            response_text, input_tokens, output_tokens = await request_fn(current_prompt, json_mode, max_tokens)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            last_response_text = response_text

            extracted_text = extract_json(response_text)

            try:
                parsed_json = json.loads(extracted_text)
                data = schema.model_validate(parsed_json)
                usage = Usage(
                    provider=provider,
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
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

    logger.error(f"All {max_retries} attempts failed to generate valid structured output from {provider}.")
    raise LLMError(f"Failed to generate structured output after {max_retries} retries. Last error: {last_error}")
