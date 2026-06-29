from typing import List
from agent.models.schemas import Constraint, ConstraintExtractionResult
from agent.llm.providers.base import BaseLLMClient
from agent.config import settings, logger

class ConstraintExtractor:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.model = getattr(llm_client, "model", None) or settings.planner_model
        self.last_usage = None

    async def extract(self, task_description: str) -> ConstraintExtractionResult:
        constraint_keywords = [
            "do not modify", "only modify", "do not delete", 
            "leave untouched", "preserve", "never edit", "do not touch"
        ]
        has_constraint_language = any(kw in task_description.lower() for kw in constraint_keywords)

        prompt = f"""Extract file and operation constraints from the following user task.
Supported constraint types:
1. PROTECTED_PATH (Patterns of files that must NOT be modified. e.g., "tests/**", "*_test.py")
2. ALLOWLIST_PATH (Patterns of files that are the ONLY ones allowed to be modified)
3. NO_DELETE (No files can be deleted)

Task:
{task_description}

Return a list of constraints strictly matching the schema.
If there are no constraints, return an empty list with success=True.
"""
        try:
            result = await self.llm_client.generate_structured(
                prompt=prompt,
                model=self.model,
                schema=ConstraintExtractionResult
            )
            self.last_usage = result.usage
            return result.data
        except Exception as e:
            logger.error(f"Failed to extract constraints: {e}")
            if has_constraint_language:
                logger.error("Explicit constraint language detected, but extraction failed. Failing closed.")
                return ConstraintExtractionResult(success=False, constraints=[])
            else:
                logger.info("No explicit constraint language detected. Safely defaulting to empty constraints.")
                return ConstraintExtractionResult(success=True, constraints=[])
