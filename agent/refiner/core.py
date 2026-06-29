"""Phase 3 — optional Prompt Refiner stage.

``PromptRefiner`` reads a raw user task and rewrites it into a clearer, more
complete instruction for the planner. It is deliberately conservative: it
preserves the user's intent and constraints and NEVER invents or expands scope.
"""

from agent.llm.providers.base import BaseLLMClient
from agent.config import settings, logger
from agent.refiner.schemas import RefinedPrompt

SYSTEM_INSTRUCTIONS = """You are a prompt refiner for a coding agent. You take a \
user's raw task and rewrite it into a clearer, better-structured instruction for \
a downstream planner.

STRICT RULES:
- PRESERVE the user's intent and ALL stated constraints exactly.
- NEVER invent, add, or expand scope. Do NOT introduce new features, files, \
technologies, or requirements the user did not ask for.
- Only disambiguate wording, structure the request, and surface acceptance \
criteria that are clearly IMPLIED by the original request.
- If something is genuinely ambiguous, record it in open_questions rather than \
guessing or inventing a requirement.
- assumptions and acceptance_criteria must follow logically from the original \
request — when in doubt, leave them out."""


class PromptRefiner:
    def __init__(self, client: BaseLLMClient):
        self.client = client
        self.model = getattr(client, "model", None) or settings.planner_model
        self.last_usage = None

    async def refine(self, raw_task: str) -> RefinedPrompt:
        """Rewrite ``raw_task`` into a structured :class:`RefinedPrompt`.

        Records token usage on ``self.last_usage``. Raises on LLM failure so the
        orchestrator can decide to fall back to the raw prompt.
        """
        prompt = f"""{SYSTEM_INSTRUCTIONS}

Raw user task:
<task>
{raw_task}
</task>

Return ONLY a JSON object with these fields:
{{
  "refined_task": "the clarified, structured rewrite (same scope as the original)",
  "clarified_goal": "one-line statement of the underlying goal",
  "assumptions": ["implicit assumptions made explicit"],
  "acceptance_criteria": ["concrete, checkable conditions for 'done'"],
  "open_questions": ["genuine ambiguities, if any"]
}}
"""
        logger.info("Refining task prompt before planning...")
        result = await self.client.generate_structured(self.model, prompt, RefinedPrompt)
        self.last_usage = result.usage
        logger.info("Prompt refinement complete.")
        return result.data
