from agent.llm.client import OllamaClient
from agent.models.schemas import Task, Plan
from agent.config import settings, logger
from agent.exceptions.errors import PlannerError, LLMError

class Planner:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        self.model = settings.planner_model
        
    async def create_plan(self, task: Task) -> Plan:
        prompt = f"""You are an expert software architect.
Your task is to create a structured execution plan for the following user request:

<task>
{task.description}
</task>

Return a JSON object matching this exact schema:
{{
  "goal": "Overall goal description",
  "summary": "Brief summary",
  "steps": [
    {{
      "id": 1,
      "description": "Step description",
      "expected_output": "What the step should produce"
    }}
  ]
}}

You must ONLY return valid JSON. Do not generate code.
"""
        logger.info(f"Generating plan for task...")
        try:
            plan = await self.llm_client.generate_structured(self.model, prompt, Plan)
            logger.info(f"Plan generated successfully with {len(plan.steps)} steps.")
            return plan
        except LLMError as e:
            logger.error(f"Planner failed to generate plan: {e}")
            raise PlannerError(f"Failed to generate plan: {e}")
