import json
from pathlib import Path
from agent.llm.client import OllamaClient
from agent.models.schemas import Task, Plan, Patch, RetrievedContext
from agent.config import settings, logger
from agent.exceptions.errors import CoderError, LLMError

class Coder:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        self.model = settings.coder_model
        self.workspace = settings.get_workspace_path()

    async def generate_patch(self, task: Task, plan: Plan, context: RetrievedContext) -> Patch:
        context_str = "Retrieved Files:\n"
        for res in context.results:
            filepath = res.file
            context_str += f"\n--- {filepath} ---\n"
            try:
                content = (self.workspace / filepath).read_text(encoding='utf-8', errors='replace')
                context_str += content + "\n"
            except Exception:
                context_str += "<Error reading file>\n"
                
        context_str += "\n\nRetrieved Symbols (for reference):\n"
        for res in context.results:
            for sym in res.matched_symbols:
                context_str += f"- {sym.name} ({sym.type}) in {sym.file}: {sym.signature}\n"
                
        prompt = f"""You are an expert software developer.
You need to generate code changes and propose commands to fulfill the following task and plan.

<task>
{task.description}
</task>

<plan>
{plan.model_dump_json(indent=2)}
</plan>

<repository_context>
{context_str}
</repository_context>

Generate a structured JSON patch representing the file modifications and any commands to run.
The schema MUST strictly be:
{{
  "operations": [
    {{
      "type": "create_file" | "update_file" | "delete_file",
      "path": "path/relative/to/workspace",
      "content": "Full content of the file (required for create and update)"
    }}
  ],
  "commands": [
    "command to run in terminal"
  ]
}}

Return ONLY valid JSON matching this schema. Do not include markdown formatting or explanations outside the JSON.
"""
        logger.info("Coder generating patch operations and commands...")
        try:
            patch = await self.llm_client.generate_structured(self.model, prompt, Patch)
            logger.info(f"Coder successfully generated {len(patch.operations)} operations and {len(patch.commands)} commands.")
            return patch
        except LLMError as e:
            logger.error(f"Coder failed to generate patch: {e}")
            raise CoderError(f"Failed to generate patch: {e}")
