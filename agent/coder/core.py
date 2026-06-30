from agent.llm.providers.base import BaseLLMClient
from agent.models.schemas import Task, Plan, Patch, RetrievedContext
from agent.config import settings, logger
from agent.exceptions.errors import CoderError, LLMError

class Coder:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.model = getattr(llm_client, "model", None) or settings.coder_model
        self.last_usage = None
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
      "type": "create_file" | "update_file" | "delete_file" | "search_replace",
      "path": "path/relative/to/workspace",
      "content": "Full file content (required for create_file and update_file)",
      "search": "Exact text block to find (ONLY for search_replace; must match the file EXACTLY ONCE)",
      "replace": "Replacement text (ONLY for search_replace)"
    }}
  ],
  "commands": [
    "command to run in terminal"
  ]
}}

Choosing the operation type:
- Use "create_file" for brand-new files, and "update_file" (full content) for new files or large rewrites.
- Prefer "search_replace" for small, targeted edits to LARGE existing files: instead of re-emitting the
  whole file, provide a `search` block copied VERBATIM from the current file (include enough surrounding
  lines to be UNIQUE) and the `replace` block. The `search` text must match the file content exactly once;
  if it matches zero or multiple times the edit is rejected and you will be asked to correct it.
- When unsure, fall back to "update_file" with the full content — it always works.

Return ONLY valid JSON matching this schema. Do not include markdown formatting or explanations outside the JSON.
"""
        logger.info("Coder generating patch operations and commands...")
        try:
            result = await self.llm_client.generate_structured(self.model, prompt, Patch)
            self.last_usage = result.usage
            patch = result.data
            logger.info(f"Coder successfully generated {len(patch.operations)} operations and {len(patch.commands)} commands.")
            return patch
        except LLMError as e:
            logger.error(f"Coder failed to generate patch: {e}")
            raise CoderError(f"Failed to generate patch: {e}")
