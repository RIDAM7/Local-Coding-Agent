from pathlib import Path
from agent.llm.providers.base import BaseLLMClient
from agent.models.schemas import RepairContext, RepairPatch
from agent.config import settings, logger

class RepairCoder:
    def __init__(self, llm_client: BaseLLMClient, workspace_path: Path = None):
        self.llm_client = llm_client
        self.model = getattr(llm_client, "model", None) or settings.coder_model
        self.last_usage = None
        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()

    async def generate_repair(self, context: RepairContext) -> RepairPatch:
        logger.info("Generating repair patch...")
        
        context_str = "Retrieved Files (Targeting Error Areas):\n"
        for res in context.retrieved_context.results:
            filepath = res.file
            context_str += f"\n--- {filepath} ---\n"
            try:
                content = (self.workspace / filepath).read_text(encoding='utf-8', errors='replace')
                context_str += content + "\n"
            except Exception:
                context_str += "<Error reading file>\n"
                
        constraints_str = ""
        if context.constraints:
            constraints_str = "Task Constraints:\n"
            for c in context.constraints:
                constraints_str += f"- {c.type}: {', '.join(c.patterns) if c.patterns else ''}\n"

        scope_str = ""
        if context.repair_scope:
            scope_str = "Repair Scope (Allowed modifications):\n"
            for p in context.repair_scope.allowed_paths:
                scope_str += f"- {p}\n"
                
        prompt = f"""You are an expert software developer repairing broken code.
Your previous implementation for the original task failed validation.

Original Task:
{context.original_task}

{constraints_str}
{scope_str}

Failure Classification: {context.normalized_diagnostic.classification}
Primary Error: {context.normalized_diagnostic.primary_error_message}

Detailed Diagnostics:
Build Errors:
{context.diagnostics.build_errors}

Lint Errors:
{context.diagnostics.lint_errors}

Test Errors:
{context.diagnostics.test_errors}

{context_str}

Analyze the error logs and the retrieved context. 
Generate a repair patch that fixes the errors. 
The JSON must contain:
- 'explanation': clear string explaining why the repair works
- 'confidence': float between 0.0 and 1.0
- 'operations': list of file operations. Each operation MUST have 'type' (create_file, update_file, delete_file), 'path', and 'content'
- 'commands': list of terminal commands to run.

You MUST obey the Task Constraints and Repair Scope above.
If a repair requires violating these constraints, you must return an empty patch.

Repair must prioritize:
1. Constraint preservation
2. Validation success

Output strictly in JSON matching the RepairPatch schema. Do NOT return an empty patch unless you absolutely must to avoid violating a constraint.
"""
        
        try:
            result = await self.llm_client.generate_structured(
                prompt=prompt,
                model=self.model,
                schema=RepairPatch
            )
            self.last_usage = result.usage
            return result.data
        except Exception as e:
            logger.error(f"Failed to generate repair patch: {e}")
            raise
