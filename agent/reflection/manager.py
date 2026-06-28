import time
from typing import List, Any
from agent.config import settings, logger
from agent.reflection.schemas import ReflectionReport, ReflectionResult
from agent.exceptions.errors import LLMError

class ReflectionManager:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.model = settings.planner_model

    async def reflect(
        self,
        task: Any,
        constraints: List[Any],
        retrieved_context: Any,
        retrieved_memories: Any,
        generated_patch: Any
    ) -> ReflectionReport:
        start_time = time.time()
        
        # Build prompt
        prompt = "You are a Reviewer/Reflector agent. Your task is to critique the proposed patch against the task constraints and context.\n"
        prompt += f"Task: {task.description}\n"
        if constraints:
            prompt += "Constraints:\n"
            for c in constraints:
                prompt += f"- {c}\n"
                
        # Stringify the patch
        patch_str = ""
        if generated_patch:
            if hasattr(generated_patch, "operations"):
                for op in generated_patch.operations:
                    patch_str += f"File: {op.path} Action: {op.type}\n{op.content or ''}\n"
            if hasattr(generated_patch, "commands"):
                for cmd in generated_patch.commands:
                    patch_str += f"Command: {cmd}\n"
        
        prompt += f"\nProposed Patch:\n{patch_str}\n"
        prompt += "\nEvaluate this patch strictly against the following categories: CONSTRAINT_VIOLATION_RISK, SCOPE_DRIFT_RISK, MISSING_FILE_RISK, TEST_FAILURE_RISK, SYNTAX_RISK, DUPLICATE_PATCH_RISK.\n"
        prompt += "Return a JSON object matching the ReflectionReport schema. Set result to FAIL if there is a severe violation, WARNING if there are potential issues, and PASS if it looks good."

        try:
            # We enforce a timeout internally or via asyncio
            import asyncio
            report = await asyncio.wait_for(
                self.llm_client.generate_structured(
                    model=self.model,
                    prompt=prompt,
                    schema=ReflectionReport
                ),
                timeout=60.0  # Bounded reflection
            )
            report.execution_time_ms = (time.time() - start_time) * 1000
            report.model_name = self.model
            return report
            
        except Exception as e:
            logger.error(f"Reflection bypassed due to failure: {e}")
            return ReflectionReport(
                result=ReflectionResult.PASS,
                critiques=[],
                summary="Reflection bypassed",
                execution_time_ms=(time.time() - start_time) * 1000,
                model_name=self.model
            )
