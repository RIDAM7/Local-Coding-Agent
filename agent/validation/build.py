import time
from agent.execution.core import Executor
from agent.models.schemas import ValidationDiagnostic
from agent.config import settings, logger

class BuildValidator:
    def __init__(self, executor: Executor):
        self.executor = executor
        self.command = settings.build_command

    async def validate(self) -> ValidationDiagnostic:
        if not self.command:
            return ValidationDiagnostic(
                stage="BUILD",
                command="",
                success=True,
                stdout="Skipped: No build command configured.",
                stderr="",
                exit_code=0,
                duration=0.0
            )
            
        logger.info(f"Running build validation: {self.command}")
        start_time = time.time()
        result = await self.executor.run_command(self.command)
        duration = time.time() - start_time
        
        success = result.exit_code == 0
        if not success:
            logger.warning(f"Build validation failed with exit code {result.exit_code}")
        
        return ValidationDiagnostic(
            stage="BUILD",
            command=self.command,
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration=duration
        )
