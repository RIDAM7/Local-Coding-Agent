import time
from agent.execution.core import Executor
from agent.models.schemas import ValidationDiagnostic
from agent.config import settings, logger

class TestValidator:
    def __init__(self, executor: Executor):
        self.executor = executor
        self.command = settings.test_command

    async def validate(self) -> ValidationDiagnostic:
        if not self.command:
            return ValidationDiagnostic(
                stage="TEST",
                command="",
                success=True,
                stdout="Skipped: No test command configured.",
                stderr="",
                exit_code=0,
                duration=0.0
            )
            
        logger.info(f"Running test validation: {self.command}")
        start_time = time.time()
        result = await self.executor.run_command(self.command)
        duration = time.time() - start_time
        
        success = result.exit_code == 0
        if not success:
            logger.warning(f"Test validation failed with exit code {result.exit_code}")
        
        return ValidationDiagnostic(
            stage="TEST",
            command=self.command,
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration=duration
        )
