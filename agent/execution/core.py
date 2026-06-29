from pathlib import Path
import asyncio
import time
from agent.config import settings, logger
from agent.models.schemas import CommandExecution
from agent.exceptions.errors import ExecutionError
from agent.safety.commands import find_denied

class Executor:
    def __init__(self, workspace_path: Path = None, dry_run: bool = False):
        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()
        self.timeout = settings.command_timeout
        self.dry_run = dry_run

    async def run_command(self, command: str) -> CommandExecution:
        # Phase 5: hard denylist backstop — non-bypassable, even with --yes. A
        # denylisted command never reaches a subprocess no matter who calls this.
        denied = find_denied(command)
        if denied:
            msg = f"Command blocked by safety denylist ({denied}): {command}"
            logger.error(msg)
            raise ExecutionError(msg)

        # Phase 5: --dry-run never runs anything.
        if self.dry_run:
            logger.info(f"[dry-run] command not executed: {command}")
            return CommandExecution(
                command=command,
                stdout="Skipped (dry-run): command not executed.",
                stderr="",
                exit_code=0,
                duration=0.0,
            )

        logger.info(f"Executing command: {command}")
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                raise ExecutionError(f"Command timed out after {self.timeout} seconds: {command}")
                
            duration = time.time() - start_time
            
            result = CommandExecution(
                command=command,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                exit_code=process.returncode,
                duration=round(duration, 3)
            )
            
            if result.exit_code != 0:
                logger.warning(f"Command failed with exit code {result.exit_code}: {result.stderr}")
            else:
                logger.info(f"Command succeeded in {result.duration}s")
                
            return result
            
        except Exception as e:
            if isinstance(e, ExecutionError):
                raise
            raise ExecutionError(f"Failed to execute command '{command}': {str(e)}")
