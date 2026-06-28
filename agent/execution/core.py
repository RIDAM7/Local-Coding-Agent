from pathlib import Path
import asyncio
import time
from agent.config import settings, logger
from agent.models.schemas import CommandExecution
from agent.exceptions.errors import ExecutionError

class Executor:
    def __init__(self, workspace_path: Path = None):
        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()
        self.timeout = settings.command_timeout

    async def run_command(self, command: str) -> CommandExecution:
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
