import asyncio
import json
from typing import List, Dict, Any
from agent.config import logger

class RipgrepSearch:
    def __init__(self):
        pass

    async def _run_rg(self, args: List[str], workspace_path: str) -> List[Dict[str, Any]]:
        cmd = ["rg", "--json"] + args
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            results = []
            if stdout:
                lines = stdout.decode('utf-8', errors='replace').strip().split('\n')
                for line in lines:
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("type") == "match":
                            results.append(data["data"])
                    except json.JSONDecodeError:
                        continue
            
            return results
        except Exception as e:
            logger.error(f"Ripgrep execution failed: {e}. Is 'rg' installed?")
            return []

    async def exact_search(self, query: str, workspace_path: str) -> List[Dict[str, Any]]:
        return await self._run_rg(["-F", query], workspace_path)

    async def regex_search(self, pattern: str, workspace_path: str) -> List[Dict[str, Any]]:
        return await self._run_rg(["-e", pattern], workspace_path)

    async def symbol_search(self, symbol_name: str, workspace_path: str) -> List[Dict[str, Any]]:
        return await self._run_rg(["-w", symbol_name], workspace_path)
