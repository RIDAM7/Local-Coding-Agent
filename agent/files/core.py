import os
from pathlib import Path
from typing import List, Dict
import aiofiles
from agent.config import settings, logger
from agent.exceptions.errors import FileOperationError
from agent.safety.jail import assert_within_workspace


def apply_search_replace_text(content: str, search: str, replace: str) -> str:
    """Apply a single search/replace to ``content`` (Phase 7A).

    The ``search`` block must occur EXACTLY ONCE — zero or multiple matches raise
    :class:`FileOperationError` so a wrong edit can never be applied silently. This
    is a defense-in-depth backstop; the validator enforces the same invariant first.
    """
    if not search:
        raise FileOperationError("search_replace requires a non-empty 'search' block.")
    count = content.count(search)
    if count == 0:
        raise FileOperationError("search_replace: the 'search' block was not found in the file.")
    if count > 1:
        raise FileOperationError(f"search_replace: the 'search' block is ambiguous ({count} matches).")
    return content.replace(search, replace or "", 1)


class FileManager:
    def __init__(self, workspace_path: Path = None):
        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()
        logger.info(f"FileManager initialized with workspace: {self.workspace}")

    def _resolve_and_check_path(self, relative_path: str) -> Path:
        """Resolve the path inside the workspace jail (Phase 5).

        Delegates to the shared jail helper so traversal/absolute escapes are
        rejected with a clear error rather than clamped.
        """
        return assert_within_workspace(self.workspace, relative_path)

    async def read_file(self, path: str) -> str:
        target = self._resolve_and_check_path(path)
        if not target.exists():
            raise FileOperationError(f"File not found: {path}")
        if not target.is_file():
            raise FileOperationError(f"Target is not a file: {path}")
            
        try:
            async with aiofiles.open(target, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            raise FileOperationError(f"Failed to read file {path}: {str(e)}")

    async def create_file(self, path: str, content: str) -> None:
        target = self._resolve_and_check_path(path)
        if target.exists():
            raise FileOperationError(f"File already exists: {path}. Use update_file instead.")
            
        target.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            async with aiofiles.open(target, 'w', encoding='utf-8') as f:
                await f.write(content)
            logger.info(f"Created file: {path}")
        except Exception as e:
            raise FileOperationError(f"Failed to create file {path}: {str(e)}")

    async def update_file(self, path: str, content: str) -> None:
        target = self._resolve_and_check_path(path)
        if not target.exists():
            raise FileOperationError(f"File does not exist: {path}. Use create_file instead.")
            
        try:
            async with aiofiles.open(target, 'w', encoding='utf-8') as f:
                await f.write(content)
            logger.info(f"Updated file: {path}")
        except Exception as e:
            raise FileOperationError(f"Failed to update file {path}: {str(e)}")

    async def delete_file(self, path: str) -> None:
        target = self._resolve_and_check_path(path)
        if not target.exists():
            raise FileOperationError(f"File does not exist: {path}")
        if not target.is_file():
            raise FileOperationError(f"Target is not a file: {path}")
            
        try:
            os.remove(target)
            logger.info(f"Deleted file: {path}")
        except Exception as e:
            raise FileOperationError(f"Failed to delete file {path}: {str(e)}")

    async def list_directory(self, path: str = ".") -> List[str]:
        target = self._resolve_and_check_path(path)
        if not target.exists():
            raise FileOperationError(f"Directory not found: {path}")
        if not target.is_dir():
            raise FileOperationError(f"Target is not a directory: {path}")
            
        try:
            items = []
            for item in target.iterdir():
                items.append(str(item.relative_to(self.workspace)).replace('\\', '/'))
            return items
        except Exception as e:
            raise FileOperationError(f"Failed to list directory {path}: {str(e)}")
            
    async def get_repository_context(self) -> Dict[str, str]:
        """Reads all non-hidden text files in the workspace."""
        context = {}
        ignore_dirs = {'.git', 'venv', '.venv', 'node_modules', '__pycache__'}
        
        try:
            for root, dirs, files in os.walk(self.workspace):
                dirs[:] = [d for d in dirs if d not in ignore_dirs]
                
                for file in files:
                    if file.startswith('.'):
                        continue
                        
                    full_path = Path(root) / file
                    rel_path = str(full_path.relative_to(self.workspace)).replace('\\', '/')
                    
                    try:
                        content = await self.read_file(rel_path)
                        context[rel_path] = content
                    except Exception:
                        pass
            return context
        except Exception as e:
            logger.error(f"Failed to build repository context: {e}")
            return {}
