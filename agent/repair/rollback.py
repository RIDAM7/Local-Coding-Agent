import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from agent.config import settings, logger

class RollbackManager:
    def __init__(self, workspace_path: Path = None):
        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()
        self.file_backups: Dict[str, Tuple[str, str]] = {} # path -> (content, hash)
        self.new_files: List[str] = []
        self.checkpoint_active = False

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def checkpoint(self, files_to_modify: List[str]):
        """Capture the state of files before any modifications."""
        if self.checkpoint_active:
            return 
            
        logger.info("RollbackManager: Creating baseline checkpoint.")
        for filepath in files_to_modify:
            target_path = (self.workspace / filepath).resolve()
            if target_path.exists():
                try:
                    content = target_path.read_text(encoding='utf-8', errors='replace')
                    self.file_backups[filepath] = (content, self._hash_content(content))
                except Exception as e:
                    logger.warning(f"RollbackManager: Failed to read {filepath} for backup: {e}")
            else:
                self.new_files.append(filepath)
                
        self.checkpoint_active = True

    def track_new_file(self, filepath: str):
        if self.checkpoint_active and filepath not in self.file_backups and filepath not in self.new_files:
            self.new_files.append(filepath)

    def restore(self):
        """Restore workspace to the baseline checkpoint."""
        if not self.checkpoint_active:
            logger.warning("RollbackManager: No checkpoint to restore.")
            return

        logger.info("RollbackManager: Restoring workspace to baseline.")
        for filepath, (content, _) in self.file_backups.items():
            target_path = (self.workspace / filepath).resolve()
            try:
                target_path.write_text(content, encoding='utf-8')
            except Exception as e:
                logger.error(f"RollbackManager: Failed to restore {filepath}: {e}")

        for filepath in self.new_files:
            target_path = (self.workspace / filepath).resolve()
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception as e:
                    logger.error(f"RollbackManager: Failed to delete new file {filepath}: {e}")

    def verify(self) -> Dict[str, bool]:
        """Verify that the baseline hashes match current file hashes."""
        results = {}
        for filepath, (_, expected_hash) in self.file_backups.items():
            target_path = (self.workspace / filepath).resolve()
            if target_path.exists():
                content = target_path.read_text(encoding='utf-8', errors='replace')
                results[filepath] = (self._hash_content(content) == expected_hash)
            else:
                results[filepath] = False
                
        for filepath in self.new_files:
            target_path = (self.workspace / filepath).resolve()
            results[filepath] = not target_path.exists()
            
        return results
