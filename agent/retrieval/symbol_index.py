import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict
from agent.retrieval.tree_sitter_indexer import TreeSitterIndexer
from agent.models.schemas import Symbol, IndexMetadata
from agent.config import logger
from datetime import datetime, timezone

class SymbolIndex:
    def __init__(self, indexer: TreeSitterIndexer):
        self.indexer = indexer

    def _get_file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def build_index(self, workspace_path: str, index_dir: str) -> List[Symbol]:
        logger.info("Building full symbol index...")
        symbols = []
        file_hashes = {}
        
        ignore_dirs = {'.git', 'venv', '.venv', 'node_modules', '__pycache__', 'index', 'reports'}
        supported_exts = self.indexer.get_supported_extensions()
        
        for root, dirs, filenames in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in filenames:
                if file.startswith('.'):
                    continue
                    
                full_path = Path(root) / file
                if full_path.suffix not in supported_exts:
                    continue
                    
                rel_path = str(full_path.relative_to(workspace_path)).replace('\\', '/')
                
                try:
                    content = full_path.read_text(encoding='utf-8', errors='replace')
                    file_symbols = self.indexer.parse_file(rel_path, content)
                    symbols.extend(file_symbols)
                    file_hashes[rel_path] = self._get_file_hash(full_path)
                except Exception as e:
                    logger.warning(f"Failed to index {rel_path}: {e}")
                    
        self.save(symbols, index_dir)
        self.save_metadata(file_hashes, index_dir)
        logger.info(f"Indexed {len(symbols)} symbols.")
        return symbols

    def incremental_update(self, workspace_path: str, index_dir: str) -> List[Symbol]:
        metadata = self.load_metadata(index_dir)
        if not metadata:
            return self.build_index(workspace_path, index_dir)
            
        logger.info("Running incremental symbol update...")
        old_symbols = self.load(index_dir) or []
        old_hashes = metadata.file_hashes
        new_hashes = {}
        changed_files = []
        
        ignore_dirs = {'.git', 'venv', '.venv', 'node_modules', '__pycache__', 'index', 'reports'}
        supported_exts = self.indexer.get_supported_extensions()
        
        current_files = []
        for root, dirs, filenames in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in filenames:
                if file.startswith('.'):
                    continue
                full_path = Path(root) / file
                if full_path.suffix not in supported_exts:
                    continue
                rel_path = str(full_path.relative_to(workspace_path)).replace('\\', '/')
                current_files.append((rel_path, full_path))
                
        # Find changes
        for rel_path, full_path in current_files:
            try:
                curr_hash = self._get_file_hash(full_path)
                new_hashes[rel_path] = curr_hash
                if old_hashes.get(rel_path) != curr_hash:
                    changed_files.append((rel_path, full_path))
            except Exception:
                pass
                
        # Remove deleted files from old symbols
        current_rel_paths = set(p[0] for p in current_files)
        updated_symbols = [s for s in old_symbols if s.file in current_rel_paths and s.file not in [cf[0] for cf in changed_files]]
        
        # Parse changed files
        for rel_path, full_path in changed_files:
            try:
                content = full_path.read_text(encoding='utf-8', errors='replace')
                file_symbols = self.indexer.parse_file(rel_path, content)
                updated_symbols.extend(file_symbols)
            except Exception as e:
                logger.warning(f"Failed to re-index {rel_path}: {e}")
                
        if changed_files or len(current_rel_paths) != len(old_hashes):
            logger.info(f"Incremental update: {len(changed_files)} files changed, {len(old_hashes) - len(current_rel_paths)} deleted.")
            self.save(updated_symbols, index_dir)
            self.save_metadata(new_hashes, index_dir)
            return updated_symbols
            
        logger.info("Index is up to date.")
        return old_symbols

    def load(self, index_dir: str) -> List[Symbol] | None:
        target = Path(index_dir) / "symbols.json"
        if target.exists():
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    data = json.loads(f.read())
                    return [Symbol.model_validate(s) for s in data]
            except Exception as e:
                logger.error(f"Failed to load symbols: {e}")
        return None

    def save(self, symbols: List[Symbol], index_dir: str) -> None:
        target = Path(index_dir) / "symbols.json"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = [s.model_dump() for s in symbols]
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save symbols: {e}")

    def load_metadata(self, index_dir: str) -> IndexMetadata | None:
        target = Path(index_dir) / "metadata.json"
        if target.exists():
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    return IndexMetadata.model_validate_json(f.read())
            except Exception:
                pass
        return None

    def save_metadata(self, file_hashes: Dict[str, str], index_dir: str) -> None:
        target = Path(index_dir) / "metadata.json"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            meta = IndexMetadata(
                last_indexed=datetime.now(timezone.utc).isoformat(),
                total_files_indexed=len(file_hashes),
                file_hashes=file_hashes
            )
            with open(target, 'w', encoding='utf-8') as f:
                f.write(meta.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
