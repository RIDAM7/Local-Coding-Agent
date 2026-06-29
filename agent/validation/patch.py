import os
from pathlib import Path
from typing import List, Dict
from agent.models.schemas import Patch, PatchValidationResult, FileOperation
from agent.config import settings, logger

class PatchValidator:
    def __init__(self, workspace_path: Path = None):
        self.workspace_path = workspace_path if workspace_path else settings.get_workspace_path()

    def validate_and_repair(self, patch: Patch) -> PatchValidationResult:
        errors = []
        warnings = []
        modified_operations: List[FileOperation] = []
        is_valid = True
        
        seen_paths: Dict[str, str] = {} 

        for op in patch.operations:
            target_path = op.path.replace('\\', '/')
            full_path = (self.workspace_path / target_path).resolve()
            
            # Check for path traversal / out of workspace
            try:
                full_path.relative_to(self.workspace_path.resolve())
            except ValueError:
                errors.append(f"Operation on '{op.path}' is outside the workspace.")
                is_valid = False
                continue

            # Duplicate operation detection
            if target_path in seen_paths:
                prev_op = seen_paths[target_path]
                errors.append(f"Duplicate operation on '{target_path}'. Previously '{prev_op}', now '{op.type}'.")
                is_valid = False
                continue
                
            seen_paths[target_path] = op.type
            
            exists = full_path.exists()
            is_dir = full_path.is_dir() if exists else False
            
            if is_dir:
                errors.append(f"Target '{op.path}' is a directory, not a file.")
                is_valid = False
                continue
                
            new_op = op.model_copy()

            if op.type == "create_file":
                if exists:
                    warnings.append(f"File '{op.path}' already exists. Converting create_file to update_file.")
                    new_op.type = "update_file"
                    modified_operations.append(new_op)
                else:
                    modified_operations.append(new_op)
            
            elif op.type == "update_file":
                if not exists:
                    warnings.append(f"File '{op.path}' does not exist. Converting update_file to create_file.")
                    new_op.type = "create_file"
                    modified_operations.append(new_op)
                else:
                    modified_operations.append(new_op)
                    
            elif op.type == "delete_file":
                if not exists:
                    warnings.append(f"File '{op.path}' already missing. Removing delete_file operation.")
                else:
                    modified_operations.append(new_op)

            elif op.type == "search_replace":
                # Phase 7A: the search block MUST match the current file content
                # EXACTLY ONCE. Zero or multiple matches => validation failure fed
                # into the repair loop, never a silent wrong edit.
                if not exists:
                    errors.append(f"search_replace on '{op.path}' but the file does not exist. Use create_file/update_file instead.")
                    is_valid = False
                elif not op.search:
                    errors.append(f"search_replace on '{op.path}' is missing a non-empty 'search' block.")
                    is_valid = False
                else:
                    try:
                        current = full_path.read_text(encoding='utf-8', errors='replace')
                    except Exception as e:
                        errors.append(f"search_replace on '{op.path}' could not read the file: {e}")
                        is_valid = False
                        continue
                    count = current.count(op.search)
                    if count == 0:
                        errors.append(f"search_replace on '{op.path}': the 'search' block was not found. It must match the file content exactly.")
                        is_valid = False
                    elif count > 1:
                        errors.append(f"search_replace on '{op.path}': the 'search' block is ambiguous ({count} matches). Make it unique (add surrounding context).")
                        is_valid = False
                    else:
                        modified_operations.append(new_op)

        modified_patch = Patch(operations=modified_operations, commands=patch.commands)

        if errors:
            logger.error("Patch validation failed due to structural errors.")
            is_valid = False

        return PatchValidationResult(
            is_valid=is_valid,
            modified_patch=modified_patch,
            errors=errors,
            warnings=warnings
        )
