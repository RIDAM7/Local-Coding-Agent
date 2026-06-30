import os
from pathlib import Path
from agent.models.schemas import RepositoryMapData, RepositoryComponent
from agent.config import logger

class RepositoryMap:
    def __init__(self):
        pass

    def generate(self, workspace_path: str) -> RepositoryMapData:
        logger.info("Generating repository map...")
        files = []
        file_types = {}
        components = []
        
        ignore_dirs = {'.git', 'venv', '.venv', 'node_modules', '__pycache__', 'index', 'reports'}
        
        for root, dirs, filenames in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in filenames:
                if file.startswith('.'):
                    continue
                    
                full_path = Path(root) / file
                rel_path = str(full_path.relative_to(workspace_path)).replace('\\', '/')
                
                files.append(rel_path)
                
                ext = full_path.suffix
                if ext:
                    file_types[ext] = file_types.get(ext, 0) + 1
                    
                # Basic heuristic component detection
                type_name = "file"
                pattern = "generic"
                
                if "routes" in rel_path or "route." in rel_path:
                    type_name = "route"
                    pattern = "routing"
                elif "components" in rel_path or "component." in rel_path:
                    type_name = "component"
                    pattern = "ui-component"
                elif "controllers" in rel_path or "controller." in rel_path:
                    type_name = "controller"
                    pattern = "mvc-controller"
                elif "models" in rel_path or "model." in rel_path:
                    type_name = "model"
                    pattern = "mvc-model"
                    
                if type_name != "file":
                    components.append(RepositoryComponent(
                        type=type_name,
                        file=rel_path,
                        framework_pattern=pattern
                    ))
                    
        logger.info(f"Generated map with {len(files)} files and {len(components)} identified components.")
        return RepositoryMapData(
            files=files,
            file_types=file_types,
            components=components
        )

    def load(self, index_dir: str) -> RepositoryMapData | None:
        target = Path(index_dir) / "repository_map.json"
        if target.exists():
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    return RepositoryMapData.model_validate_json(f.read())
            except Exception as e:
                logger.error(f"Failed to load repository map: {e}")
        return None

    def save(self, data: RepositoryMapData, index_dir: str) -> None:
        target = Path(index_dir) / "repository_map.json"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                f.write(data.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save repository map: {e}")
