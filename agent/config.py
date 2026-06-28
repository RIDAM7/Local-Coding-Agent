import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    planner_model: str = "qwen2.5:14b"
    coder_model: str = "qwen2.5-coder:32b"
    
    # Anthropic settings
    anthropic_api_key: str = ""
    claude_model: str = "claude-3-5-sonnet-20240620"
    
    # Workspace settings
    workspace_dir: str = "./workspace"
    
    # Execution
    command_timeout: int = 60
    
    # LLM Retries
    max_retries: int = 3
    
    # Validation Commands
    build_command: str = ""
    lint_command: str = ""
    test_command: str = ""
    
    # Repair
    max_repair_attempts: int = 3
    
    # Logging
    log_level: str = "DEBUG"

    def get_workspace_path(self) -> Path:
        return Path(self.workspace_dir).resolve()

settings = AgentSettings()

# Ensure directories exist
os.makedirs(settings.get_workspace_path(), exist_ok=True)
os.makedirs(Path("logs"), exist_ok=True)
os.makedirs(Path("reports"), exist_ok=True)

import logging
import sys

def setup_logging():
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    logger = logging.getLogger("agent")
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler('logs/agent.log')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

logger = setup_logging()
