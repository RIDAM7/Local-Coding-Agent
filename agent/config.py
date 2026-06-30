import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    planner_model: str = "qwen2.5:14b"
    coder_model: str = "qwen2.5-coder:32b"

    # Optional per-role model overrides (empty => inherit via the factory:
    # constraint<-planner, repair/reflection<-coder, refiner/reviewer<-planner).
    refiner_model: str = ""
    repair_model: str = ""
    constraint_model: str = ""
    reflection_model: str = ""
    reviewer_model: str = ""

    # Per-role provider routing (Phase 2). Supported: ollama, openai, anthropic, google.
    # Base roles default to "ollama"; derived roles default to "" meaning "inherit"
    # the same way models do (constraint<-planner, repair/reflection<-coder,
    # refiner/reviewer<-planner). With nothing set, every role is local Ollama.
    planner_provider: str = "ollama"
    coder_provider: str = "ollama"
    refiner_provider: str = ""
    repair_provider: str = ""
    constraint_provider: str = ""
    reflection_provider: str = ""
    reviewer_provider: str = ""

    # Phase 3: optional prompt refiner (pre-planning rewrite). Off by default so
    # the pipeline is byte-for-byte unchanged. When enabled, it uses the "refiner"
    # role's provider/model (REFINER_PROVIDER / REFINER_MODEL, inheriting planner).
    refiner_enabled: bool = False

    # Anthropic settings
    anthropic_api_key: str = ""
    claude_model: str = "claude-3-5-sonnet-20240620"

    # OpenAI / OpenAI-compatible gateway settings (OpenRouter, Groq, Together, ...)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Google Gemini settings
    google_api_key: str = ""
    
    # Workspace settings
    workspace_dir: str = "./workspace"
    
    # Execution
    command_timeout: int = 60

    # Phase 4a: execute the coder's proposed shell commands after a patch applies.
    # OFF by default — until the Phase 5 safety layer (allowlist/diff-preview/jail)
    # lands, nothing runs silently. When false, commands are reported as proposed
    # only and never executed.
    execute_commands: bool = False
    
    # LLM Retries
    max_retries: int = 3
    
    # Validation Commands
    build_command: str = ""
    lint_command: str = ""
    test_command: str = ""
    
    # Repair
    max_repair_attempts: int = 3

    # Phase 7B: git integration. When true AND workspace/ is a git repo AND not in
    # --dry-run, create a task branch at run start and commit applied changes on a
    # successful run (complementing the existing snapshot rollback). Off by default
    # so existing runs are byte-for-byte unchanged.
    git_integration: bool = False

    # Phase 9: Context Engine (Repository Understanding). A proactive, 100%-local
    # layer (on top of agent/retrieval) that builds a durable repo understanding
    # — tech stack, frameworks, entry points, conventions, dependency map,
    # architecture summary — before planning. Default-ON because it is local,
    # zero-config, cached, and never makes a network call. When disabled, the
    # planner prompt is byte-for-byte identical to Round 1 (pipeline parity).
    #   CONTEXT_ENGINE_ENABLED — master switch (default true).
    #   CONTEXT_CACHE          — reuse the cached bundle when the file set is
    #                            unchanged, instead of a full rescan (default true).
    #   CONTEXT_DIR            — where the bundle is cached. Relative paths are
    #                            resolved under the scanned workspace.
    context_engine_enabled: bool = True
    context_cache: bool = True
    context_dir: str = ".localcli/context"

    # Phase 10: Hybrid Execution Engine.
    #   EXECUTION_MODE — auto | pipeline | agent. `pipeline` is byte-for-byte the
    #     Round 1 flow; `agent` runs the governed Think->Act->Observe tool loop;
    #     `auto` lets the Capability Detector pick (agent only when the role model
    #     has reliable structured output + tool calling + adequate context, else
    #     pipeline). Default `auto` resolves to pipeline whenever capabilities are
    #     unknown/insufficient, so existing local setups behave as before.
    execution_mode: str = "auto"
    #   OFFLINE_ONLY — hard offline guarantee: preflight rejects any non-Ollama
    #     provider and the web/remote tools are not registered. Default false.
    offline_only: bool = False
    #   Execution Governor caps (shared by BOTH engines). 0 == disabled where noted.
    max_steps: int = 25
    tool_call_budget: int = 0
    run_budget_usd: float = 0.0
    step_timeout_seconds: int = 120
    run_timeout_seconds: int = 0
    #   Capability Detector: allow a one-time cached probe for unknown models.
    capability_probe: bool = True
    #   Reviewer escalation (replaces the hard Claude coupling): only runs when the
    #     governor exhausts iterations with confidence < CONFIDENCE_THRESHOLD.
    reviewer_enabled: bool = False
    confidence_threshold: float = 0.75

    # Phase 11: Incremental Planning & Replanning. Replaces plan-once-execute-all
    # with plan -> execute step -> observe -> replan. Small, validated steps with
    # replanning on new information.
    #   INCREMENTAL_PLANNING — master switch (default true). When false, behavior
    #     is the Round 1 plan-once-execute-all path, byte-for-byte (parity).
    #   REPLAN_ON_FAILURE   — when a step fails, revise only the REMAINING steps
    #     (never a full restart; completed steps are untouched). Default true.
    #   MAX_REPLANS         — bound on replanning, enforced by the Execution
    #     Governor (a replan counts against the leash with its own stop reason).
    #     0 disables the cap. Default 3.
    incremental_planning: bool = True
    replan_on_failure: bool = True
    max_replans: int = 3

    # Phase 12: Project Memory. Local, human-readable markdown files that are
    # loaded at run start and updated only after successful runs. This is NOT
    # vector memory: no embeddings, cloud calls, or external database.
    project_memory_enabled: bool = True
    project_memory_dir: str = ".localcli/memory"

    # Phase 13: Repository Graph MVP. Local graph.json with import/module edges
    # and impact queries for primary languages (py/js/ts). No call graph here.
    repo_graph_enabled: bool = True
    graph_dir: str = ".localcli"

    # Phase 14: Observability. Timeline events live on AgentState and the
    # dashboard/report are read-only renderers over that state.
    observability_enabled: bool = True
    verbosity: str = "normal"  # quiet | normal | verbose

    # Phase 15: Session Persistence. Checkpoint AgentState after every step
    # and enable `localcli resume` to continue an interrupted session.
    # When disabled, no checkpoint files are written and resume commands
    # raise a clear error.
    session_persistence: bool = True
    session_dir: str = ".localcli"

    # Phase 16: Agent Orchestration Layer. Opt-in coordinator that decomposes
    # multi-concern tasks into stateless specialized workers. Default off so
    # existing runs are byte-for-byte unchanged.
    orchestration_enabled: bool = False
    max_parallel_workers: int = 1
    orchestration_budget_usd: float = 0.0

    # Phase 17: Plugin Architecture. Community plugins and MCP are opt-in;
    # builtin and project plugins load without flags. All plugin tools register
    # into the same ToolRegistry from Phase 10.
    plugins_enabled: bool = False
    plugin_dir: str = ".localcli/community_plugins"
    plugin_tool_allow: str = ""
    plugin_tool_deny: str = ""
    mcp_enabled: bool = False
    mcp_servers: str = ""

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

        # Phase 5: scrub secrets from every record before it reaches a handler so
        # nothing sensitive is ever written to logs/agent.log. Imported here (after
        # settings exists) to avoid an import cycle with the safety package.
        from agent.safety.redact import RedactionFilter
        redaction_filter = RedactionFilter()

        # File handler
        fh = logging.FileHandler('logs/agent.log')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        fh.addFilter(redaction_filter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        ch.addFilter(redaction_filter)
        logger.addHandler(ch)

    return logger

logger = setup_logging()
