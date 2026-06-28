from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class ProvenanceType(str, Enum):
    BENCHMARK = "BENCHMARK"
    USER_TASK = "USER_TASK"
    MANUAL_TEST = "MANUAL_TEST"
    REGRESSION = "REGRESSION"

class ReplayTier(str, Enum):
    GOLDEN = "GOLDEN"
    REGRESSION = "REGRESSION"
    NORMAL = "NORMAL"

from pydantic import BaseModel, Field, model_validator

class EnvironmentSnapshot(BaseModel):
    python_version: str
    os: str
    agent_version: str
    target_git_commit: str = "unknown"
    ollama_models: List[str]
    dependency_versions: Dict[str, str] = {}

    @model_validator(mode='before')
    @classmethod
    def migrate_legacy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "os_info" in data and "os" not in data:
                data["os"] = data["os_info"]
            if "configured_models" in data and "ollama_models" not in data:
                data["ollama_models"] = list(data["configured_models"].values())
        return data

from typing import Literal

class OracleSolution(BaseModel):
    source: Literal["HUMAN", "GPT4", "CLAUDE", "CODEX"]
    patch: Dict[str, Any]
    verified_success: bool
    timestamp: str

class DatasetManifest(BaseModel):
    dataset_version: str
    generated_at: str
    replay_ids_included: List[str]
    filters_applied: Dict[str, Any]
    dataset_hash: str

class ReplayEvolutionTracking(BaseModel):
    first_failure_date: str
    last_failure_date: str
    first_success_date: Optional[str] = None
    fixing_phase: Optional[str] = None
    root_cause_classification: str

class ReplayArtifact(BaseModel):
    replay_id: str
    artifact_version: str = "4.7B"
    timestamp: str
    task: str
    benchmark_id: str
    environment: EnvironmentSnapshot
    constraints: List[str]
    plan: Optional[Dict[str, Any]]
    diagnostics: Optional[Dict[str, Any]]
    repair_history: List[Dict[str, Any]]
    final_status: str
    snapshot_uri: str
    archive_hash: str
    archive_file_count: int
    provenance: ProvenanceType = ProvenanceType.BENCHMARK
    tier: ReplayTier = ReplayTier.NORMAL
    difficulty_score: float = 0.0
    quality_score: float = 0.0
    evolution_tracking: Optional[ReplayEvolutionTracking] = None
    oracle_solutions: List[OracleSolution] = []

    @model_validator(mode='before')
    @classmethod
    def infer_version(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "artifact_version" not in data:
                # If tier or provenance are missing, it's a 4.7A artifact
                if "tier" not in data and "difficulty_score" not in data:
                    data["artifact_version"] = "4.7A"
                else:
                    data["artifact_version"] = "4.7B"
        return data
