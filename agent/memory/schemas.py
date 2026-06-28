from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class MemoryType(str, Enum):
    REPAIR_SUCCESS = "REPAIR_SUCCESS"
    ORACLE_SOLUTION = "ORACLE_SOLUTION"
    REPLAY_ARTIFACT = "REPLAY_ARTIFACT"
    BENCHMARK_OUTCOME = "BENCHMARK_OUTCOME"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"

class MemoryMetadata(BaseModel):
    task: str
    diagnostics: str = ""
    constraints: List[str] = []
    patch_summary: str = ""
    outcome: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    source_id: str
    access_count: int = 0
    last_accessed: Optional[str] = None
    workspace_fingerprint: List[str] = []

class MemoryRecord(BaseModel):
    memory_id: str
    memory_version: str = "v1"
    memory_type: MemoryType
    importance_score: float
    embedding_text: str
    content: str
    metadata: MemoryMetadata
    
    # Optional dynamic fields during retrieval
    retrieval_similarity: float = 0.0
    compatibility_score: float = 0.0
    final_score: float = 0.0
    retrieval_reason: str = ""

def get_importance_score(memory_type: MemoryType) -> float:
    mapping = {
        MemoryType.ORACLE_SOLUTION: 1.0,
        MemoryType.REPAIR_SUCCESS: 0.9,
        MemoryType.CONSTRAINT_VIOLATION: 0.8,
        MemoryType.REPLAY_ARTIFACT: 0.5,
        MemoryType.BENCHMARK_OUTCOME: 0.3
    }
    return mapping.get(memory_type, 0.0)
