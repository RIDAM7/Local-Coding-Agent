import hashlib
import json
from typing import List, Optional
from agent.models.schemas import ValidationReport, RepairResult, RepairContext, RepairPatch, Task, Plan, RetrievedContext, RepairScope
from agent.retrieval.retrieval_manager import RetrievalManager
from agent.repair.normalizer import DiagnosticsNormalizer
from agent.repair.coder import RepairCoder
from agent.repair.rollback import RollbackManager
from agent.config import logger

class RepairManager:
    def __init__(self, retrieval_manager: RetrievalManager, repair_coder: RepairCoder, rollback_manager: RollbackManager, memory_manager=None):
        self.retrieval_manager = retrieval_manager
        self.repair_coder = repair_coder
        self.rollback_manager = rollback_manager
        self.memory_manager = memory_manager
        self.applied_patch_hashes = set()
        
    def _hash_patch(self, patch: RepairPatch) -> str:
        # Create a strict deterministic fingerprint from operations
        parts = []
        for op in patch.operations:
            parts.append(f"{op.type}:{op.path}:{op.content or ''}")
        fingerprint_str = "|".join(parts)
        return hashlib.sha256(fingerprint_str.encode('utf-8')).hexdigest()

    async def build_context(self, task: Task, plan: Plan | None, report: ValidationReport, constraints: list, repair_scope: Optional[RepairScope]) -> RepairContext:
        struct_diag, norm_diag = DiagnosticsNormalizer.normalize(report)
        
        mem_summaries = []
        if self.memory_manager and self.memory_manager.enabled:
            try:
                # Mock fingerprint for now, ideally passed down or extracted from workspace
                active_fingerprint = ["python"] 
                
                sim_fails = await self.memory_manager.similar_failures(norm_diag.classification, active_fingerprint, limit=2)
                repairs = await self.memory_manager.successful_repairs(norm_diag.classification, active_fingerprint, limit=3)
                
                combined = sim_fails + repairs
                for m in combined[:5]:
                    type_str = m.memory_type.value
                    mem_summaries.append(f"[{type_str}] {m.metadata.task[:100]}: {m.content[:300]}...")
            except Exception as e:
                logger.warning(f"Memory retrieval during repair failed, continuing: {e}")
                
        if mem_summaries:
            norm_diag.classification += "\n\nHistorical Context (Learnings from previous runs):\n"
            for s in mem_summaries:
                norm_diag.classification += f"- {s}\n"
                
        retrieved_context = await self.retrieval_manager.search_context(
            task=task.description,
            plan=plan,
            diagnostic=norm_diag
        )
        
        return RepairContext(
            original_task=task.description,
            diagnostics=struct_diag,
            normalized_diagnostic=norm_diag,
            retrieved_context=retrieved_context,
            constraints=constraints,
            repair_scope=repair_scope
        )
        
    async def generate_repair(self, context: RepairContext) -> Optional[RepairPatch]:
        patch = await self.repair_coder.generate_repair(context)
        
        patch_hash = self._hash_patch(patch)
        if patch_hash in self.applied_patch_hashes:
            logger.error("RepairManager: Generated an identical patch. Aborting to prevent infinite loop.")
            return None
            
        self.applied_patch_hashes.add(patch_hash)
        return patch
