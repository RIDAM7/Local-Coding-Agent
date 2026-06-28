import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiohttp
from datetime import datetime

from agent.memory.schemas import MemoryRecord, MemoryType, MemoryMetadata, get_importance_score

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, memory_dir: Path, ollama_url: str = "http://localhost:11434"):
        self.memory_dir = memory_dir
        self.ollama_url = ollama_url
        self.stats_file = self.memory_dir / "stats.json"
        
        os.makedirs(self.memory_dir, exist_ok=True)
        
        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=str(self.memory_dir / "chroma_db"))
            self.collection = self.chroma_client.get_or_create_collection(name="agent_memory")
            self.enabled = True
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.enabled = False
            
    async def _get_embedding(self, text: str) -> List[float]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ollama_url}/api/embeddings"
                payload = {
                    "model": "nomic-embed-text",
                    "prompt": text
                }
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("embedding", [])
                    else:
                        logger.warning(f"Ollama embedding failed: {response.status}")
                        return []
        except Exception as e:
            logger.warning(f"Ollama connection error during embedding: {e}")
            return []

    def _scrub_paths(self, text: str) -> str:
        # Simple neutralizer to prevent absolute path pollution
        import re
        # Windows paths like C:\Users\...
        text = re.sub(r'[a-zA-Z]:\\[^\s"\']+', '[SCRUBBED_PATH]', text)
        # Unix paths like /home/user/...
        text = re.sub(r'/[^\s"\']+', '[SCRUBBED_PATH]', text)
        return text

    def _update_stats(self, op: str, type_str: str):
        stats = {}
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
            except:
                pass
                
        stats.setdefault("total_records", 0)
        stats.setdefault("retrieval_count", 0)
        stats.setdefault("successful_retrieval_count", 0)
        stats.setdefault("by_type", {})
        
        if type_str:
            stats["by_type"].setdefault(type_str, 0)
        
        if op == "store":
            stats["total_records"] += 1
            stats["by_type"][type_str] += 1
        elif op == "retrieve":
            stats["retrieval_count"] += 1
        elif op == "successful_retrieve":
            stats["successful_retrieval_count"] += 1
            
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

    async def store(self, record: MemoryRecord) -> str:
        if not self.enabled:
            return ""
            
        scrubbed_emb_text = self._scrub_paths(record.embedding_text)
        embedding = await self._get_embedding(scrubbed_emb_text)
        
        if not embedding:
            return ""
            
        try:
            meta_dict = record.metadata.model_dump()
            # ChromaDB only accepts str, int, float, bool. Convert lists.
            meta_dict["constraints"] = json.dumps(meta_dict["constraints"] or [])
            meta_dict["workspace_fingerprint"] = json.dumps(meta_dict["workspace_fingerprint"] or [])
            
            if "last_accessed" in meta_dict and isinstance(meta_dict["last_accessed"], datetime):
                meta_dict["last_accessed"] = meta_dict["last_accessed"].isoformat()
                
            # Filter None
            meta_dict = {k: v for k, v in meta_dict.items() if v is not None}
            
            # Store memory type for filtering
            meta_dict["memory_type"] = record.memory_type.value
            meta_dict["importance_score"] = float(record.importance_score)
            
            self.collection.add(
                ids=[record.memory_id],
                embeddings=[embedding],
                metadatas=[meta_dict],
                documents=[record.content]
            )
            
            self._update_stats("store", record.memory_type.value)
            return record.memory_id
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return ""

    def _calculate_compatibility(self, record_fingerprint_str: str, active_fingerprints: List[str]) -> float:
        if not active_fingerprints:
            return 1.0 # If we don't have active fingerprints, don't penalize
            
        try:
            record_fps = json.loads(record_fingerprint_str)
        except:
            record_fps = []
            
        if not record_fps:
            return 0.5 # Neutral if record has no fingerprint
            
        overlap = set(active_fingerprints).intersection(set(record_fps))
        return len(overlap) / max(len(active_fingerprints), 1)

    async def search(self, query: str, active_fingerprints: List[str], limit: int = 5, filter_type: Optional[str] = None) -> List[MemoryRecord]:
        if not self.enabled:
            return []
            
        scrubbed_query = self._scrub_paths(query)
        embedding = await self._get_embedding(scrubbed_query)
        
        if not embedding:
            return []
            
        try:
            where_clause = {}
            if filter_type:
                where_clause = {"memory_type": filter_type}
                
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit * 2, # Fetch more to rerank
                where=where_clause if where_clause else None
            )
            
            records = []
            if not results["ids"] or not results["ids"][0]:
                return []
                
            for idx in range(len(results["ids"][0])):
                mem_id = results["ids"][0][idx]
                meta = results["metadatas"][0][idx]
                content = results["documents"][0][idx]
                distance = results["distances"][0][idx] if results["distances"] else 0.0
                
                # Convert distance to similarity (rough proxy: cosine dist -> sim)
                # Chroma uses L2 by default. L2 dist squared.
                # Assuming embeddings are normalized: dist^2 = 2 - 2*cos_sim => cos_sim = 1 - dist^2 / 2
                sim = 1.0 - (distance / 2.0)
                sim = max(0.0, min(1.0, sim))
                
                importance = meta.get("importance_score", 0.0)
                
                # Calculate compatibility
                compat_score = self._calculate_compatibility(meta.get("workspace_fingerprint", "[]"), active_fingerprints)
                
                # Filter strictly incompatible 
                if compat_score == 0.0 and active_fingerprints:
                    continue
                    
                # Rerank
                final_score = (sim * 0.8) + (importance * 0.2)
                
                # Reconstruct Metadata
                try:
                    meta_constraints = json.loads(meta.get("constraints", "[]"))
                    meta_fingerprint = json.loads(meta.get("workspace_fingerprint", "[]"))
                except:
                    meta_constraints = []
                    meta_fingerprint = []
                    
                metadata = MemoryMetadata(
                    task=meta.get("task", ""),
                    diagnostics=meta.get("diagnostics", ""),
                    constraints=meta_constraints,
                    patch_summary=meta.get("patch_summary", ""),
                    outcome=meta.get("outcome", ""),
                    timestamp=meta.get("timestamp", ""),
                    source_id=meta.get("source_id", ""),
                    access_count=meta.get("access_count", 0),
                    last_accessed=meta.get("last_accessed"),
                    workspace_fingerprint=meta_fingerprint
                )
                
                from agent.memory.schemas import get_importance_score, MemoryType
                try:
                    mem_type = MemoryType(meta.get("memory_type"))
                except:
                    mem_type = MemoryType.BENCHMARK_OUTCOME
                    
                record = MemoryRecord(
                    memory_id=mem_id,
                    memory_type=mem_type,
                    importance_score=importance,
                    embedding_text="", # We don't need it back
                    content=content,
                    metadata=metadata,
                    retrieval_similarity=sim,
                    compatibility_score=compat_score,
                    final_score=final_score,
                    retrieval_reason=f"Similarity: {sim:.2f}, Importance: {importance:.2f}, Compatibility: {compat_score:.2f}"
                )
                records.append(record)
                
            final_records = records[:limit]
            
            # Update Memory Usage Tracking
            if final_records:
                self._update_stats("retrieve", "")
                for r in final_records:
                    r.metadata.access_count += 1
                    r.metadata.last_accessed = datetime.utcnow().isoformat()
                    try:
                        meta_dict = r.metadata.model_dump()
                        meta_dict["constraints"] = json.dumps(meta_dict["constraints"] or [])
                        meta_dict["workspace_fingerprint"] = json.dumps(meta_dict["workspace_fingerprint"] or [])
                        meta_dict["memory_type"] = r.memory_type.value
                        meta_dict["importance_score"] = float(r.importance_score)
                        meta_dict = {k: v for k, v in meta_dict.items() if v is not None}
                        
                        self.collection.update(
                            ids=[r.memory_id],
                            metadatas=[meta_dict]
                        )
                    except Exception as meta_e:
                        logger.warning(f"Failed to update metadata tracking: {meta_e}")
                        
            return final_records
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def similar_failures(self, diagnostics: str, active_fingerprints: List[str], limit: int = 3) -> List[MemoryRecord]:
        return await self.search(diagnostics, active_fingerprints, limit, filter_type=MemoryType.REPLAY_ARTIFACT.value)

    async def successful_repairs(self, task_context: str, active_fingerprints: List[str], limit: int = 3) -> List[MemoryRecord]:
        # Merge repair successes and oracle solutions
        repairs = await self.search(task_context, active_fingerprints, limit, filter_type=MemoryType.REPAIR_SUCCESS.value)
        oracles = await self.search(task_context, active_fingerprints, limit, filter_type=MemoryType.ORACLE_SOLUTION.value)
        merged = repairs + oracles
        merged.sort(key=lambda x: x.final_score, reverse=True)
        return merged[:limit]

    async def constraint_similar_failures(self, constraint: str, active_fingerprints: List[str], limit: int = 3) -> List[MemoryRecord]:
        return await self.search(constraint, active_fingerprints, limit, filter_type=MemoryType.CONSTRAINT_VIOLATION.value)

    async def ingest_replay_artifact(self, replay_dict: Dict[str, Any]):
        if not self.enabled: return
        
        status = replay_dict.get("final_status")
        rep_history = replay_dict.get("repair_history", [])
        
        # Ingest Repair Success
        if status == "SUCCESS" and len(rep_history) > 0:
            last_repair = rep_history[-1]
            diag = last_repair.get("validation_result", {}).get("diagnostics", "")
            patch = json.dumps(last_repair.get("patch_applied", {}))
            rec = MemoryRecord(
                memory_id=replay_dict["replay_id"] + "_repair",
                memory_type=MemoryType.REPAIR_SUCCESS,
                importance_score=get_importance_score(MemoryType.REPAIR_SUCCESS),
                embedding_text=diag[:500],
                content=f"Diagnostics: {diag}\nPatch: {patch}",
                metadata=MemoryMetadata(
                    task=replay_dict.get("task", ""),
                    source_id=replay_dict["replay_id"],
                    workspace_fingerprint=[]
                )
            )
            await self.store(rec)
            
        # Ingest Constraint Violations
        for i, rep in enumerate(rep_history):
            if "CONSTRAINT_VIOLATION" in rep.get("classification", ""):
                patch = json.dumps(rep.get("patch_applied", {}))
                rec = MemoryRecord(
                    memory_id=replay_dict["replay_id"] + f"_cv_{i}",
                    memory_type=MemoryType.CONSTRAINT_VIOLATION,
                    importance_score=get_importance_score(MemoryType.CONSTRAINT_VIOLATION),
                    embedding_text=rep.get("classification", ""),
                    content=f"Violation: {rep.get('classification')}\nPatch: {patch}",
                    metadata=MemoryMetadata(
                        task=replay_dict.get("task", ""),
                        source_id=replay_dict["replay_id"],
                        workspace_fingerprint=[]
                    )
                )
                await self.store(rec)
                
        # Ingest Oracle Solutions
        for i, oracle in enumerate(replay_dict.get("oracle_solutions", [])):
            if hasattr(oracle, "model_dump"):
                oracle = oracle.model_dump()
            elif hasattr(oracle, "__dict__"):
                oracle = oracle.__dict__
            patch = json.dumps(oracle.get("patch", {}))
            rec = MemoryRecord(
                memory_id=replay_dict["replay_id"] + f"_oracle_{i}",
                memory_type=MemoryType.ORACLE_SOLUTION,
                importance_score=get_importance_score(MemoryType.ORACLE_SOLUTION),
                embedding_text=replay_dict.get("task", ""),
                content=f"Task: {replay_dict.get('task')}\nOracle Patch: {patch}",
                metadata=MemoryMetadata(
                    task=replay_dict.get("task", ""),
                    source_id=replay_dict["replay_id"],
                    workspace_fingerprint=[]
                )
            )
            await self.store(rec)
            
        # High value failure
        if status == "FAILURE" and replay_dict.get("difficulty_score", 0) > 50:
            rec = MemoryRecord(
                memory_id=replay_dict["replay_id"] + "_fail",
                memory_type=MemoryType.REPLAY_ARTIFACT,
                importance_score=get_importance_score(MemoryType.REPLAY_ARTIFACT),
                embedding_text=replay_dict.get("task", ""),
                content=str(replay_dict),
                metadata=MemoryMetadata(
                    task=replay_dict.get("task", ""),
                    source_id=replay_dict["replay_id"],
                    workspace_fingerprint=[]
                )
            )
            await self.store(rec)
