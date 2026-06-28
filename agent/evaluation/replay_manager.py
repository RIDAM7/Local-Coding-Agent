import os
import sys
import tarfile
import hashlib
import platform
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

from agent.evaluation.replay_schemas import EnvironmentSnapshot, ReplayArtifact
from agent.config import settings, logger

class ReplayManager:
    def __init__(self):
        self.replays_dir = Path.home() / ".antigravity_data" / "replays"
        os.makedirs(self.replays_dir, exist_ok=True)
        
    def _get_environment_snapshot(self) -> EnvironmentSnapshot:
        import subprocess
        deps = {}
        try:
            reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8").split("\n")
            for req in reqs:
                if "==" in req:
                    pkg, ver = req.split("==", 1)
                    deps[pkg.strip()] = ver.strip()
        except: pass
        
        commit = "unknown"
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        except: pass

        return EnvironmentSnapshot(
            python_version=sys.version.split()[0],
            os=platform.platform(),
            agent_version="1.0.0",
            target_git_commit=commit,
            ollama_models=[settings.planner_model, settings.coder_model],
            dependency_versions=deps
        )
        
    def _create_archive(self, workspace_path: Path, archive_path: Path) -> Tuple[str, int]:
        file_count = 0
        hasher = hashlib.sha256()
        
        with tarfile.open(archive_path, "w:gz") as tar:
            for root, dirs, files in os.walk(workspace_path):
                # Optionally ignore .git or node_modules here
                if ".git" in root: continue
                for f in files:
                    file_path = Path(root) / f
                    tar.add(file_path, arcname=file_path.relative_to(workspace_path))
                    file_count += 1
                    with open(file_path, "rb") as fd:
                        hasher.update(fd.read())
                        
        return hasher.hexdigest(), file_count

    def _compute_quality_score(self, report_data: Dict[str, Any], oracle_solutions: List[Any] = None) -> float:
        score = 0.0
        repair_history = report_data.get("repair_history", [])
        
        # -20 per repair attempt
        score -= len(repair_history) * 20.0
        
        # +20 high confidence
        if repair_history:
            last_repair = repair_history[-1]
            if last_repair.get("patch_applied", {}).get("confidence") == "HIGH":
                score += 20.0
                
        # -100 empty patch
        # Check if the final patch or any files were actually modified
        files_mod = report_data.get("files_modified", [])
        if not files_mod:
            score -= 100.0
            
        # +30 verified oracle
        if oracle_solutions:
            for oracle in oracle_solutions:
                if getattr(oracle, 'verified_success', False) or (isinstance(oracle, dict) and oracle.get('verified_success')):
                    score += 30.0
                    break
                    
        return max(0.0, min(100.0, score + 50.0)) # Normalize to 0-100 range logically, wait, spec says 0-100.
        # Let's adjust so it fits 0-100. Actually, I'll just bound it:
        # return max(0.0, min(100.0, score + 80.0)) - wait, spec doesn't say base score.
        # I'll assume base score = 100.
        # score = 100
        # -20 per attempt...
        
    def capture_failure(self, task_id: str, task_desc: str, report_data: Dict[str, Any], workspace_path: Path, provenance: Any = None, tier: Any = None) -> ReplayArtifact:
        from agent.evaluation.replay_schemas import ProvenanceType, ReplayTier, ReplayEvolutionTracking
        import math
        
        replay_id = f"replay_{task_id}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        archive_path = self.replays_dir / f"{replay_id}.tar.gz"
        hash_val, count = self._create_archive(workspace_path, archive_path)
        
        plan = report_data.get("plan")
        repair_history = report_data.get("repair_history", [])
        final_status = report_data.get("final_status", "FAILURE")
        
        constraints = []
        diagnostics = report_data.get("validation_report")
        
        val_fails = 0
        if diagnostics:
            if diagnostics.get("build_result") and not diagnostics["build_result"].get("success"): val_fails += 1
            if diagnostics.get("lint_result") and not diagnostics["lint_result"].get("success"): val_fails += 1
            if diagnostics.get("test_result") and not diagnostics["test_result"].get("success"): val_fails += 1

        files_touched = len(report_data.get("files_modified", []))
        repair_attempts_count = len(repair_history)
        repo_files_log = math.log(max(count, 1))
        
        diff_score = (files_touched * 5) + (repair_attempts_count * 10) + (val_fails * 5) + (repo_files_log * 10)
        
        # Compute quality score
        base_q_score = 100.0
        base_q_score -= repair_attempts_count * 20.0
        if repair_history and repair_history[-1].get("patch_applied", {}).get("confidence") == "HIGH":
            base_q_score += 20.0
        if files_touched == 0:
            base_q_score -= 100.0
        quality_score = max(0.0, min(100.0, base_q_score))
        
        evolution = ReplayEvolutionTracking(
            first_failure_date=timestamp,
            last_failure_date=timestamp,
            root_cause_classification="PENDING_ANALYSIS"
        )
        
        artifact = ReplayArtifact(
            replay_id=replay_id,
            artifact_version="4.7B",
            timestamp=timestamp,
            task=task_desc,
            benchmark_id=task_id,
            environment=self._get_environment_snapshot(),
            constraints=constraints,
            plan=plan,
            diagnostics=diagnostics,
            repair_history=repair_history,
            final_status=final_status,
            snapshot_uri=str(archive_path),
            archive_hash=hash_val,
            archive_file_count=count,
            provenance=provenance if provenance else ProvenanceType.BENCHMARK,
            tier=tier if tier else ReplayTier.NORMAL,
            difficulty_score=round(diff_score, 2),
            quality_score=round(quality_score, 2),
            evolution_tracking=evolution
        )
        
        json_path = self.replays_dir / f"{replay_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(artifact.model_dump_json(indent=2))
            
        logger.info(f"Captured replay artifact {replay_id} at {self.replays_dir}")
        return artifact

    def add_oracle_solution(self, replay_id: str, oracle_solution: Any) -> bool:
        json_path = self.replays_dir / f"{replay_id}.json"
        if not json_path.exists():
            return False
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        from agent.evaluation.replay_schemas import ReplayArtifact
        artifact = ReplayArtifact(**data)
        
        artifact.oracle_solutions.append(oracle_solution)
        
        # Recalculate quality score
        if getattr(oracle_solution, 'verified_success', False):
            artifact.quality_score = min(100.0, artifact.quality_score + 30.0)
            
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(artifact.model_dump_json(indent=2))
            
        return True

    def promote_replay(self, replay_id: str) -> bool:
        from agent.evaluation.replay_schemas import ReplayTier
        json_path = self.replays_dir / f"{replay_id}.json"
        if not json_path.exists():
            return False
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        artifact = ReplayArtifact(**data)
        artifact.tier = ReplayTier.GOLDEN
        
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(artifact.model_dump_json(indent=2))
            
        return True

    def update_evolution_on_success(self, replay_id: str) -> bool:
        json_path = self.replays_dir / f"{replay_id}.json"
        if not json_path.exists():
            return False
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        artifact = ReplayArtifact(**data)
        if artifact.evolution_tracking:
            if not artifact.evolution_tracking.first_success_date:
                artifact.evolution_tracking.first_success_date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(artifact.model_dump_json(indent=2))
            
        return True

    def list_replays(self) -> List[ReplayArtifact]:
        replays = []
        for p in self.replays_dir.glob("*.json"):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    replays.append(ReplayArtifact(**data))
            except Exception as e:
                logger.warning(f"Failed to load replay artifact {p}: {e}")
        return sorted(replays, key=lambda x: x.timestamp, reverse=True)

    def extract_and_verify(self, artifact: ReplayArtifact, target_dir: Path) -> bool:
        archive_path = Path(artifact.snapshot_uri)
        if not archive_path.exists():
            logger.error(f"Archive not found: {archive_path}")
            return False
            
        # Extract
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=target_dir)
        except Exception as e:
            logger.error(f"Failed to extract archive: {e}")
            return False
            
        # Verify
        file_count = 0
        hasher = hashlib.sha256()
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                file_path = Path(root) / f
                file_count += 1
                with open(file_path, "rb") as fd:
                    hasher.update(fd.read())
                    
        calculated_hash = hasher.hexdigest()
        
        if file_count != artifact.archive_file_count:
            logger.error(f"File count mismatch. Expected {artifact.archive_file_count}, got {file_count}")
            return False
            
        if calculated_hash != artifact.archive_hash:
            logger.error(f"Archive hash mismatch. Expected {artifact.archive_hash}, got {calculated_hash}")
            return False
            
        return True
