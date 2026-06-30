import tempfile
import shutil
import time
import json
import os
import asyncio
from pathlib import Path
from agent.config import settings, logger
from agent.orchestrator import Orchestrator
from agent.evaluation.schemas import BenchmarkTask, BenchmarkResult, BenchmarkSuiteResult

class BenchmarkRunner:
    def __init__(self, suite_name: str, tasks: list[BenchmarkTask], reflection_enabled: bool = True, shadow_reflection: bool = False, claude_enabled: bool = True, claude_always_on: bool = False):
        self.suite_name = suite_name
        self.tasks = tasks
        self.reflection_enabled = reflection_enabled
        self.shadow_reflection = shadow_reflection
        self.claude_enabled = claude_enabled
        self.claude_always_on = claude_always_on
        self.original_workspace = settings.get_workspace_path()
        self.results_dir = Path("benchmarks/results")
        self.reports_dir = Path("benchmarks/reports")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    async def run_suite(self) -> BenchmarkSuiteResult:
        results = []
        for task in self.tasks:
            result = await self.run_task(task)
            results.append(result)

        total = len(results)
        if total == 0:
            return None
            
        successes = [r for r in results if r.success]
        success_rate = len(successes) / total
        
        repair_triggers = [r for r in results if r.repair_triggered]
        repair_trigger_rate = len(repair_triggers) / total
        
        avg_repair_attempts = sum(r.repair_attempts for r in results) / total
        
        rollbacks = [r for r in results if r.rollback_triggered]
        rollback_rate = len(rollbacks) / total
        
        total_constraint_violations = sum(r.constraint_violations for r in results)
        avg_execution_time = sum(r.execution_time for r in results) / total

        from datetime import datetime, timezone
        suite_result = BenchmarkSuiteResult(
            suite_name=self.suite_name,
            timestamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            total_tasks=total,
            success_rate=success_rate,
            repair_trigger_rate=repair_trigger_rate,
            avg_repair_attempts=avg_repair_attempts,
            rollback_rate=rollback_rate,
            total_constraint_violations=total_constraint_violations,
            avg_execution_time=avg_execution_time,
            results=results
        )
        
        return suite_result

    async def run_task(self, task: BenchmarkTask) -> BenchmarkResult:
        logger.info(f"--- Running Benchmark Task: {task.id} ---")
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_workspace = Path(temp_dir) / "workspace"
            shutil.copytree(self.original_workspace, temp_workspace)
            
            # Setup script if provided
            if task.setup_script:
                setup_path = temp_workspace / task.setup_script
                if setup_path.exists():
                    import subprocess
                    subprocess.run(["bash", task.setup_script], cwd=temp_workspace)
                    logger.info(f"Ran setup script: {task.setup_script}")

            try:
                orchestrator = Orchestrator(
                    workspace_path=temp_workspace, 
                    reports_dir=self.reports_dir, 
                    reflection_enabled=self.reflection_enabled,
                    shadow_reflection=self.shadow_reflection,
                    claude_enabled=self.claude_enabled,
                    claude_always_on=self.claude_always_on,
                    budget_enforcement=False # Phase 6.4: Benchmark mode safely bypasses budget checks
                )
                
                # Run the agent with timeout
                try:
                    report_path = await asyncio.wait_for(orchestrator.run(task.task), timeout=task.max_runtime_seconds)
                    duration = time.time() - start_time
                except asyncio.TimeoutError:
                    duration = time.time() - start_time
                    return BenchmarkResult(
                        task_id=task.id,
                        success=False,
                        repair_triggered=False,
                        repair_attempts=0,
                        rollback_triggered=False,
                        constraint_violations=0,
                        execution_time=duration,
                        patch_count=0,
                        validation_failures=0,
                        planner_failures=0,
                        coder_failures=0,
                        error_message=f"Task timed out after {duration:.2f} seconds."
                    )

                # Retrieve JSON report
                json_path = report_path.replace('.md', '.json')
                with open(json_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                success = (report_data.get("final_status") == task.expected_status)
                error_message = report_data.get("execution_results", "") if not success else None
                
                files_modified = report_data.get("files_modified", [])
                
                # Check expected files modified
                if task.expected_files_modified is not None:
                    missing = set(task.expected_files_modified) - set(files_modified)
                    extra = set(files_modified) - set(task.expected_files_modified)
                    if missing or extra:
                        success = False
                        errs = []
                        if missing: errs.append(f"Missing expected modified files: {missing}")
                        if extra: errs.append(f"Extra modified files not expected: {extra}")
                        error_message = " | ".join(errs)

                repair_history = report_data.get("repair_history", [])
                repair_metrics = report_data.get("repair_metrics") or {}
                
                repair_triggered = len(repair_history) > 0
                repair_attempts = repair_metrics.get("total_attempts", 0)
                rollback_triggered = repair_metrics.get("rollback_triggered", False)
                
                constraint_violations = len([r for r in repair_history if "CONSTRAINT_VIOLATION" in r.get("classification", "")])
                
                val_failures = 0
                val_report = report_data.get("validation_report") or {}
                if val_report.get("build_result") and not val_report["build_result"].get("success"): val_failures += 1
                if val_report.get("lint_result") and not val_report["lint_result"].get("success"): val_failures += 1
                if val_report.get("test_result") and not val_report["test_result"].get("success"): val_failures += 1
                
                exec_results = report_data.get("execution_results", "")
                planner_failures = 1 if "Agent encountered an error" in exec_results and "planner" in exec_results.lower() else 0
                coder_failures = 1 if "Agent encountered an error" in exec_results and "coder" in exec_results.lower() else 0
                
                conf_report = report_data.get("confidence_report") or {}
                confidence_score = conf_report.get("confidence_score") if conf_report else None
                review_decision = report_data.get("review_decision")
                
                reflection_triggered = report_data.get("reflection_triggered", False)
                reflection_retry_used = report_data.get("reflection_retry_used", False)
                reflection_result = report_data.get("reflection_result", "UNKNOWN")
                
                ext_report = report_data.get("external_review_report")
                claude_review_triggered = ext_report is not None
                claude_review_latency_ms = ext_report.get("latency_ms", 0.0) if ext_report else 0.0
                claude_issue_count = len(ext_report.get("issues", [])) if ext_report else 0
                # REPLAY CAPTURE HOOK
                if not success:
                    from agent.evaluation.replay_manager import ReplayManager
                    replay_mgr = ReplayManager()
                    replay_mgr.capture_failure(task.id, task.task, report_data, temp_workspace)

                return BenchmarkResult(
                    task_id=task.id,
                    success=success,
                    repair_triggered=repair_triggered,
                    repair_attempts=repair_attempts,
                    rollback_triggered=rollback_triggered,
                    constraint_violations=constraint_violations,
                    execution_time=duration,
                    patch_count=len(repair_history) + 1 if files_modified else 0,
                    validation_failures=val_failures,
                    planner_failures=planner_failures,
                    coder_failures=coder_failures,
                    error_message=error_message,
                    confidence_score=confidence_score,
                    review_decision=review_decision,
                    reflection_triggered=reflection_triggered,
                    reflection_retry_used=reflection_retry_used,
                    reflection_result=reflection_result or "UNKNOWN",
                    claude_review_triggered=claude_review_triggered,
                    claude_review_latency_ms=claude_review_latency_ms,
                    claude_issue_count=claude_issue_count,
                    report_path=json_path
                )
                
            except Exception as e:
                logger.error(f"Benchmark Runner Error on task {task.id}: {e}")
                return BenchmarkResult(
                    task_id=task.id,
                    success=False,
                    repair_triggered=False,
                    repair_attempts=0,
                    rollback_triggered=False,
                    constraint_violations=0,
                    execution_time=time.time() - start_time,
                    patch_count=0,
                    validation_failures=0,
                    planner_failures=0,
                    coder_failures=0,
                    error_message=str(e)
                )
