import json
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

from agent.evaluation.schemas import BenchmarkSuiteResult, BenchmarkResult
from agent.evaluation.improvement_schemas import (
    FailureAnalysis, Recommendation, HealthScore, AnalysisResult, HistoricalRun,
    TaxonomyClassification, RootCauseAttribution
)

class FailureAnalyzer:
    def __init__(self):
        self.history_dir = Path("benchmarks/history")
        os.makedirs(self.history_dir, exist_ok=True)
        self.history_file = self.history_dir / "benchmark_history.json"

    def analyze(self, suite_json_path: str) -> AnalysisResult:
        with open(suite_path := Path(suite_json_path), 'r', encoding='utf-8') as f:
            suite = BenchmarkSuiteResult.model_validate_json(f.read())

        failure_analyses = []
        tax_counts = defaultdict(int)
        rc_counts = defaultdict(int)

        for res in suite.results:
            if not res.success:
                analysis = self._analyze_failure(res, suite_path.parent)
                failure_analyses.append(analysis)
                for tax in analysis.taxonomies:
                    tax_counts[tax] += 1
                rc_counts[analysis.root_cause] += 1

        health_score = self._calculate_health_score(suite, tax_counts, rc_counts)
        recommendations = self._generate_recommendations(tax_counts, rc_counts)

        analysis_result = AnalysisResult(
            benchmark_run=suite_path.name,
            timestamp=suite.timestamp,
            health_score=health_score,
            failure_analyses=failure_analyses,
            taxonomy_counts=dict(tax_counts),
            root_cause_counts=dict(rc_counts),
            recommendations=recommendations
        )

        self._save_history(suite, analysis_result)
        return analysis_result

    def _analyze_failure(self, res: BenchmarkResult, base_dir: Path) -> FailureAnalysis:
        taxonomies = []
        explanation = []

        if res.error_message and "timed out" in res.error_message.lower():
            taxonomies.append("TIMEOUT")
            explanation.append("Task timed out.")
            return FailureAnalysis(task_id=res.task_id, taxonomies=taxonomies, root_cause="Unknown", explanation=" ".join(explanation))

        if not res.report_path:
            taxonomies.append("UNHANDLED_EXCEPTION")
            return FailureAnalysis(task_id=res.task_id, taxonomies=taxonomies, root_cause="Unknown", explanation="No report path provided.")

        json_report_path = Path(res.report_path.replace('.md', '.json'))
        if not json_report_path.exists():
             json_report_path = Path("benchmarks") / json_report_path
             
        if not json_report_path.exists():
            taxonomies.append("UNHANDLED_EXCEPTION")
            return FailureAnalysis(task_id=res.task_id, taxonomies=taxonomies, root_cause="Unknown", explanation=f"Report json missing: {json_report_path}")

        try:
            with open(json_report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception:
            taxonomies.append("UNHANDLED_EXCEPTION")
            return FailureAnalysis(task_id=res.task_id, taxonomies=taxonomies, root_cause="Unknown", explanation="Failed to load report json.")

        # Heuristics
        exec_res = report.get("execution_results", "").lower()
        if "constraint extraction failed" in exec_res:
            taxonomies.append("CONSTRAINT_EXTRACTION_FAILURE")
        elif res.planner_failures > 0:
            taxonomies.append("PLANNER_SCHEMA_ERROR")
            
        repair_history = report.get("repair_history", [])
        for rh in repair_history:
            cls = rh.get("classification", "")
            if "CONSTRAINT_VIOLATION" in cls and "CONSTRAINT_VIOLATION" not in taxonomies:
                taxonomies.append("CONSTRAINT_VIOLATION")
            elif "EMPTY_PATCH" in cls and "EMPTY_PATCH" not in taxonomies:
                taxonomies.append("EMPTY_PATCH")
            elif "PATCH_FAILURE" in cls and "PATCH_VALIDATION_FAILURE" not in taxonomies:
                taxonomies.append("PATCH_VALIDATION_FAILURE")
            # Newly implemented taxonomies
            elif "SYNTAX_ERROR" in cls and "SYNTAX_ERROR" not in taxonomies:
                taxonomies.append("SYNTAX_ERROR")
            elif "DUPLICATE_PATCH" in cls and "DUPLICATE_PATCH" not in taxonomies:
                taxonomies.append("DUPLICATE_PATCH")
            elif "CONFLICT" in cls and "CONFLICTING_OPERATIONS" not in taxonomies:
                taxonomies.append("CONFLICTING_OPERATIONS")
                
            # If the repair was aborted due to duplicate patch logic
            if "duplicate patch" in exec_res and "DUPLICATE_PATCH" not in taxonomies:
                taxonomies.append("DUPLICATE_PATCH")

        val_report = report.get("validation_report") or {}
        if val_report.get("build_result") and not val_report["build_result"].get("success"):
            if "BUILD_FAILURE" not in taxonomies: taxonomies.append("BUILD_FAILURE")
        elif val_report.get("lint_result") and not val_report["lint_result"].get("success"):
            if "LINT_FAILURE" not in taxonomies: taxonomies.append("LINT_FAILURE")
            # Lint failures often catch syntax errors if syntax validators failed.
            if "syntax error" in str(val_report.get("lint_result")).lower() and "SYNTAX_ERROR" not in taxonomies:
                taxonomies.append("SYNTAX_ERROR")
        elif val_report.get("test_result") and not val_report["test_result"].get("success"):
            if "TEST_FAILURE" not in taxonomies: taxonomies.append("TEST_FAILURE")

        if res.rollback_triggered:
            if "ROLLBACK_TRIGGERED" not in taxonomies: taxonomies.append("ROLLBACK_TRIGGERED")
            
        # Fallback error heuristics
        if res.error_message:
            err_msg_lower = res.error_message.lower()
            if "missing expected modified files" in err_msg_lower and "MISSING_EVIDENCE" not in taxonomies:
                taxonomies.append("MISSING_EVIDENCE")
            if "context truncated" in err_msg_lower and "CONTEXT_TRUNCATION" not in taxonomies:
                taxonomies.append("CONTEXT_TRUNCATION")

        if not taxonomies:
            taxonomies.append("UNKNOWN")

        # Root Cause Precedence: PLANNING -> RETRIEVAL -> CODING -> VALIDATION -> SAFETY -> REPAIR -> UNKNOWN
        root_cause = "Unknown"
        
        planning_tax = {"PLANNER_SCHEMA_ERROR", "CONSTRAINT_EXTRACTION_FAILURE"}
        retrieval_tax = {"MISSING_EVIDENCE", "CONTEXT_TRUNCATION"}
        coding_tax = {"EMPTY_PATCH", "PATCH_VALIDATION_FAILURE", "SYNTAX_ERROR", "CONFLICTING_OPERATIONS", "DUPLICATE_PATCH"}
        validation_tax = {"BUILD_FAILURE", "LINT_FAILURE", "TEST_FAILURE"}
        safety_tax = {"CONSTRAINT_VIOLATION"}
        repair_tax = {"ROLLBACK_TRIGGERED"}
        
        tax_set = set(taxonomies)
        if tax_set.intersection(planning_tax):
            root_cause = "Planning"
        elif tax_set.intersection(retrieval_tax):
            root_cause = "Retrieval"
        elif tax_set.intersection(coding_tax):
            root_cause = "Coding"
        elif tax_set.intersection(validation_tax):
            root_cause = "Validation"
        elif tax_set.intersection(safety_tax):
            root_cause = "Safety"
        elif tax_set.intersection(repair_tax):
            root_cause = "Repair"
            
        return FailureAnalysis(
            task_id=res.task_id,
            taxonomies=taxonomies,
            root_cause=root_cause,
            explanation=f"Identified {len(taxonomies)} taxonomies via log tracing. Root cause derived via precedence matrix."
        )

    def _calculate_health_score(self, suite: BenchmarkSuiteResult, tax_counts: dict, rc_counts: dict) -> HealthScore:
        # Agent Health Score: 0-100
        # Base score from success rate
        sr_score = suite.success_rate * 100
        
        # Penalties (max 100 combined)
        rr_penalty = suite.rollback_rate * 15  # Increased penalty for rollback
        cv_penalty = min((suite.total_constraint_violations / max(suite.total_tasks, 1)) * 30, 30)
        att_penalty = min(suite.avg_repair_attempts * 5, 15)
        
        raw_metrics = {
            "success_rate": suite.success_rate,
            "rollback_rate": suite.rollback_rate,
            "constraint_violations": suite.total_constraint_violations,
            "avg_repair_attempts": suite.avg_repair_attempts
        }

        final_score = max(0, min(100, sr_score - rr_penalty - cv_penalty - att_penalty))

        return HealthScore(
            raw_metrics=raw_metrics,
            normalized_metrics={
                "success_contribution": sr_score,
                "rollback_penalty": -rr_penalty,
                "constraint_penalty": -cv_penalty,
                "attempts_penalty": -att_penalty
            },
            final_score=round(final_score, 2)
        )

    def _generate_recommendations(self, tax_counts: dict, rc_counts: dict) -> List[Recommendation]:
        recs = []
        if tax_counts.get("CONSTRAINT_EXTRACTION_FAILURE", 0) > 0:
            recs.append(Recommendation(
                issue="Constraint extraction schema errors",
                severity="HIGH",
                suggestion="Improve planner schema prompting and add few-shot examples for constraint extraction."
            ))
        if tax_counts.get("MISSING_EVIDENCE", 0) > 0:
            recs.append(Recommendation(
                issue="Files expected were not modified",
                severity="MEDIUM",
                suggestion="Increase retrieval context limit or implement intelligent context ranking."
            ))
        if tax_counts.get("EMPTY_PATCH", 0) > 0:
            recs.append(Recommendation(
                issue="Empty patches generated",
                severity="MEDIUM",
                suggestion="Provide the RepairCoder prompt with more explicit examples of handling underspecified requirements safely."
            ))
        if tax_counts.get("TEST_FAILURE", 0) > 0:
            recs.append(Recommendation(
                issue="Repeated test failures",
                severity="LOW",
                suggestion="Provide the LLM with deeper repository map types to avoid repeated logic/type check failures."
            ))
        if tax_counts.get("TIMEOUT", 0) > 0:
            recs.append(Recommendation(
                issue="Tasks are timing out",
                severity="HIGH",
                suggestion="Investigate infinite loops in LLM generation or increase max_runtime_seconds."
            ))
            
        if not recs:
            recs.append(Recommendation(
                issue="General improvements",
                severity="LOW",
                suggestion="No specific negative trends detected. Consider upgrading the LLM."
            ))
            
        return recs

    def _save_history(self, suite: BenchmarkSuiteResult, analysis: AnalysisResult):
        self.history_file_jsonl = self.history_dir / "benchmark_history.jsonl"
        
        # Backward migration for old .json to .jsonl
        if self.history_file.exists():
            import shutil
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                with open(self.history_file_jsonl, 'a', encoding='utf-8') as f_out:
                    for item in data.get("history", []):
                        f_out.write(json.dumps(item) + "\n")
                # Remove the old json file
                os.remove(self.history_file)
            except Exception as e:
                # Corrupted json! Don't swallow it.
                bak_path = str(self.history_file) + ".bak"
                shutil.copy2(self.history_file, bak_path)
                raise RuntimeError(f"Corrupted benchmark_history.json found. Backup created at {bak_path}. Aborting write.") from e

        # Ensure jsonl is not corrupted before appending
        if self.history_file_jsonl.exists():
            try:
                with open(self.history_file_jsonl, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            json.loads(line)
            except Exception as e:
                import shutil
                bak_path = str(self.history_file_jsonl) + ".bak"
                shutil.copy2(self.history_file_jsonl, bak_path)
                raise RuntimeError(f"Corrupted benchmark_history.jsonl found. Backup created at {bak_path}. Aborting write.") from e
        
        run = HistoricalRun(
            timestamp=analysis.timestamp,
            benchmark_run=analysis.benchmark_run,
            suite_name=suite.suite_name,
            final_score=analysis.health_score.final_score,
            taxonomy_counts=analysis.taxonomy_counts,
            root_cause_counts=analysis.root_cause_counts
        )
        
        with open(self.history_file_jsonl, 'a', encoding='utf-8') as f:
            f.write(run.model_dump_json() + "\n")

    def generate_markdown(self, analysis: AnalysisResult, out_path: str):
        lines = [
            f"# Phase 4.6 Improvement Analysis Report",
            f"**Agent Health Score:** {analysis.health_score.final_score}/100\n",
            f"### Top Failures"
        ]
        
        sorted_tax = sorted(analysis.taxonomy_counts.items(), key=lambda x: x[1], reverse=True)
        if not sorted_tax:
            lines.append("No failures detected.")
        else:
            total_fails = len(analysis.failure_analyses)
            for tax, count in sorted_tax:
                pct = (count / max(total_fails, 1)) * 100
                lines.append(f"- **{tax}** ({pct:.0f}%) - Count: {count}")
                
        lines.append(f"\n### Root Cause Attribution")
        for rc, count in analysis.root_cause_counts.items():
             lines.append(f"- **{rc}**: {count}")
             
        lines.append(f"\n### Architectural Recommendations")
        for rec in analysis.recommendations:
            lines.append(f"- **{rec.severity}**: {rec.issue} -> *{rec.suggestion}*")
            
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
