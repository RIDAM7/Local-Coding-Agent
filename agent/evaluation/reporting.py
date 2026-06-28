import os
import json
from pathlib import Path
from agent.evaluation.schemas import BenchmarkSuiteResult, BenchmarkComparison

class BenchmarkReporter:
    def __init__(self):
        self.results_dir = Path("benchmarks/results")
        os.makedirs(self.results_dir, exist_ok=True)

    def save_suite_result(self, suite_result: BenchmarkSuiteResult) -> str:
        json_path = self.results_dir / f"benchmark_{suite_result.suite_name}_{suite_result.timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(suite_result.model_dump_json(indent=2))
            
        md_path = self.results_dir / f"benchmark_{suite_result.suite_name}_{suite_result.timestamp}.md"
        content = self._generate_markdown(suite_result)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return str(json_path)

    def _generate_markdown(self, suite_result: BenchmarkSuiteResult) -> str:
        lines = [
            f"# Benchmark Suite: {suite_result.suite_name}",
            f"**Timestamp:** {suite_result.timestamp}",
            f"\n## Summary Metrics",
            f"- **Tasks Run:** {suite_result.total_tasks}",
            f"- **Success Rate:** {suite_result.success_rate * 100:.1f}%",
            f"- **Repair Trigger Rate:** {suite_result.repair_trigger_rate * 100:.1f}%",
            f"- **Average Repair Attempts:** {suite_result.avg_repair_attempts:.2f}",
            f"- **Rollback Rate:** {suite_result.rollback_rate * 100:.1f}%",
            f"- **Constraint Violations Prevented:** {suite_result.total_constraint_violations}",
            f"- **Average Runtime:** {suite_result.avg_execution_time:.2f}s",
            f"\n## Individual Results"
        ]
        
        for r in suite_result.results:
            icon = "✅" if r.success else "❌"
            lines.append(f"### {icon} {r.task_id}")
            lines.append(f"- Execution Time: {r.execution_time:.2f}s")
            lines.append(f"- Repair Triggered: {r.repair_triggered} (Attempts: {r.repair_attempts})")
            if r.rollback_triggered:
                lines.append(f"- Rollback Triggered: True")
            if r.constraint_violations > 0:
                lines.append(f"- Constraint Violations: {r.constraint_violations}")
            if r.error_message:
                lines.append(f"- **Error:** {r.error_message}")
            lines.append("")
            
        return "\n".join(lines)

    def compare(self, base_file: str, compare_file: str) -> BenchmarkComparison:
        with open(base_file, 'r', encoding='utf-8') as f:
            base = BenchmarkSuiteResult.model_validate_json(f.read())
        with open(compare_file, 'r', encoding='utf-8') as f:
            comp = BenchmarkSuiteResult.model_validate_json(f.read())
            
        return BenchmarkComparison(
            base_run=base.timestamp,
            compare_run=comp.timestamp,
            delta_success_rate=comp.success_rate - base.success_rate,
            delta_repair_trigger_rate=comp.repair_trigger_rate - base.repair_trigger_rate,
            delta_avg_repair_attempts=comp.avg_repair_attempts - base.avg_repair_attempts,
            delta_rollback_rate=comp.rollback_rate - base.rollback_rate,
            delta_total_constraint_violations=comp.total_constraint_violations - base.total_constraint_violations,
            delta_avg_execution_time=comp.avg_execution_time - base.avg_execution_time
        )
