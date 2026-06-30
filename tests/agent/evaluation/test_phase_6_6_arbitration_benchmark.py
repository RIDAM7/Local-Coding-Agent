import pytest
import asyncio
import json
from pathlib import Path
from agent.evaluation.arbitration_benchmark import ArbitrationBenchmarkRunner
from agent.evaluation.arbitration_schemas import ArbitrationEffectivenessMetrics

@pytest.fixture
def runner(tmp_path):
    reports_dir = tmp_path / "reports"
    return ArbitrationBenchmarkRunner(reports_dir=str(reports_dir))

def test_benchmark_runner_full_suite(runner):
    # Tests the mock suite logic and that it parses successfully
    suite_path = "tests/suites/arbitration_mock_suite.json"
    
    metrics = asyncio.run(runner.run_benchmark(suite_path))
    
    assert isinstance(metrics, ArbitrationEffectivenessMetrics)
    assert metrics.total_tasks == 7
    
    # 1. Decision distributions sum correctly (7 tasks total, 2 fail_closed -> 5 decisions)
    dist = metrics.decision_distribution
    assert sum(dist.values()) + len(metrics.tasks_fail_closed) == 7
    
    # 2. Coverage detection works
    assert metrics.coverage.all_states_observed is True
    assert len(metrics.coverage.missing_states) == 0
    
    # 3. Authority checks work
    assert metrics.authority.validation_remains_authoritative is True
    assert metrics.authority.claude_never_merges_failing_code is True
    assert metrics.authority.claude_never_blocks_validated_code is True
    
    # 4. Traceability lists populate correctly
    assert "task_02_claude_fp" in metrics.tasks_claude_overridden
    assert "task_05_reflect_override" in metrics.tasks_reflection_overridden
    assert "task_03_claude_fn" in metrics.tasks_claude_false_approval
    assert "task_06_budget_limit" in metrics.tasks_fail_closed
    assert "task_07_payload_limit" in metrics.tasks_fail_closed
    
    # 5. JSON artifact generation succeeds
    output_path = Path(runner.reports_dir) / "arbitration_effectiveness.json"
    assert output_path.exists()
    
    with open(output_path, "r") as f:
        data = json.load(f)
        assert data["total_tasks"] == 7
        assert data["coverage"]["all_states_observed"] is True
