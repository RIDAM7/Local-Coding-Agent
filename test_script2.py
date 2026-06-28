import asyncio
from agent.evaluation.arbitration_benchmark import ArbitrationBenchmarkRunner

async def main():
    runner = ArbitrationBenchmarkRunner('workspace', 'reports')
    metrics = await runner.run_benchmark('tests/suites/arbitration_mock_suite.json')
    print("Decision distribution:", metrics.decision_distribution)
    print("Total reports checked?", len(metrics.tasks_fail_closed) + sum(metrics.decision_distribution.values()))
    
asyncio.run(main())
