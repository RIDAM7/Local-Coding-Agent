import asyncio
import logging
logging.basicConfig(level=logging.ERROR)

from agent.evaluation.arbitration_benchmark import ArbitrationBenchmarkRunner

async def main():
    runner = ArbitrationBenchmarkRunner('workspace', 'reports')
    metrics = await runner.run_benchmark('tests/suites/arbitration_mock_suite.json')
    print("Decision distribution:", metrics.decision_distribution)
    print("Total reports checked?", len(metrics.tasks_fail_closed) + sum(metrics.decision_distribution.values()))
    
if __name__ == '__main__':
    asyncio.run(main())
