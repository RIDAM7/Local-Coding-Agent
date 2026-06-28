import json
from agent.evaluation.reporting import BenchmarkReporter
from agent.evaluation.schemas import BenchmarkSuiteResult

with open('benchmarks/results/benchmark_core_suite_20260606_070000.json') as f:
    res = BenchmarkSuiteResult.model_validate_json(f.read())
    
rep = BenchmarkReporter()
rep.save_suite_result(res)
print("MD Generated")
