import json
import glob
from agent.models.schemas import Report

reports = []
paths = glob.glob(r'C:\Users\ridam\AppData\Local\Temp\pytest-of-ridam\pytest-26\test_benchmark_runner_full_sui0\reports\*.json')
for p in paths:
    try:
        with open(p, 'r') as f:
            d = json.load(f)
        reports.append(Report(**d))
    except Exception as e:
        print(f"Error reading {p}: {e}")

print([r.arbitration_report.decision.value if r.arbitration_report else None for r in reports])
