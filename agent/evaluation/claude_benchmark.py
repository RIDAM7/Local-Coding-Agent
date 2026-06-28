import sys
import json
import asyncio
from pathlib import Path
import math

from agent.evaluation.schemas import BenchmarkTask
from agent.evaluation.runner import BenchmarkRunner

async def run_claude_analysis(suite_path: str):
    suite_path = Path(suite_path)
    with open(suite_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    tasks = [BenchmarkTask(**t) for t in data.get("tasks", [])]
    suite_name = data.get("name", "Unknown Suite")
    
    print(f"--- Running Claude Review Effectiveness Benchmark on suite: {suite_name} ({len(tasks)} tasks) ---")
    
    # Mode A: Claude OFF
    print(">>> Mode A: Claude OFF")
    runner_a = BenchmarkRunner(f"{suite_name}_mode_a", tasks, claude_enabled=False, claude_always_on=False)
    res_a = await runner_a.run_suite()
    
    # Mode B: Claude ON (Normal Routing)
    print(">>> Mode B: Claude ON (Normal Routing)")
    runner_b = BenchmarkRunner(f"{suite_name}_mode_b", tasks, claude_enabled=True, claude_always_on=False)
    res_b = await runner_b.run_suite()
    
    # Mode C: Claude ALWAYS ON
    print(">>> Mode C: Claude ALWAYS ON")
    runner_c = BenchmarkRunner(f"{suite_name}_mode_c", tasks, claude_enabled=True, claude_always_on=True)
    res_c = await runner_c.run_suite()
    
    # Helpers for metrics
    def calc_metrics(res):
        total = len(res.results) if res and res.results else 1
        success_rate = res.success_rate if res else 0.0
        val_failures = sum([r.validation_failures for r in res.results]) if res else 0
        repair_attempts = sum([r.repair_attempts for r in res.results]) / total if res else 0.0
        rollback_rate = res.rollback_rate if res else 0.0
        runtime_ms = sum([r.execution_time for r in res.results]) * 1000 if res else 0.0
        return success_rate, val_failures, repair_attempts, rollback_rate, runtime_ms, total

    a_succ, a_val_fail, a_rep, a_roll, a_time, a_total = calc_metrics(res_a)
    b_succ, b_val_fail, b_rep, b_roll, b_time, b_total = calc_metrics(res_b)
    c_succ, c_val_fail, c_rep, c_roll, c_time, c_total = calc_metrics(res_c)
    
    # Runtimes
    all_runtimes = []
    if res_a: all_runtimes.extend([r.execution_time * 1000 for r in res_a.results])
    if res_b: all_runtimes.extend([r.execution_time * 1000 for r in res_b.results])
    if res_c: all_runtimes.extend([r.execution_time * 1000 for r in res_c.results])
    
    all_runtimes.sort()
    average_runtime_ms = sum(all_runtimes) / len(all_runtimes) if all_runtimes else 0.0
    p95_runtime_ms = all_runtimes[int(math.ceil(len(all_runtimes) * 0.95)) - 1] if all_runtimes else 0.0

    # Claude-specific metrics
    reviews_triggered = 0
    mandatory_reviews = 0
    reviews_with_issues = 0
    reviews_triggering_repair = 0
    issues_detected = 0
    tokens_in = 0
    tokens_out = 0
    claude_latency = 0.0
    cost_usd = 0.0
    
    # Repair Effectiveness Analysis
    improved = 0
    unchanged = 0
    worsened = 0
    
    # Mode B logic for repair effectiveness and general metrics
    if res_b and res_a:
        for i, r_b in enumerate(res_b.results):
            r_a = res_a.results[i] if i < len(res_a.results) else None
            
            report_path = r_b.report_path
            if report_path and Path(report_path).exists():
                with open(report_path, 'r', encoding='utf-8') as rf:
                    full_report = json.load(rf)
                
                ext_report = full_report.get("external_review_report")
                if ext_report:
                    if ext_report.get("review_status") != "BYPASSED":
                        reviews_triggered += 1
                        
                    issues = ext_report.get("issues", [])
                    has_issues = len(issues) > 0
                    if has_issues:
                        reviews_with_issues += 1
                        
                    issues_detected += len(issues)
                    tokens_in += ext_report.get("tokens_sent", 0)
                    tokens_out += ext_report.get("tokens_received", 0)
                    cost_usd += ext_report.get("estimated_cost", 0.0)
                    claude_latency += ext_report.get("latency_ms", 0.0)
                    
                    decision = full_report.get("review_decision")
                    if decision == "MANDATORY_REVIEW":
                        mandatory_reviews += 1
                        if has_issues:
                            reviews_triggering_repair += 1
                            
                    # Repair Effectiveness
                    if r_a and decision == "MANDATORY_REVIEW" and has_issues:
                        a_failed = r_a.validation_failures > 0
                        b_failed = r_b.validation_failures > 0
                        
                        if a_failed and not b_failed:
                            improved += 1
                        elif (a_failed and b_failed) or (not a_failed and not b_failed):
                            unchanged += 1
                        elif not a_failed and b_failed:
                            worsened += 1

    # Confusion matrix built from Mode C against Mode A Validation
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    if res_c and res_a:
        for i, r_c in enumerate(res_c.results):
            r_a = res_a.results[i] if i < len(res_a.results) else None
            
            report_path = r_c.report_path
            if report_path and Path(report_path).exists():
                with open(report_path, 'r', encoding='utf-8') as rf:
                    full_report = json.load(rf)
                    
                ext_report = full_report.get("external_review_report")
                if ext_report and ext_report.get("review_status") != "BYPASSED":
                    issues = ext_report.get("issues", [])
                    has_issues = len(issues) > 0
                    
                    if r_a:
                        a_failed = r_a.validation_failures > 0
                        if has_issues:
                            if a_failed:
                                true_positive += 1
                            else:
                                false_positive += 1
                        else:
                            if a_failed:
                                false_negative += 1
                            else:
                                true_negative += 1
                                
    # Accuracy formulas
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Cost Efficiency
    cost_per_improvement = cost_usd / max(1, improved)
    
    # Rate calculations
    total_tasks = b_total
    total_reviews = reviews_triggered
    
    review_trigger_rate = reviews_triggered / max(1, total_tasks)
    mandatory_review_rate = mandatory_reviews / max(1, total_tasks)
    issue_detection_rate = reviews_with_issues / max(1, total_reviews)
    actionable_issue_rate = reviews_triggering_repair / max(1, total_reviews)
    
    # Build effectiveness.json
    eff_data = {
        "success_rate_delta": b_succ - a_succ,
        "repair_attempt_delta": b_rep - a_rep,
        "rollback_delta": b_roll - a_roll,
        
        "average_runtime_ms": average_runtime_ms,
        "p95_runtime_ms": p95_runtime_ms,
        
        "reviews_triggered": reviews_triggered,
        "review_trigger_rate": review_trigger_rate,
        "mandatory_review_rate": mandatory_review_rate,
        "issue_detection_rate": issue_detection_rate,
        "actionable_issue_rate": actionable_issue_rate,
        
        "IMPROVED": improved,
        "UNCHANGED": unchanged,
        "WORSENED": worsened,
        
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "TN": true_negative,
        
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        
        "estimated_cost_usd": cost_usd,
        "cost_per_improvement": cost_per_improvement,
        
        "recommended_threshold": 95 if cost_per_improvement < 10.0 and b_succ >= a_succ else 99
    }
    
    out_eff = Path("benchmarks/results/claude_effectiveness.json")
    out_eff.parent.mkdir(exist_ok=True, parents=True)
    with open(out_eff, 'w', encoding='utf-8') as f:
        json.dump(eff_data, f, indent=2)

    # Threshold Optimization Analysis
    # We simulate thresholds using Mode C's detailed reports where Claude runs on everything.
    thresholds = [95, 90, 85, 80]
    thresh_data = {}
    
    if res_c:
        for t in thresholds:
            t_calls = 0
            t_cost = 0.0
            t_helpful = 0
            
            for i, r_c in enumerate(res_c.results):
                report_path = r_c.report_path
                if not report_path or not Path(report_path).exists(): continue
                with open(report_path, 'r', encoding='utf-8') as rf:
                    full_report = json.load(rf)
                    
                conf_report = full_report.get("confidence_report", {})
                score = conf_report.get("confidence_score", 100)
                ext_report = full_report.get("external_review_report", {})
                
                if score < t:
                    t_calls += 1
                    t_cost += ext_report.get("estimated_cost", 0.0)
                    
                    # Assume helpful if there were issues and Mode A failed
                    issues = ext_report.get("issues", [])
                    r_a = res_a.results[i] if res_a and i < len(res_a.results) else None
                    if len(issues) > 0 and r_a and r_a.validation_failures > 0:
                        t_helpful += 1
                        
            thresh_data[str(t)] = {
                "success_rate": b_succ, # Placeholder estimate
                "review_rate": t_calls / c_total if c_total > 0 else 0,
                "claude_calls": t_calls,
                "estimated_cost": t_cost,
                "cost_per_successful_improvement": t_cost / max(1, t_helpful)
            }
            
    out_thresh = Path("benchmarks/results/claude_threshold_analysis.json")
    with open(out_thresh, 'w', encoding='utf-8') as f:
        json.dump(thresh_data, f, indent=2)
        
    print("\n=============================================")
    print("      CLAUDE EFFECTIVENESS METRICS           ")
    print("=============================================")
    print(f"Success Rate Delta:      {eff_data['success_rate_delta']*100:+.1f}%")
    print(f"Repair Attempt Delta:    {eff_data['repair_attempt_delta']:+.2f}")
    print(f"Rollback Rate Delta:     {eff_data['rollback_delta']*100:+.1f}%")
    print("---------------------------------------------")
    print(f"Reviews Triggered:       {reviews_triggered} ({review_trigger_rate*100:.1f}%)")
    print(f"Mandatory Review Rate:   {mandatory_review_rate*100:.1f}%")
    print(f"Issue Detection Rate:    {issue_detection_rate*100:.1f}%")
    print(f"Actionable Issue Rate:   {actionable_issue_rate*100:.1f}%")
    print("---------------------------------------------")
    print("Repair Effectiveness:")
    print(f"  IMPROVED:              {improved}")
    print(f"  UNCHANGED:             {unchanged}")
    print(f"  WORSENED:              {worsened}")
    print("---------------------------------------------")
    print(f"Estimated Cost (USD):    ${cost_usd:.4f}")
    print(f"Cost per Improvement:    ${cost_per_improvement:.4f}")
    print(f"Tokens Sent:             {tokens_in}")
    print(f"Tokens Received:         {tokens_out}")
    print(f"Avg Latency (ms):        {claude_latency / max(1, reviews_triggered):.1f}")
    print(f"Average Runtime (ms):    {average_runtime_ms:.1f}")
    print(f"P95 Runtime (ms):        {p95_runtime_ms:.1f}")
    print("=============================================")
    print("      CLAUDE ACCURACY METRICS                ")
    print("=============================================")
    print(f"True Positives (TP):     {true_positive}")
    print(f"False Positives (FP):    {false_positive}")
    print(f"False Negatives (FN):    {false_negative}")
    print(f"True Negatives (TN):     {true_negative}")
    print("---------------------------------------------")
    print(f"Precision:               {precision:.2f}")
    print(f"Recall:                  {recall:.2f}")
    print(f"F1 Score:                {f1_score:.2f}")
    print("=============================================")
    print("      THRESHOLD ANALYSIS                     ")
    for t, d in thresh_data.items():
        print(f"Threshold {t}: Calls={d['claude_calls']}, Cost=${d['estimated_cost']:.4f}, CPI=${d['cost_per_successful_improvement']:.4f}")
    print("=============================================")
    print(f"Recommended Threshold: {eff_data['recommended_threshold']}")
    print(f"Reports saved to benchmarks/results/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agent.evaluation.claude_benchmark <suite_json_path>")
        sys.exit(1)
    asyncio.run(run_claude_analysis(sys.argv[1]))
