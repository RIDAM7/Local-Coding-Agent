import sys
import json
import asyncio
from pathlib import Path

from agent.evaluation.schemas import BenchmarkTask
from agent.evaluation.runner import BenchmarkRunner

async def run_reflection_analysis(suite_path: str):
    suite_path = Path(suite_path)
    with open(suite_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    tasks = [BenchmarkTask(**t) for t in data.get("tasks", [])]
    suite_name = data.get("name", "Unknown Suite")
    
    print(f"--- Running Reflection Evaluation on suite: {suite_name} ({len(tasks)} tasks) ---")
    
    # Pass A: Reflection OFF + Shadow Reflection ON
    print(">>> Pass A: Reflection OFF + Shadow Reflection ON")
    runner_a = BenchmarkRunner(f"{suite_name}_shadow", tasks, reflection_enabled=False, shadow_reflection=True)
    res_a = await runner_a.run_suite()
    
    # Pass B: Reflection ON
    print(">>> Pass B: Reflection ON")
    runner_b = BenchmarkRunner(f"{suite_name}_active", tasks, reflection_enabled=True, shadow_reflection=False)
    res_b = await runner_b.run_suite()
    
    # Analyze Accuracy (using Pass A)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    cat_stats = {}
    
    for r in res_a.results:
        val_failed = r.validation_failures > 0
        ref_result = r.reflection_result
        
        # Confusion matrix
        if ref_result == "FAIL":
            if val_failed:
                true_positive += 1
            else:
                false_positive += 1
        else: # PASS or WARNING
            if val_failed:
                false_negative += 1
            else:
                true_negative += 1
                
        # Category analysis
        report_path = r.report_path
        if report_path and Path(report_path).exists():
            with open(report_path, 'r', encoding='utf-8') as rf:
                full_report = json.load(rf)
                
            ref_report = full_report.get("reflection_report") or {}
            critiques = ref_report.get("critiques", [])
            categories = [c.get("category", c) for c in critiques]
            
            if ref_result == "FAIL":
                for c in categories:
                    if c not in cat_stats:
                        cat_stats[c] = {"predictions": 0, "correct": 0}
                    cat_stats[c]["predictions"] += 1
                    if val_failed:
                        cat_stats[c]["correct"] += 1

    # Calculate metrics safely
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Retry Effectiveness Tracking
    retries_generated = 0
    retries_improved = 0
    retries_worsened = 0
    
    for i, r_b in enumerate(res_b.results):
        r_a = res_a.results[i]
        
        if r_b.reflection_retry_used:
            retries_generated += 1
            b_val_failed = r_b.validation_failures > 0
            a_val_failed = r_a.validation_failures > 0
            
            if a_val_failed and not b_val_failed:
                retries_improved += 1
            elif not a_val_failed and b_val_failed:
                retries_worsened += 1
            elif b_val_failed and r_b.validation_failures > r_a.validation_failures:
                retries_worsened += 1

    # Category Accuracy formatting
    category_analysis = {}
    for cat, stats in cat_stats.items():
        preds = stats["predictions"]
        correct = stats["correct"]
        acc = correct / preds if preds > 0 else 0.0
        status = "UNRELIABLE" if acc < 0.50 else "RELIABLE"
        category_analysis[cat] = {
            "predictions": preds,
            "correct": correct,
            "accuracy": round(acc, 2),
            "status": status
        }
        
    # Cost metrics
    runtime_a = sum([r.execution_time for r in res_a.results])
    runtime_b = sum([r.execution_time for r in res_b.results])
    runtime_overhead_percent = ((runtime_b - runtime_a) / runtime_a * 100) if runtime_a > 0 else 0.0

    output_data = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "runtime_overhead_percent": round(runtime_overhead_percent, 2),
        "retries_generated": retries_generated,
        "retries_improved": retries_improved,
        "retries_worsened": retries_worsened,
        "category_analysis": category_analysis
    }
    
    out_path = Path("benchmarks/results/reflection_accuracy.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    # Output to CLI
    print("\n=============================================")
    print("        REFLECTION ACCURACY METRICS        ")
    print("=============================================")
    print(f"True Positive:  {true_positive}")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    print(f"True Negative:  {true_negative}")
    print("---------------------------------------------")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1_score:.2f}")
    print("=============================================")
    print(f"Runtime Overhead: {runtime_overhead_percent:.2f}%")
    print("=============================================")
    print("        RETRY EFFECTIVENESS                ")
    print(f"Retries Generated: {retries_generated}")
    print(f"Retries Improved:  {retries_improved}")
    print(f"Retries Worsened:  {retries_worsened}")
    print("=============================================")
    print("        CATEGORY ACCURACY                  ")
    for cat, data in category_analysis.items():
        print(f"{cat}: Predictions={data['predictions']}, Correct={data['correct']}, Acc={data['accuracy']:.2f} [{data['status']}]")
        if data["status"] == "UNRELIABLE":
            print(f"  --> WARNING: Category {cat} is unreliable!")
    print("=============================================")
    print(f"Saved accuracy report to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agent.evaluation.reflection_benchmark <suite_json_path>")
        sys.exit(1)
    asyncio.run(run_reflection_analysis(sys.argv[1]))
