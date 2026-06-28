import argparse
import asyncio
import json
import os
from pathlib import Path
from agent.evaluation.schemas import BenchmarkTask
from agent.evaluation.runner import BenchmarkRunner
from agent.evaluation.reporting import BenchmarkReporter

async def main():
    parser = argparse.ArgumentParser(description="Phase 4.5 Evaluation Harness")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark suite")
    run_parser.add_argument("suite", help="Path to the suite JSON file")

    # Compare command
    comp_parser = subparsers.add_parser("compare", help="Compare two benchmark runs")
    comp_parser.add_argument("run_a", help="Path to first run JSON")
    comp_parser.add_argument("run_b", help="Path to second run JSON")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze benchmark results (Phase 4.6)")
    analyze_parser.add_argument("result", help="Path to result JSON or 'latest'")

    # Replay commands
    replay_list_parser = subparsers.add_parser("replay-list", help="List stored replay artifacts (Phase 4.7A/B)")
    
    replay_parser = subparsers.add_parser("replay", help="Replay a benchmark failure (Phase 4.7A/B)")
    replay_parser.add_argument("replay_id", help="ID of the replay artifact to run")
    
    replay_promote_parser = subparsers.add_parser("replay-promote", help="Promote a replay to GOLDEN tier (Phase 4.7B)")
    replay_promote_parser.add_argument("replay_id", help="ID of the replay artifact to promote")
    
    replay_suite_parser = subparsers.add_parser("replay-suite", help="Execute a suite of replays (Phase 4.7B)")
    replay_suite_parser.add_argument("--tier", choices=["GOLDEN", "REGRESSION", "NORMAL"], default="NORMAL", help="Tier to run")
    
    replay_capture_parser = subparsers.add_parser("replay-capture", help="Manually capture a replay artifact (Phase 4.7B)")
    replay_capture_parser.add_argument("--task", required=True, help="Task description")
    replay_capture_parser.add_argument("--workspace", required=True, help="Path to the workspace to snapshot")

    oracle_add_parser = subparsers.add_parser("oracle-add", help="Add Oracle Solution to ReplayArtifact (Phase 4.7C)")
    oracle_add_parser.add_argument("--replay-id", required=True, help="ID of the replay artifact")
    oracle_add_parser.add_argument("--source", required=True, choices=["HUMAN", "GPT4", "CLAUDE", "CODEX"])
    oracle_add_parser.add_argument("--patch-file", required=True, help="Path to JSON file containing patch")

    export_dataset_parser = subparsers.add_parser("export-dataset", help="Export datasets (Phase 4.7C)")
    export_dataset_parser.add_argument("--split", action='append', choices=["TRAIN", "VALIDATION"], help="Splits to export")
    export_dataset_parser.add_argument("--min-quality", type=float, default=75.0, help="Minimum quality score")
    export_dataset_parser.add_argument("--version", required=True, help="Dataset version")

    verify_dataset_parser = subparsers.add_parser("verify-dataset", help="Verify dataset integrity (Phase 4.7C)")
    verify_dataset_parser.add_argument("--manifest", required=True, help="Path to dataset_manifest.json")

    memory_bench_parser = subparsers.add_parser("memory-benchmark", help="Run Memory Benefit Benchmark (Phase 5.0)")
    memory_bench_parser.add_argument("--replay-id", required=True, help="Replay to benchmark")

    reflection_bench_parser = subparsers.add_parser("reflection-analysis", help="Run Reflection Analysis Benchmark (Phase 6.1)")
    reflection_bench_parser.add_argument("suite", help="Path to the suite JSON file")

    arbitration_bench_parser = subparsers.add_parser("arbitration-analysis", help="Run Arbitration Effectiveness Benchmark (Phase 6.6)")
    arbitration_bench_parser.add_argument("suite", help="Path to the suite JSON file")

    claude_bench_parser = subparsers.add_parser("claude-analysis", help="Run Claude Effectiveness Benchmark (Phase 6.3)")
    claude_bench_parser.add_argument("suite", help="Path to the suite JSON file")

    args = parser.parse_args()

    if args.command == "run":
        if not os.path.exists(args.suite):
            print(f"Error: Suite file {args.suite} not found.")
            return

        with open(args.suite, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tasks = [BenchmarkTask(**t) for t in data.get("tasks", [])]
        suite_name = data.get("name", Path(args.suite).stem)
        
        print(f"Loaded suite '{suite_name}' with {len(tasks)} tasks.")
        
        runner = BenchmarkRunner(suite_name, tasks)
        suite_result = await runner.run_suite()
        
        if suite_result:
            reporter = BenchmarkReporter()
            json_path = reporter.save_suite_result(suite_result)
            print(f"\nSuite complete. Results saved to: {json_path}")
            
    elif args.command == "compare":
        reporter = BenchmarkReporter()
        comp = reporter.compare(args.run_a, args.run_b)
        
        print(f"\n# Benchmark Comparison")
        print(f"Base: {comp.base_run} | Compare: {comp.compare_run}\n")
        print(f"- Success Rate Delta: {comp.delta_success_rate * 100:+.1f}%")
        print(f"- Repair Trigger Rate Delta: {comp.delta_repair_trigger_rate * 100:+.1f}%")
        print(f"- Avg Repair Attempts Delta: {comp.delta_avg_repair_attempts:+.2f}")
        print(f"- Rollback Rate Delta: {comp.delta_rollback_rate * 100:+.1f}%")
        print(f"- Constraint Violations Delta: {comp.delta_total_constraint_violations:+d}")
        print(f"- Avg Execution Time Delta: {comp.delta_avg_execution_time:+.2f}s")
        
        if comp.delta_success_rate <= -0.05:
            print("\n🚨 REGRESSION DETECTED: Success rate dropped by > 5%. Failing build.")
            import sys
            sys.exit(1)
            
    elif args.command == "analyze":
        from agent.evaluation.analyzer import FailureAnalyzer
        from pydantic import ValidationError
        
        target = args.result
        if target == "latest":
            results_dir = Path("benchmarks/results")
            files = list(results_dir.glob("benchmark_*.json"))
            if not files:
                print("Error: No benchmark results found in benchmarks/results.")
                return
            target = max(files, key=os.path.getmtime)
            print(f"Analyzing latest result: {target}")
            
        if not os.path.exists(target):
            print(f"Error: Result file '{target}' not found.")
            return
            
        analyzer = FailureAnalyzer()
        try:
            analysis = analyzer.analyze(target)
            
            out_json = Path(target).parent / f"analysis_{Path(target).name}"
            out_md = Path(target).parent / f"improvement_report_{Path(target).stem}.md"
            
            with open(out_json, 'w', encoding='utf-8') as f:
                f.write(analysis.model_dump_json(indent=2))
                
            analyzer.generate_markdown(analysis, str(out_md))
            print(f"\nAnalysis complete!")
            print(f"Health Score: {analysis.health_score.final_score}/100")
            print(f"JSON saved to: {out_json}")
            print(f"Report saved to: {out_md}")
            
        except ValidationError as e:
            print(f"\nError: The benchmark result file '{target}' is malformed or invalid.")
            print(f"Pydantic Validation Error Details:\n{e}")
        except RuntimeError as e:
            print(f"\nError: Runtime constraint failure during analysis.")
            print(f"Details: {e}")
        except Exception as e:
            print(f"\nError: An unexpected error occurred during analysis.")
            print(f"Details: {str(e)}")
            
    elif args.command == "reflection-analysis":
        from agent.evaluation.reflection_benchmark import ReflectionBenchmarkRunner
        runner = ReflectionBenchmarkRunner()
        asyncio.run(runner.run_benchmark(args.suite))
        
    elif args.command == "arbitration-analysis":
        from agent.evaluation.arbitration_benchmark import ArbitrationBenchmarkRunner
        runner = ArbitrationBenchmarkRunner()
        await runner.run_benchmark(args.suite)
            
    elif args.command == "replay-list":
        from agent.evaluation.replay_manager import ReplayManager
        mgr = ReplayManager()
        replays = mgr.list_replays()
        if not replays:
            print("No replay artifacts found.")
            return
        print(f"{'REPLAY ID':<40} | {'TIER':<10} | {'SCORE':<6} | {'BENCHMARK ID':<30} | {'STATUS'}")
        print("-" * 110)
        for r in replays:
            # Safe fallback for artifacts created in Phase 4.7A before fields existed
            tier = getattr(r, 'tier', 'N/A')
            diff = getattr(r, 'difficulty_score', 0.0)
            print(f"{r.replay_id:<40} | {str(tier.value) if hasattr(tier, 'value') else tier:<10} | {diff:<6.1f} | {r.benchmark_id:<30} | {r.final_status}")
            
    elif args.command == "replay-promote":
        from agent.evaluation.replay_manager import ReplayManager
        mgr = ReplayManager()
        if mgr.promote_replay(args.replay_id):
            print(f"Successfully promoted {args.replay_id} to GOLDEN tier.")
        else:
            print(f"Error: Replay artifact {args.replay_id} not found.")

    elif args.command == "replay-suite":
        from agent.evaluation.replay_manager import ReplayManager
        from agent.orchestrator import Orchestrator
        from agent.evaluation.replay_schemas import ProvenanceType, ReplayTier
        import tempfile
        import time
        
        mgr = ReplayManager()
        replays = mgr.list_replays()
        target_replays = [r for r in replays if getattr(r, 'tier', 'NORMAL') == args.tier or getattr(getattr(r, 'tier', None), 'value', '') == args.tier]
        
        if not target_replays:
            print(f"No replays found in tier: {args.tier}")
            return
            
        print(f"Executing {len(target_replays)} replays from tier {args.tier}...")
        
        successes = 0
        total_time = 0.0
        
        for r in target_replays:
            print(f"\n[{r.replay_id}] Starting...")
            start_time = time.time()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_workspace = Path(temp_dir) / "workspace"
                if not mgr.extract_and_verify(r, temp_workspace):
                    print(f"[{r.replay_id}] FAILED (Archive Error)")
                    continue
                    
                orchestrator = Orchestrator(workspace_path=temp_workspace)
                try:
                    report_path = await orchestrator.run(r.task)
                    
                    json_path = report_path.replace('.md', '.json')
                    with open(json_path, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                        
                    replay_status = report_data.get("final_status", "FAILURE")
                except Exception as e:
                    print(f"[{r.replay_id}] Error during execution: {e}")
                    replay_status = "FAILURE"
                    report_data = {"final_status": "FAILURE"}
                    
                duration = time.time() - start_time
                total_time += duration
                
                print(f"[{r.replay_id}] Status: {replay_status} ({duration:.1f}s)")
                
                if replay_status == "SUCCESS":
                    successes += 1
                    mgr.update_evolution_on_success(r.replay_id)
                elif replay_status == "FAILURE":
                    if r.final_status == "SUCCESS" or True: # Capture regression on any failure
                        mgr.capture_failure(
                            task_id=r.benchmark_id,
                            task_desc=r.task,
                            report_data=report_data,
                            workspace_path=temp_workspace,
                            provenance=ProvenanceType.REGRESSION,
                            tier=ReplayTier.REGRESSION
                        )
                        print("Regression captured.")
                    
        success_rate = (successes / len(target_replays)) * 100 if target_replays else 0
        avg_runtime = total_time / len(target_replays) if target_replays else 0
        print("\n# Replay Suite Execution Summary")
        print(f"Total: {len(target_replays)}")
        print(f"Success: {successes}")
        print(f"Failed: {len(target_replays) - successes}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Avg Runtime: {avg_runtime:.2f}s")
            
    elif args.command == "replay":
        from agent.evaluation.replay_manager import ReplayManager
        from agent.orchestrator import Orchestrator
        from agent.evaluation.replay_schemas import ProvenanceType, ReplayTier
        import tempfile
        
        mgr = ReplayManager()
        replays = mgr.list_replays()
        artifact = next((r for r in replays if r.replay_id == args.replay_id), None)
        if not artifact:
            print(f"Error: Replay artifact {args.replay_id} not found.")
            return
            
        print(f"Replaying {artifact.replay_id}")
        prov = getattr(artifact, 'provenance', 'UNKNOWN')
        prov_val = prov.value if hasattr(prov, 'value') else prov
        tier = getattr(artifact, 'tier', 'UNKNOWN')
        tier_val = tier.value if hasattr(tier, 'value') else tier
        print(f"Provenance: {prov_val} | Tier: {tier_val} | Difficulty: {getattr(artifact, 'difficulty_score', 0.0):.1f}")
        print(f"Original Status: {artifact.final_status}")
        
        if hasattr(artifact, 'environment') and artifact.environment:
            env = artifact.environment
            print("\nEnvironment Snapshot:")
            print(f"- OS: {getattr(env, 'os', getattr(env, 'os_info', 'Unknown'))}")
            print(f"- Python: {env.python_version}")
            models = getattr(env, 'ollama_models', []) or list(getattr(env, 'configured_models', {}).values())
            print(f"- Models: {', '.join(models)}")
            if getattr(env, 'target_git_commit', 'unknown') != 'unknown':
                print(f"- Git Commit: {env.target_git_commit}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_workspace = Path(temp_dir) / "workspace"
            if not mgr.extract_and_verify(artifact, temp_workspace):
                print("Error: Archive verification or extraction failed. Aborting replay.")
                return
                
            orchestrator = Orchestrator(workspace_path=temp_workspace)
            try:
                report_path = await orchestrator.run(artifact.task)
                
                json_path = report_path.replace('.md', '.json')
                with open(json_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    
                replay_status = report_data.get("final_status", "FAILURE")
            except Exception as e:
                print(f"Error during replay execution: {e}")
                replay_status = "FAILURE"
                report_data = {"final_status": "FAILURE"}
                
            print(f"\nOriginal Status: {artifact.final_status}")
            print(f"Replay Status: {replay_status}")
            
            if artifact.final_status == replay_status:
                print("Result: MATCH")
            else:
                print("Result: MISMATCH")
                
            if replay_status == "SUCCESS":
                mgr.update_evolution_on_success(artifact.replay_id)
                print("Evolution tracking updated with first success date.")
            else:
                mgr.capture_failure(
                    task_id=artifact.benchmark_id,
                    task_desc=artifact.task,
                    report_data=report_data,
                    workspace_path=temp_workspace,
                    provenance=ProvenanceType.REGRESSION,
                    tier=ReplayTier.REGRESSION
                )
                print("Regression captured.")

    elif args.command == "replay-capture":
        from agent.evaluation.replay_manager import ReplayManager
        from agent.evaluation.replay_schemas import ProvenanceType, ReplayTier
        
        mgr = ReplayManager()
        ws_path = Path(args.workspace)
        if not ws_path.exists() or not ws_path.is_dir():
            print(f"Error: Workspace path '{args.workspace}' is invalid.")
            return
            
        report_data = {
            "final_status": "FAILURE",
            "plan": {},
            "repair_history": [],
            "validation_report": {},
            "files_modified": []
        }
        
        artifact = mgr.capture_failure(
            task_id="manual",
            task_desc=args.task,
            report_data=report_data,
            workspace_path=ws_path,
            provenance=ProvenanceType.MANUAL_TEST,
            tier=ReplayTier.NORMAL
        )
        print(f"Successfully captured MANUAL_TEST replay: {artifact.replay_id}")

    elif args.command == "oracle-add":
        from agent.evaluation.replay_manager import ReplayManager
        from agent.evaluation.replay_schemas import OracleSolution
        from datetime import datetime
        
        mgr = ReplayManager()
        
        with open(args.patch_file, 'r', encoding='utf-8') as f:
            patch_data = json.load(f)
            
        oracle = OracleSolution(
            source=args.source,
            patch=patch_data,
            verified_success=True,
            timestamp=datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        )
        
        if mgr.add_oracle_solution(args.replay_id, oracle):
            print(f"Successfully added {args.source} OracleSolution to {args.replay_id}")
        else:
            print(f"Error: Replay {args.replay_id} not found.")

    elif args.command == "export-dataset":
        from agent.evaluation.replay_manager import ReplayManager
        from agent.evaluation.replay_schemas import ProvenanceType, ReplayTier, DatasetManifest
        import hashlib
        from datetime import datetime
        
        mgr = ReplayManager()
        replays = mgr.list_replays()
        
        out_dir = Path("datasets")
        out_dir.mkdir(exist_ok=True)
        
        splits_requested = args.split if args.split else ["TRAIN", "VALIDATION"]
        
        train_data = []
        val_data = []
        ids_included = []
        
        for r in replays:
            # 7. Leakage Protection
            # Benchmark-derived artifacts must never enter TRAIN/VALIDATION
            if r.provenance == ProvenanceType.BENCHMARK:
                continue
                
            if getattr(r, 'quality_score', 0.0) < args.min_quality:
                continue
                
            # Deterministic split: 90/10
            h = int(hashlib.md5(r.replay_id.encode()).hexdigest(), 16)
            split = "VALIDATION" if (h % 10) == 0 else "TRAIN"
            
            # Format Dataset Streams
            # For simplicity, we create one record per replay capturing available streams
            record = {
                "replay_id": r.replay_id,
                "task": r.task,
                "plan": r.plan,
                "repair_history_length": len(r.repair_history),
                "oracle_solutions": [o.model_dump() for o in r.oracle_solutions] if r.oracle_solutions else []
            }
            
            # 6. Dataset Streams Formatting
            if r.final_status == "SUCCESS" and len(r.repair_history) == 0:
                record["stream"] = "SUCCESS"
                record["context"] = r.diagnostics # Mock context mapping
            elif r.final_status == "FAILURE" and r.oracle_solutions:
                record["stream"] = "FAILURE"
                record["failed_patch"] = r.repair_history[-1].get("patch_applied") if r.repair_history else None
                record["oracle_patch"] = r.oracle_solutions[-1].patch
            elif r.final_status == "SUCCESS" and len(r.repair_history) > 0:
                record["stream"] = "REPAIR"
                record["diagnostics"] = r.diagnostics
                record["previous_patch"] = r.repair_history[-2].get("patch_applied") if len(r.repair_history) > 1 else None
                record["repair_patch"] = r.repair_history[-1].get("patch_applied")
            else:
                continue # Unknown stream / Not useful for dataset
                
            if split == "TRAIN" and "TRAIN" in splits_requested:
                train_data.append(record)
                ids_included.append(r.replay_id)
            elif split == "VALIDATION" and "VALIDATION" in splits_requested:
                val_data.append(record)
                ids_included.append(r.replay_id)
                
        # Export logic
        hasher = hashlib.sha256()
        
        if "TRAIN" in splits_requested:
            with open(out_dir / "train.jsonl", 'w', encoding='utf-8', newline='\n') as f:
                for d in train_data:
                    line = json.dumps(d) + "\n"
                    f.write(line)
                    hasher.update(line.encode('utf-8'))
                    
        if "VALIDATION" in splits_requested:
            with open(out_dir / "validation.jsonl", 'w', encoding='utf-8', newline='\n') as f:
                for d in val_data:
                    line = json.dumps(d) + "\n"
                    f.write(line)
                    hasher.update(line.encode('utf-8'))
                    
        manifest_hash = hasher.hexdigest()
        
        manifest = DatasetManifest(
            dataset_version=args.version,
            generated_at=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            replay_ids_included=ids_included,
            filters_applied={"min_quality": args.min_quality, "splits": splits_requested},
            dataset_hash=manifest_hash
        )
        
        with open(out_dir / "dataset_manifest.json", 'w', encoding='utf-8') as f:
            f.write(manifest.model_dump_json(indent=2))
            
        print(f"Dataset Export Complete. Version: {args.version}")
        print(f"TRAIN records: {len(train_data)}")
        print(f"VALIDATION records: {len(val_data)}")
        print(f"Manifest written to {out_dir / 'dataset_manifest.json'}")

    elif args.command == "verify-dataset":
        from agent.evaluation.replay_schemas import DatasetManifest
        import hashlib
        
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Error: Manifest {args.manifest} not found.")
            return
            
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)
            
        manifest = DatasetManifest(**manifest_data)
        out_dir = manifest_path.parent
        
        splits_applied = manifest.filters_applied.get("splits", ["TRAIN", "VALIDATION"])
        
        hasher = hashlib.sha256()
        
        if "TRAIN" in splits_applied:
            train_path = out_dir / "train.jsonl"
            if train_path.exists():
                with open(train_path, 'rb') as f:
                    for line in f:
                        hasher.update(line)
            else:
                print(f"Error: Missing {train_path}")
                return
                
        if "VALIDATION" in splits_applied:
            val_path = out_dir / "validation.jsonl"
            if val_path.exists():
                with open(val_path, 'rb') as f:
                    for line in f:
                        hasher.update(line)
            else:
                print(f"Error: Missing {val_path}")
                return
                
        actual_hash = hasher.hexdigest()
        
        if actual_hash == manifest.dataset_hash:
            print("PASS")
        else:
            print("FAIL")
            print(f"Expected hash: {manifest.dataset_hash}")
            print(f"Actual hash: {actual_hash}")

    elif args.command == "memory-benchmark":
        from agent.evaluation.replay_manager import ReplayManager
        from agent.orchestrator import Orchestrator
        from agent.memory.manager import MemoryManager
        from agent.memory.schemas import MemoryRecord, MemoryType, MemoryMetadata
        import tempfile
        import time
        
        mgr = ReplayManager()
        replays = mgr.list_replays()
        artifact = next((r for r in replays if r.replay_id == args.replay_id), None)
        if not artifact:
            print(f"Error: Replay artifact {args.replay_id} not found.")
            return
            
        mem_mgr = MemoryManager(Path("~/.antigravity_data/memory").expanduser())
        
        results = []
        variants = ["OFF", "ON", "ON_INCORRECT"]
        
        for variant in variants:
            print(f"\n--- Running Memory Benchmark Variant: {variant} ---")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_workspace = Path(temp_dir) / "workspace"
                if not mgr.extract_and_verify(artifact, temp_workspace):
                    print("Error: Archive extraction failed.")
                    continue
                    
                orchestrator = Orchestrator(workspace_path=temp_workspace, memory_manager=mem_mgr)
                
                injected_memories = None
                if variant == "ON":
                    injected_memories = await mem_mgr.successful_repairs(artifact.task, ["python"], limit=2)
                elif variant == "ON_INCORRECT":
                    injected_memories = await mem_mgr.successful_repairs(artifact.task, ["python"], limit=2)
                    fake_mem = MemoryRecord(
                        memory_id="fake_1",
                        memory_type=MemoryType.ORACLE_SOLUTION,
                        importance_score=1.0,
                        embedding_text="incorrect memory",
                        content="An completely unrelated fix that might confuse the model. import os\nos.system('echo bad')",
                        metadata=MemoryMetadata(task="fake", source_id="fake", workspace_fingerprint=[])
                    )
                    injected_memories.insert(0, fake_mem)
                
                start_time = time.time()
                try:
                    report_path = await orchestrator.run(artifact.task, injected_memories=injected_memories)
                    json_path = report_path.replace('.md', '.json')
                    with open(json_path, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    status = report_data.get("final_status", "FAILURE")
                    attempts = len(report_data.get("repair_history", []))
                except Exception as e:
                    print(f"Error during execution: {e}")
                    status = "FAILURE"
                    attempts = 0
                    
                duration = time.time() - start_time
                print(f"Variant {variant} -> Status: {status}, Attempts: {attempts}, Runtime: {duration:.2f}s")
                results.append((variant, status, attempts, duration))
                
        print("\n# Memory Benefit Benchmark Results")
        for res in results:
            print(f"Variant: {res[0]:<15} | Status: {res[1]:<7} | Repairs: {res[2]} | Runtime: {res[3]:.2f}s")

    elif args.command == "reflection-analysis":
        if not os.path.exists(args.suite):
            print(f"Error: Suite file {args.suite} not found.")
            return
            
        from agent.evaluation.reflection_benchmark import run_reflection_analysis
        await run_reflection_analysis(args.suite)

    elif args.command == "claude-analysis":
        if not os.path.exists(args.suite):
            print(f"Error: Suite file {args.suite} not found.")
            return
            
        from agent.evaluation.claude_benchmark import run_claude_analysis
        await run_claude_analysis(args.suite)

if __name__ == "__main__":
    asyncio.run(main())
