from datetime import datetime, timezone
from pathlib import Path
from agent.llm.factory import build_client
from agent.planner.core import Planner
from agent.coder.core import Coder
from agent.files.core import FileManager, apply_search_replace_text
from agent.execution.core import Executor
from agent.git.core import GitManager
from agent.llm.pricing import estimate_cost
from agent.reporting.core import Reporter
from agent.retrieval import RipgrepSearch, TreeSitterIndexer, RepositoryMap, SymbolIndex, RetrievalManager
from agent.validation import PatchValidator, BuildValidator, LintValidator, TestValidator
from agent.repair import RollbackManager, RepairCoder, RepairManager, ConstraintExtractor, ConstraintValidator
from agent.review.budget import BudgetManager
from agent.models.schemas import Task, Report, ValidationReport, RepairResult, RepairMetrics, RepairPatch, RepairScope, RoutingCause, ExecutionOutcome, CommandExecution, ValidationDiagnostic, RoleUsage, CostSummary
from agent.exceptions.errors import AgentError, ExecutionError
from agent.config import settings, logger
from agent.review.confidence import ConfidenceEngine
from agent.review.router import ReviewRouter
from agent.review.schemas import ReviewDecision
from agent.reviewers.claude_reviewer import ClaudeReviewer
from agent.review.arbitration import Arbitrator
from agent.reflection.manager import ReflectionManager
from agent.reflection.schemas import ReflectionResult
from agent.safety.controller import SafetyController, SafetyMode
from agent.memory.project_memory import ProjectMemoryManager

class Orchestrator:
    def __init__(self, workspace_path: Path = None, reports_dir: Path = None, memory_manager=None, reflection_enabled: bool = True, shadow_reflection: bool = False, claude_enabled: bool = True, claude_always_on: bool = False, budget_enforcement: bool = True, safety_mode: SafetyMode = None, safety_controller: SafetyController = None):
        ws_path = workspace_path if workspace_path else settings.get_workspace_path()
        self.reflection_enabled = reflection_enabled
        self.shadow_reflection = shadow_reflection
        self.claude_enabled = claude_enabled
        self.claude_always_on = claude_always_on
        self.memory_manager = memory_manager
        self.project_memory = ProjectMemoryManager(ws_path)
        # Phase 5: safety controls. The CLI (main.py) builds an interactive
        # controller from --yes/--dry-run. For programmatic/test construction the
        # default auto-approves confirmation prompts (so no run hangs on stdin) —
        # the hard denylist and workspace jail still apply regardless.
        self.safety = safety_controller or SafetyController(safety_mode or SafetyMode(auto_approve=True))
        # Phase 1: each role gets its own client from the factory. In default env
        # every role resolves to a local Ollama client with that role's model.
        self.planner = Planner(build_client("planner"))
        self.coder = Coder(build_client("coder"))
        self.file_manager = FileManager(ws_path)
        self.executor = Executor(ws_path, dry_run=self.safety.mode.dry_run)
        # Phase 7B: git integration (gated; only acts when GIT_INTEGRATION is on and
        # the workspace is a git repo and not in --dry-run).
        self.git_manager = GitManager(ws_path)
        self.reporter = Reporter(reports_dir)
        
        # Retrieval components
        self.rg = RipgrepSearch()
        self.ts_indexer = TreeSitterIndexer()
        self.repo_map = RepositoryMap()
        self.sym_idx = SymbolIndex(self.ts_indexer)
        self.retrieval_manager = RetrievalManager(self.rg, self.sym_idx, self.repo_map, ws_path)
        
        # Validation components
        self.patch_validator = PatchValidator(ws_path)
        self.build_validator = BuildValidator(self.executor)
        self.lint_validator = LintValidator(self.executor)
        self.test_validator = TestValidator(self.executor)
        
        # Repair components
        self.rollback_manager = RollbackManager(ws_path)
        self.repair_coder = RepairCoder(build_client("repair"), ws_path)
        self.repair_manager = RepairManager(self.retrieval_manager, self.repair_coder, self.rollback_manager, self.memory_manager)
        self.constraint_extractor = ConstraintExtractor(build_client("constraint"))
        
        # Review components
        self.confidence_engine = ConfidenceEngine()
        self.review_router = ReviewRouter()
        self.claude_reviewer = ClaudeReviewer()
        self.budget_manager = BudgetManager(enforcement_enabled=budget_enforcement)
        self.arbitrator = Arbitrator()
        
        # Reflection component
        self.reflection_manager = ReflectionManager(build_client("reflection"))
        self.last_state = None

    async def run(self, task_description: str, injected_memories: list = None) -> str:
        logger.info(f"Starting new task: {task_description}")
        
        original_task = Task(description=task_description)
        task = Task(description=task_description)
        runtime_state = None
        try:
            from agent.state.agent_state import AgentState, TaskMetadata
            runtime_state = AgentState(user_request=task_description,
                                       task=TaskMetadata(description=task_description),
                                       execution_mode="pipeline")
            runtime_state.objective = task_description
            runtime_state.add_timeline("engine", "pipeline run started")
        except Exception:
            runtime_state = None

        # Phase 3: optional prompt refinement (toggleable, fail-open). Runs BEFORE
        # constraint extraction / planning. The untouched original_task is always
        # preserved for the report; ANY refiner failure falls back to the raw prompt
        # so a run is never blocked. When REFINER_ENABLED is false (default) this
        # whole block is skipped and the path is byte-for-byte the old one.
        refinement = None
        refiner_usage = None
        if settings.refiner_enabled:
            try:
                from agent.refiner.core import PromptRefiner
                refiner = PromptRefiner(build_client("refiner"))
                refinement = await refiner.refine(task_description)
                refiner_usage = getattr(refiner, "last_usage", None)
                refined_desc = refinement.refined_task
                if refinement.acceptance_criteria:
                    refined_desc += "\n\nAcceptance Criteria:\n" + "\n".join(
                        f"- {c}" for c in refinement.acceptance_criteria
                    )
                task.description = refined_desc
                logger.info("Prompt refiner rewrote the task for planning.")
            except Exception as e:
                logger.warning(f"Prompt refiner failed; falling back to raw prompt: {e}")
                refinement = None

        # Phase 7B: git integration — create a task branch at run start (only when
        # GIT_INTEGRATION is on, the workspace is a git repo, and not a dry-run).
        # Non-git workspaces fall back to today's snapshot-only behavior.
        git_branch = None
        git_commit = None
        git_active = (
            settings.git_integration
            and not self.safety.mode.dry_run
            and self.git_manager.is_git_repo()
        )
        if git_active:
            try:
                git_branch = self.git_manager.create_task_branch(original_task.description)
                logger.info(f"Git: created task branch '{git_branch}'.")
            except Exception as e:
                logger.warning(f"Git: could not create task branch, continuing without it: {e}")
                git_active = False

        # Phase 5.1: Memory Retrieval During Planning
        active_fingerprint = []
        try:
            if list(self.file_manager.workspace_path.glob("*.py")) or (self.file_manager.workspace_path / "requirements.txt").exists():
                active_fingerprint.append("python")
            if (self.file_manager.workspace_path / "package.json").exists():
                active_fingerprint.append("javascript")
        except Exception:
            pass
            
        mem_summaries = []
        
        if injected_memories:
            for m in injected_memories:
                mem_summaries.append(f"INJECTED MEMORY: {m.embedding_text[:100]}... -> {m.content[:200]}...")
        elif hasattr(self, 'memory_manager') and self.memory_manager and self.memory_manager.enabled:
            try:
                repairs = await self.memory_manager.successful_repairs(task_description, active_fingerprint, limit=2)
                constraints = await self.memory_manager.constraint_similar_failures(task_description, active_fingerprint, limit=1)
                
                # Combine up to 3
                combined = repairs + constraints
                for m in combined[:3]:
                    type_str = m.memory_type.value
                    mem_summaries.append(f"[{type_str}] {m.metadata.task[:100]}: {m.content[:300]}...")
            except Exception as e:
                logger.warning(f"Memory retrieval during planning failed, continuing: {e}")
                
        if mem_summaries:
            mem_text = "\n\nHistorical Context (Learnings from previous runs):\n"
            for s in mem_summaries:
                mem_text += f"- {s}\n"
            task.description += mem_text
            
        plan = None
        context = None
        files_modified = []
        commands_executed = []
        proposed_commands = []
        blocked_commands = []
        execution_results = ""
        final_status = "FAILURE"
        
        try:
            if runtime_state is not None:
                runtime_state.add_timeline("planning", "constraint extraction started")
            extraction_res = await self.constraint_extractor.extract(task.description)
            if not extraction_res.success:
                msg = "Task failed: Constraint extraction failed and explicit constraint language was found. Failing closed."
                logger.error(msg)
                if runtime_state is not None:
                    runtime_state.final_outputs.status = "FAILURE"
                    runtime_state.final_outputs.summary = msg
                    runtime_state.add_timeline("failure", msg)
                    self.last_state = runtime_state
                return msg
                
            constraints = extraction_res.constraints
            if runtime_state is not None:
                runtime_state.add_timeline("planning", f"constraints extracted: {len(constraints)}")

            # Phase 9: build/load the repository Context Bundle (local, cached) and
            # inject it into the planner prompt. When CONTEXT_ENGINE_ENABLED is
            # false, build_context_bundle() returns None and create_plan() gets no
            # bundle, so the planner prompt is byte-for-byte the Round 1 path
            # (pipeline parity). Fail-open: any error continues without context.
            # No LLM is passed here (machine-only / cache reuse) to avoid an extra
            # mid-run model call; `localcli context` builds the richer summary.
            context_bundle = None
            try:
                if runtime_state is not None:
                    runtime_state.add_timeline("context", "context loading started")
                from agent.context import build_context_bundle
                context_bundle = await build_context_bundle(self.file_manager.workspace)
                if runtime_state is not None:
                    runtime_state.add_timeline("context", "context loaded" if context_bundle else "context disabled")
            except Exception as e:
                logger.warning(f"Context engine failed; continuing without it: {e}")
                if runtime_state is not None:
                    runtime_state.add_timeline("context", f"context load failed: {type(e).__name__}")

            try:
                if runtime_state is not None:
                    runtime_state.add_timeline("memory", "project memory loading started")
                if runtime_state is not None:
                    runtime_state.loaded_context = context_bundle
                    bundle = self.project_memory.load_into_state(runtime_state)
                    context_bundle = runtime_state.loaded_context
                else:
                    bundle = self.project_memory.load()
                    if context_bundle is not None:
                        self.project_memory.inject_into_context(context_bundle, bundle)
                if runtime_state is not None:
                    runtime_state.add_timeline("memory", f"project memory loaded: {len(bundle.used_files)} file(s)")
            except Exception as e:
                logger.warning(f"Project memory load failed; continuing without it: {e}")
                if runtime_state is not None:
                    runtime_state.add_timeline("memory", f"project memory load failed: {type(e).__name__}")

            if runtime_state is not None:
                runtime_state.add_timeline("planning", "planner started")
            plan = await self.planner.create_plan(task, context_bundle)
            if runtime_state is not None:
                runtime_state.add_timeline("planning", f"planner produced {len(plan.steps)} step(s)")
                runtime_state.add_timeline("retrieval", "retrieval started")
            context = await self.retrieval_manager.search_context(task_description, plan)
            if runtime_state is not None:
                runtime_state.add_timeline("retrieval", f"retrieved {context.total_files if context else 0} file(s)")
            try:
                if runtime_state is not None and settings.repo_graph_enabled and context:
                    runtime_state.add_timeline("graph", "graph impact lookup started")
                    from agent.graph import GraphBuilder, ImpactAnalyzer
                    graph = GraphBuilder(self.file_manager.workspace).build(use_cache=True)
                    analyzer = ImpactAnalyzer(graph)
                    for res in context.results[:10]:
                        analyzer.record_impact(runtime_state, res.file)
                    runtime_state.add_timeline("graph", "graph impact lookup complete")
            except Exception as e:
                logger.warning(f"Repository graph impact lookup failed; continuing: {e}")
                if runtime_state is not None:
                    runtime_state.add_timeline("graph", f"graph impact lookup failed: {type(e).__name__}")
            
            MAX_CLAUDE_REPAIR_CYCLES = 1
            claude_cycles = 0
            external_review_report = None
            
            while claude_cycles <= MAX_CLAUDE_REPAIR_CYCLES:
                if runtime_state is not None:
                    runtime_state.add_timeline("planning", "coder patch generation started")
                patch = await self.coder.generate_patch(task, plan, context)
                if runtime_state is not None:
                    runtime_state.add_timeline(
                        "planning",
                        f"coder produced {len(patch.operations)} operation(s), {len(patch.commands)} command(s)",
                    )
                
                reflection_triggered = False
                reflection_retry_used = False
                reflection_report = None
                reflection_passes = 0
                
                if self.shadow_reflection:
                    logger.info("Running Reflection in SHADOW MODE...")
                    reflection_triggered = True
                    reflection_report = await self.reflection_manager.reflect(
                        task, constraints, context, mem_summaries, patch
                    )
                else:
                    while self.reflection_enabled and reflection_passes < 2:
                        reflection_passes += 1
                        logger.info(f"Running Reflection pass {reflection_passes}...")
                        if runtime_state is not None:
                            runtime_state.add_timeline("reflection", f"reflection pass {reflection_passes} started")
                        reflection_triggered = True
                        reflection_report = await self.reflection_manager.reflect(
                            task, constraints, context, mem_summaries, patch
                        )
                        
                        if reflection_report.result == ReflectionResult.FAIL and reflection_passes < 2:
                            logger.warning("Reflection returned FAIL. Generating replacement patch...")
                            reflection_retry_used = True
                            critiques_str = "\n".join([f"- {c.category.value}: {c.explanation} (Severity: {c.severity})" for c in reflection_report.critiques])
                            repair_task_desc = f"{task.description}\n\nREFLECTION REVIEW FAILED:\n{reflection_report.summary}\nCritiques:\n{critiques_str}\n\nPlease generate a new replacement patch that addresses these issues."
                            repair_task = Task(description=repair_task_desc)
                            patch = await self.coder.generate_patch(repair_task, plan, context)
                        else:
                            break
                            
                allowed_paths = list(set([op.path for op in patch.operations]))
                repair_scope = RepairScope(allowed_paths=allowed_paths)
                
                self.rollback_manager.checkpoint(allowed_paths)
                
                repair_history = []
                attempts = 0
                max_attempts = settings.max_repair_attempts
                all_success = False
                
                while not all_success and attempts <= max_attempts:
                    original_patch = patch
                    
                    logger.info(f"Running patch validation (Attempt {attempts})...")
                    if runtime_state is not None:
                        runtime_state.add_timeline("validation", f"patch validation attempt {attempts} started")
                    patch_val_result = self.patch_validator.validate_and_repair(patch)
                    
                    validation_report = ValidationReport()
                    validation_report.patch_validation = patch_val_result
                    
                    if not patch_val_result.is_valid:
                        logger.error("Patch validation failed. Aborting current attempt.")
                        if runtime_state is not None:
                            runtime_state.add_timeline("validation", "patch validation failed")
                        if attempts == 0:
                            execution_results = "Initial patch validation failed. See diagnostics."
                            final_status = "FAILURE"
                            break
                        else:
                            repair_history.append(RepairResult(
                                attempt_number=attempts,
                                classification="PATCH_FAILURE",
                                patch_applied=original_patch,
                                validation_result=validation_report,
                                success=False
                            ))
                            all_success = False
                    else:
                        modified = patch_val_result.modified_patch
                        if hasattr(original_patch, "explanation"):
                            patch = RepairPatch(
                                operations=modified.operations,
                                commands=modified.commands,
                                explanation=original_patch.explanation,
                                confidence=original_patch.confidence
                            )
                        else:
                            patch = modified
                            
                        if not patch.operations and not patch.commands:
                            logger.warning("Generated patch is entirely empty after validation.")
                            if runtime_state is not None:
                                runtime_state.add_timeline("validation", "patch was empty")
                            if attempts == 0:
                                execution_results = "Initial patch was empty. Nothing to do."
                                final_status = "SUCCESS"
                                break
                            else:
                                repair_history.append(RepairResult(
                                    attempt_number=attempts,
                                    classification="EMPTY_PATCH",
                                    patch_applied=patch,
                                    validation_result=validation_report,
                                    success=False
                                ))
                                all_success = False
                        else:
                            logger.info("Running constraint validation...")
                            constraint_res = ConstraintValidator.validate(patch, constraints, repair_scope if attempts > 0 else None)
                            if not constraint_res.is_valid:
                                logger.error(f"Constraint validation failed: {constraint_res.violations}")
                                if attempts == 0:
                                    execution_results = f"Initial patch violated constraints: {constraint_res.violations}"
                                    final_status = "FAILURE"
                                    break
                                else:
                                    repair_history.append(RepairResult(
                                        attempt_number=attempts,
                                        classification="CONSTRAINT_VIOLATION: " + "; ".join(constraint_res.violations),
                                        patch_applied=patch,
                                        validation_result=validation_report,
                                        success=False
                                    ))
                                    all_success = False
                            else:
                                logger.info("Applying patches...")
                                if runtime_state is not None:
                                    runtime_state.add_timeline("engine", f"applying {len(patch.operations)} file operation(s)")
                                for op in patch.operations:
                                    try:
                                        # Phase 5: preview a unified diff and require
                                        # confirmation before touching disk. Auto-approved
                                        # under --yes; previewed-only (skipped) under
                                        # --dry-run; declined ops are skipped.
                                        # Phase 7A: search_replace edits a large file in
                                        # place (old vs new computed via the exactly-once
                                        # text helper) but applies through the SAME jail +
                                        # diff-preview + dry-run + confirmation path.
                                        if op.type in ("update_file", "search_replace"):
                                            try:
                                                old_content = await self.file_manager.read_file(op.path)
                                            except Exception:
                                                old_content = ""
                                        else:
                                            old_content = ""

                                        if op.type == "search_replace":
                                            new_content = apply_search_replace_text(old_content, op.search or "", op.replace or "")
                                        else:
                                            new_content = op.content or ""

                                        if not self.safety.confirm_file_op(op.type, op.path, old_content, new_content):
                                            logger.info(f"File op skipped by safety controls: {op.type} {op.path}")
                                            continue

                                        if op.type == "create_file":
                                            await self.file_manager.create_file(op.path, new_content)
                                            files_modified.append(op.path)
                                            self.rollback_manager.track_new_file(op.path)
                                            if runtime_state is not None:
                                                runtime_state.record_file_change(op.path, op.type)
                                        elif op.type in ("update_file", "search_replace"):
                                            await self.file_manager.update_file(op.path, new_content)
                                            files_modified.append(op.path)
                                            if runtime_state is not None:
                                                runtime_state.record_file_change(op.path, op.type)
                                    except Exception as e:
                                        logger.error(f"Failed to apply patch operation: {e}")
                                        
                                build_res = await self.build_validator.validate()
                                validation_report.build_result = build_res
                                if runtime_state is not None:
                                    runtime_state.add_timeline("validation", f"BUILD {'passed' if build_res.success else 'failed'}")
                                
                                lint_res = await self.lint_validator.validate()
                                validation_report.lint_result = lint_res
                                if runtime_state is not None:
                                    runtime_state.add_timeline("validation", f"LINT {'passed' if lint_res.success else 'failed'}")
                                
                                test_res = await self.test_validator.validate()
                                validation_report.test_result = test_res
                                if runtime_state is not None:
                                    runtime_state.add_timeline("validation", f"TEST {'passed' if test_res.success else 'failed'}")

                                # Phase 4a: optionally run the coder's proposed commands.
                                # Gated behind EXECUTE_COMMANDS (default off) so nothing
                                # runs silently. A non-zero exit is folded into the BUILD
                                # diagnostic slot so the EXISTING self-healing repair loop
                                # (classify -> retrieve -> repair -> rollback) reacts to it
                                # exactly as it does to a failing build/test — no parallel path.
                                proposed_commands = list(patch.commands or [])
                                commands_success = True
                                if proposed_commands and settings.execute_commands:
                                    first_failed_cmd = None
                                    for cmd in proposed_commands:
                                        # Phase 5: route every command through safety.
                                        # Denylisted -> blocked (never runs). dry-run /
                                        # user-declined -> not executed (not a failure,
                                        # so they don't feed the repair loop).
                                        verdict = self.safety.check_command(cmd)
                                        if verdict.status == "blocked":
                                            blocked_commands.append(cmd)
                                            continue
                                        if not verdict.allowed:
                                            logger.info(f"Command not executed ({verdict.status}): {cmd}")
                                            continue
                                        try:
                                            cmd_result = await self.executor.run_command(cmd)
                                        except ExecutionError as e:
                                            cmd_result = CommandExecution(command=cmd, stdout="", stderr=str(e), exit_code=124, duration=0.0)
                                        commands_executed.append(cmd_result)
                                        if runtime_state is not None:
                                            runtime_state.add_timeline("tool", f"command {cmd}: exit={cmd_result.exit_code}")
                                        if cmd_result.exit_code != 0:
                                            commands_success = False
                                            if first_failed_cmd is None:
                                                first_failed_cmd = cmd_result
                                    if first_failed_cmd is not None and build_res.success:
                                        build_res = ValidationDiagnostic(
                                            stage="BUILD",
                                            command=first_failed_cmd.command,
                                            success=False,
                                            stdout=first_failed_cmd.stdout,
                                            stderr=first_failed_cmd.stderr or first_failed_cmd.stdout,
                                            exit_code=first_failed_cmd.exit_code,
                                            duration=first_failed_cmd.duration,
                                        )
                                        validation_report.build_result = build_res

                                all_success = build_res.success and lint_res.success and test_res.success and commands_success
                                
                                if attempts > 0:
                                    repair_history.append(RepairResult(
                                        attempt_number=attempts,
                                        classification="UNKNOWN",
                                        patch_applied=patch,
                                        validation_result=validation_report,
                                        success=all_success
                                    ))
                                    
                                if all_success:
                                    final_status = "SUCCESS"
                                    execution_results = f"Task executed and validated successfully (after {attempts} repairs)."
                                    break
                                    
                    if not all_success:
                        if attempts < max_attempts:
                            attempts += 1
                            logger.info(f"Validation failed. Initiating repair loop attempt {attempts}/{max_attempts}.")
                            if runtime_state is not None:
                                runtime_state.add_timeline("repair", f"repair attempt {attempts} started")
                            repair_context = await self.repair_manager.build_context(task, plan, validation_report, constraints, repair_scope)
                            if repair_history and repair_history[-1].classification == "UNKNOWN":
                                 repair_history[-1].classification = repair_context.normalized_diagnostic.classification
                                 
                            next_patch = await self.repair_manager.generate_repair(repair_context)
                            if not next_patch:
                                execution_results = "Repair manager aborted (likely duplicate patch)."
                                final_status = "FAILURE"
                                break
                            patch = next_patch
                            for op in patch.operations:
                                 if op.type == "create_file":
                                      self.rollback_manager.track_new_file(op.path)
                        else:
                            final_status = "FAILURE"
                            execution_results = f"Task failed validation and exhausted {max_attempts} repair attempts."
                            break
                            
                rollback_triggered = False
                rollback_results = {}
                if not all_success and final_status == "FAILURE":
                     logger.warning("Max repair attempts failed. Triggering rollback.")
                     if runtime_state is not None:
                         runtime_state.add_timeline("repair", "rollback started")
                     self.rollback_manager.restore()
                     rollback_triggered = True
                     rollback_results = self.rollback_manager.verify()
                     
                repair_metrics = RepairMetrics(
                     total_attempts=attempts,
                     resolved_in_attempt=attempts if all_success and attempts > 0 else None,
                     rollback_triggered=rollback_triggered
                )
                
                # Confidence Engine
                build_success = validation_report.build_result.success if validation_report.build_result else False
                test_success = validation_report.test_result.success if validation_report.test_result else False
                lint_success = validation_report.lint_result.success if validation_report.lint_result else False
                
                c_violations = len([r for r in repair_history if r.classification and "CONSTRAINT_VIOLATION" in r.classification])
                
                confidence_report = self.confidence_engine.evaluate(
                    build_success=build_success,
                    test_success=test_success,
                    lint_success=lint_success,
                    repair_attempts=attempts,
                    memory_count=len(mem_summaries),
                    files_modified_count=len(list(set(files_modified))),
                    plan_step_count=len(plan.steps) if plan and hasattr(plan, 'steps') else 0,
                    constraint_violations=c_violations
                )
                
                review_decision = self.review_router.route(confidence_report)
                
                # We start assuming it skipped due to high confidence if it's APPROVE
                routing_cause = RoutingCause.SKIPPED_HIGH_CONFIDENCE if review_decision == ReviewDecision.APPROVE else None
                
                if not self.claude_enabled:
                    logger.info("Claude is disabled. Bypassing review.")
                    if not routing_cause: routing_cause = RoutingCause.SKIPPED_HIGH_CONFIDENCE
                    break
                    
                if self.claude_always_on:
                    logger.info("Claude ALWAYS ON mode is active. Forcing MANDATORY_REVIEW.")
                    review_decision = ReviewDecision.MANDATORY_REVIEW
                    routing_cause = None
                
                if review_decision == ReviewDecision.APPROVE:
                    logger.info("Confidence score >= 95. Bypassing Claude.")
                    routing_cause = RoutingCause.SKIPPED_HIGH_CONFIDENCE
                    break
                    
                if not self.budget_manager.can_afford():
                    logger.error("Budget exhausted. Escalatating to FAIL_CLOSED.")
                    routing_cause = RoutingCause.BUDGET_EXHAUSTED
                    final_status = "FAILURE"
                    execution_outcome = ExecutionOutcome.FAIL_CLOSED
                    break
                    
                patch_len = len(patch.model_dump_json()) if patch else 0
                estimated_tokens = len(task_description) // 4 + patch_len // 4
                if not self.budget_manager.check_payload(estimated_tokens):
                    logger.error("Payload overflow. Escalating to FAIL_CLOSED.")
                    routing_cause = RoutingCause.PAYLOAD_OVERFLOW
                    final_status = "FAILURE"
                    execution_outcome = ExecutionOutcome.FAIL_CLOSED
                    break
                    
                logger.info(f"Review decision is {review_decision.value}. Calling ClaudeReviewer...")
                routing_cause = RoutingCause.CLAUDE_REVIEWED
                external_review_report = await self.claude_reviewer.review(
                    task=task,
                    plan=plan,
                    patch=patch,
                    validation_report=validation_report,
                    reflection_report=reflection_report
                )
                
                if external_review_report:
                    self.budget_manager.add_cost(external_review_report.estimated_cost)
                
                if review_decision == ReviewDecision.MANDATORY_REVIEW and claude_cycles < MAX_CLAUDE_REPAIR_CYCLES:
                    if external_review_report.issues:
                        logger.warning("Claude found issues during MANDATORY_REVIEW. Triggering 1 local repair cycle.")
                        claude_cycles += 1
                        issues_str = "\n".join([f"- {i.category.value} ({i.severity.value}): {i.description}" for i in external_review_report.issues])
                        repair_task_desc = f"{original_task.description}\n\nCLAUDE EXTERNAL REVIEW FAILED:\nSummary: {external_review_report.summary}\nIssues:\n{issues_str}\n\nPlease generate a new replacement patch addressing these issues."
                        task = Task(description=repair_task_desc)
                        continue
                    else:
                        logger.info("Claude found NO issues during MANDATORY_REVIEW. Proceeding.")
                        break
                else:
                    break

        except AgentError as e:
            execution_results = f"Agent encountered an error: {str(e)}"
            logger.error(execution_results)
            if runtime_state is not None:
                runtime_state.add_timeline("failure", execution_results)
        except Exception as e:
            execution_results = f"Unexpected error: {str(e)}"
            logger.exception(execution_results)
            if runtime_state is not None:
                runtime_state.add_timeline("failure", execution_results)
            
        if 'routing_cause' not in locals() or not routing_cause:
            routing_cause = RoutingCause.SKIPPED_HIGH_CONFIDENCE
            
        if 'execution_outcome' not in locals():
            if final_status == "SUCCESS":
                execution_outcome = ExecutionOutcome.SUCCESS
            else:
                execution_outcome = ExecutionOutcome.FAILURE
                
        # Phase 6.5 Arbitration (Only if not failed closed due to execution limits)
        arbitration_report = None
        if execution_outcome != ExecutionOutcome.FAIL_CLOSED:
            arbitration_report = self.arbitrator.arbitrate(
                validation_report=validation_report if 'validation_report' in locals() else None,
                reflection_report=reflection_report if 'reflection_report' in locals() else None,
                confidence_report=confidence_report if 'confidence_report' in locals() else None,
                claude_review_report=external_review_report if 'external_review_report' in locals() else None
            )
            
            # Map Arbitration Decision to ExecutionOutcome and Final Status
            from agent.models.schemas import ArbitrationDecision
            if arbitration_report.decision in [ArbitrationDecision.APPROVE_VALIDATED, ArbitrationDecision.APPROVE_CLAUDE_OVERRIDDEN]:
                execution_outcome = ExecutionOutcome.SUCCESS
                final_status = "SUCCESS"
            else:
                execution_outcome = ExecutionOutcome.FAILURE
                final_status = "FAILURE"
            
        # Report
        retrieved_files = []
        retrieved_symbols = []
        if context:
            retrieved_files = [res.file for res in context.results]
            for res in context.results:
                retrieved_symbols.extend(list(set([sym.name for sym in res.matched_symbols])))
                
        # Phase 7C: aggregate per-role LLM usage into a cost summary. est_cost is
        # computed from the price table (cloud) / 0.0 (local). No secrets — only
        # provider/model/token counts. The cloud spend is also fed to the budget
        # manager for accounting (existing gating thresholds are unchanged).
        role_clients = [
            ("planner", self.planner),
            ("coder", self.coder),
            ("constraint", self.constraint_extractor),
            ("repair", self.repair_coder),
            ("reflection", self.reflection_manager),
        ]
        role_usages = []
        if refiner_usage is not None:
            role_usages.append(("refiner", refiner_usage))
        for role, client in role_clients:
            usage = getattr(client, "last_usage", None)
            if usage is not None:
                role_usages.append((role, usage))

        per_role = []
        for role, usage in role_usages:
            usage.est_cost = estimate_cost(usage.provider, usage.model,
                                           usage.input_tokens, usage.output_tokens)
            per_role.append(RoleUsage(
                role=role, provider=usage.provider, model=usage.model,
                input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                est_cost=usage.est_cost,
            ))
        cost_summary = CostSummary(
            per_role=per_role,
            total_input_tokens=sum(r.input_tokens for r in per_role),
            total_output_tokens=sum(r.output_tokens for r in per_role),
            total_est_cost=round(sum(r.est_cost for r in per_role), 6),
        )
        self.llm_usages = [u for _, u in role_usages]
        if cost_summary.total_est_cost:
            self.budget_manager.add_cost(cost_summary.total_est_cost)
        if self.llm_usages:
            logger.debug(f"LLM usage this run: {self.llm_usages}")
        if runtime_state is not None:
            runtime_state.governor.cost_used_usd = cost_summary.total_est_cost

        # Phase 7B: on a successful run commit the applied changes (redacted message);
        # on failure use the complementary git rollback. Never in --dry-run.
        if git_active:
            try:
                if final_status == "SUCCESS":
                    summary_line = (original_task.description or "").strip().splitlines()
                    summary_line = summary_line[0][:72] if summary_line else "automated change"
                    git_commit = self.git_manager.commit_all(f"localcli: {summary_line}")
                    if git_commit:
                        logger.info(f"Git: committed changes as {git_commit[:10]} on '{git_branch}'.")
                else:
                    self.git_manager.rollback()
                    logger.info("Git: rolled back working tree (complementing snapshot rollback).")
            except Exception as e:
                logger.warning(f"Git: post-run operation failed: {e}")

        try:
            if runtime_state is not None:
                from agent.state.agent_state import StepResult, ValidationResult
                runtime_state.final_outputs.status = final_status
                runtime_state.final_outputs.summary = execution_results
                if plan and hasattr(plan, "steps"):
                    if not runtime_state.completed_steps:
                        for i, step in enumerate(plan.steps):
                            runtime_state.completed_steps.append(StepResult(
                                index=i,
                                description=step.description,
                                status="done" if final_status == "SUCCESS" else "failed",
                                summary=step.expected_output,
                                acceptance=step.expected_output,
                            ))
                if "validation_report" in locals() and validation_report:
                    for stage, res in (
                        ("BUILD", validation_report.build_result),
                        ("LINT", validation_report.lint_result),
                        ("TEST", validation_report.test_result),
                    ):
                        if res:
                            runtime_state.validation_results.append(ValidationResult(
                                stage=stage, success=res.success,
                                detail=res.stderr or res.stdout or "",
                            ))
                for summary in mem_summaries:
                    if summary not in runtime_state.memory_refs.summaries:
                        runtime_state.memory_refs.summaries.append(summary)
                if final_status == "SUCCESS":
                    self.project_memory.update_from_state(runtime_state)
        except Exception as e:
            logger.warning(f"Project memory update failed; continuing: {e}")
        if runtime_state is not None:
            runtime_state.add_timeline("completion" if final_status == "SUCCESS" else "failure",
                                       f"run finished: {final_status}")
            self.last_state = runtime_state

        report = Report(
            task=task.description,
            plan=plan,
            retrieved_files=retrieved_files,
            retrieved_symbols=retrieved_symbols,
            validation_report=validation_report if 'validation_report' in locals() else None,
            repair_history=repair_history if 'repair_history' in locals() else [],
            repair_metrics=repair_metrics if 'repair_metrics' in locals() else None,
            files_modified=list(set(files_modified)),
            commands_executed=commands_executed,
            proposed_commands=proposed_commands,
            blocked_commands=blocked_commands,
            cost_summary=cost_summary if 'cost_summary' in locals() else None,
            git_branch=git_branch if 'git_branch' in locals() else None,
            git_commit=git_commit if 'git_commit' in locals() else None,
            execution_results=execution_results,
            final_status=final_status,
            routing_cause=routing_cause,
            execution_outcome=execution_outcome,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence_report=confidence_report if 'confidence_report' in locals() else None,
            review_decision=review_decision.value if 'review_decision' in locals() else None,
            reflection_report=reflection_report if 'reflection_report' in locals() else None,
            reflection_triggered=reflection_triggered if 'reflection_triggered' in locals() else False,
            reflection_retry_used=reflection_retry_used if 'reflection_retry_used' in locals() else False,
            reflection_result=reflection_report.result.value if ('reflection_report' in locals() and reflection_report) else None,
            external_review_report=external_review_report if 'external_review_report' in locals() else None,
            arbitration_report=arbitration_report,
            claude_override_count=1 if (arbitration_report and "CLAUDE" in arbitration_report.overridden_systems) else 0,
            reflection_override_count=1 if (arbitration_report and "REFLECTION" in arbitration_report.overridden_systems) else 0,
            arbitration_triggered_count=1 if (arbitration_report and arbitration_report.overridden_systems) else 0,
            claude_false_approval_count=1 if (arbitration_report and arbitration_report.decision == "REJECT_VALIDATION_FAILED" and "CLAUDE" in arbitration_report.overridden_systems) else 0,
            observability=None,
        )
        if runtime_state is not None and settings.observability_enabled:
            from agent.observability import snapshot_for_report
            report.observability = snapshot_for_report(runtime_state, cost_summary=cost_summary)
        
        report_path = self.reporter.generate_report(report, constraints=constraints if 'constraints' in locals() else [], repair_scope=repair_scope if 'repair_scope' in locals() else None, rollback_results=rollback_results if 'rollback_results' in locals() else {}, refinement=refinement, raw_task=original_task.description)
        logger.info(f"Task finished with status {final_status}. Report: {report_path}")

        return report_path

    # --- Phase 11: incremental planning (opt-in; pipeline strategy) ----------

    async def run_incremental(self, task_description: str) -> str:
        """Pipeline strategy with step-wise execution + replanning.

        This is the pipeline's incremental path: plan -> execute step -> observe
        -> replan, driven by the shared :class:`IncrementalPlanner` over an
        :class:`AgentState`. It REUSES this orchestrator's existing Round 1
        components (planner, coder, patch validator, validators, file manager,
        safety) — nothing is reimplemented. The default ``run()`` above is left
        byte-for-byte unchanged, so ``INCREMENTAL_PLANNING=false`` is exact parity.
        """
        from agent.state.agent_state import AgentState, TaskMetadata
        from agent.planning import IncrementalPlanner, Replanner, StepPlanner, render_plan_evolution
        from agent.engine.governor import Governor

        logger.info(f"Starting incremental (step-wise) task: {task_description}")
        state = AgentState(user_request=task_description,
                           task=TaskMetadata(description=task_description),
                           execution_mode="pipeline")
        state.objective = task_description
        state.add_timeline("engine", "incremental pipeline run started")
        governor = Governor.configure(state)

        # Constraints + context, reusing Round 1 components.
        try:
            extraction_res = await self.constraint_extractor.extract(task_description)
            constraints = extraction_res.constraints if extraction_res.success else []
            state.add_timeline("planning", f"constraints extracted: {len(constraints)}")
        except Exception as e:
            logger.warning(f"Incremental: constraint extraction failed, continuing: {e}")
            state.add_timeline("planning", f"constraint extraction failed: {type(e).__name__}")
            constraints = []

        context_bundle = None
        try:
            from agent.context import build_context_bundle
            state.add_timeline("context", "context loading started")
            context_bundle = await build_context_bundle(self.file_manager.workspace)
            state.add_timeline("context", "context loaded" if context_bundle else "context disabled")
        except Exception as e:
            logger.warning(f"Incremental: context engine failed, continuing: {e}")
            state.add_timeline("context", f"context load failed: {type(e).__name__}")

        try:
            state.loaded_context = context_bundle
            self.project_memory.load_into_state(state)
            context_bundle = state.loaded_context
        except Exception as e:
            logger.warning(f"Incremental: project memory load failed, continuing: {e}")

        # Plan once (reuse the Round 1 planner); the plan lives inside AgentState.
        state.add_timeline("planning", "planner started")
        plan = await self.planner.create_plan(Task(description=task_description), context_bundle)
        state.add_timeline("planning", f"planner produced {len(plan.steps)} step(s)")
        state.add_timeline("retrieval", "retrieval started")
        context = await self.retrieval_manager.search_context(task_description, plan)
        state.add_timeline("retrieval", f"retrieved {context.total_files if context else 0} file(s)")
        try:
            if settings.repo_graph_enabled and context:
                from agent.graph import GraphBuilder, ImpactAnalyzer
                graph = GraphBuilder(self.file_manager.workspace).build(use_cache=True)
                analyzer = ImpactAnalyzer(graph)
                for res in context.results[:10]:
                    analyzer.record_impact(state, res.file)
        except Exception as e:
            logger.warning(f"Incremental: repository graph impact lookup failed, continuing: {e}")

        async def step_executor(st, step):
            return await self._execute_pipeline_step(st, step, plan=plan, context=context,
                                                     constraints=constraints)

        inc = IncrementalPlanner(
            step_planner=StepPlanner(planner=self.planner),
            replanner=Replanner(client=self.planner.llm_client) if settings.replan_on_failure else None,
        )
        await inc.run(state, step_executor, governor=governor, plan=plan,
                      context_bundle=context_bundle)
        try:
            self.project_memory.update_from_state(state)
        except Exception as e:
            logger.warning(f"Incremental: project memory update failed, continuing: {e}")

        final_status = state.final_outputs.status
        self.last_state = state
        retrieved_files = [res.file for res in context.results] if context else []
        report = Report(
            task=task_description,
            plan=plan,
            retrieved_files=retrieved_files,
            files_modified=[fc.path for fc in state.files_modified],
            execution_results=state.final_outputs.summary,
            final_status=final_status,
            execution_outcome=ExecutionOutcome.SUCCESS if final_status == "SUCCESS" else ExecutionOutcome.FAILURE,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if settings.observability_enabled:
            from agent.observability import snapshot_for_report
            report.observability = snapshot_for_report(state)
        report_path = self.reporter.generate_report(
            report, constraints=constraints,
            plan_evolution=render_plan_evolution(state))
        logger.info(f"Incremental task finished with status {final_status}. Report: {report_path}")
        return report_path

    # --- Phase 16: agent orchestration (opt-in) ------------------------------

    async def run_orchestrated(self, task_description: str, *,
                               execution_mode: str = "pipeline") -> str:
        """Coordinator-owned multi-worker run (Phase 16 MVP).

        Builds a parent AgentState with context, memory, and graph evidence,
        delegates to the Coordinator for decomposition/scheduling/merge/validate,
        then generates a report. Only called when ``ORCHESTRATION_ENABLED=true``.
        """
        from agent.state.agent_state import AgentState, TaskMetadata
        from agent.orchestration import Coordinator
        from agent.planning import render_plan_evolution

        logger.info(f"Starting orchestrated task: {task_description}")
        state = AgentState(
            user_request=task_description,
            task=TaskMetadata(description=task_description),
            execution_mode=execution_mode,
        )
        state.objective = task_description
        state.add_timeline("engine", "orchestrated run started")

        try:
            from agent.context import build_context_bundle
            state.add_timeline("context", "context loading started")
            context_bundle = await build_context_bundle(self.file_manager.workspace)
            state.loaded_context = context_bundle
            state.add_timeline("context", "context loaded" if context_bundle else "context disabled")
        except Exception as e:
            logger.warning(f"Orchestrated: context engine failed, continuing: {e}")
            state.add_timeline("context", f"context load failed: {type(e).__name__}")

        try:
            self.project_memory.load_into_state(state)
        except Exception as e:
            logger.warning(f"Orchestrated: project memory load failed, continuing: {e}")

        coordinator = Coordinator(orchestrator=self, safety_mode=self.safety.mode)
        state = await coordinator.run(state)

        try:
            self.project_memory.update_from_state(state)
        except Exception as e:
            logger.warning(f"Orchestrated: project memory update failed, continuing: {e}")

        final_status = state.final_outputs.status
        self.last_state = state
        report = Report(
            task=task_description,
            plan=None,
            retrieved_files=[],
            files_modified=[fc.path for fc in state.files_modified],
            execution_results=state.final_outputs.summary,
            final_status=final_status,
            execution_outcome=ExecutionOutcome.SUCCESS if final_status == "SUCCESS" else ExecutionOutcome.FAILURE,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if settings.observability_enabled:
            from agent.observability import snapshot_for_report
            report.observability = snapshot_for_report(state)
        report_path = self.reporter.generate_report(
            report, plan_evolution=render_plan_evolution(state))
        state.final_outputs.report_path = report_path
        logger.info(f"Orchestrated task finished with status {final_status}. Report: {report_path}")
        return report_path

    async def _execute_pipeline_step(self, state, step, *, plan, context, constraints):
        """Execute ONE plan step via the Round 1 coder + validators (no per-step
        repair loop — replanning handles failures). Appends validation signals to
        the shared state for the Observer."""
        from agent.state.agent_state import ValidationResult
        from agent.planning.incremental import StepOutcome

        step_task = Task(description=(
            f"{state.user_request}\n\nFocus ONLY on this step:\n{step.description}\n"
            f"Acceptance: {step.acceptance}"))
        patch = await self.coder.generate_patch(step_task, plan, context)

        patch_val = self.patch_validator.validate_and_repair(patch)
        if not patch_val.is_valid:
            state.validation_results.append(ValidationResult(
                stage="PATCH", success=False, detail="; ".join(patch_val.errors)))
            return StepOutcome(success=False, summary="patch validation failed")
        patch = patch_val.modified_patch

        if not patch.operations and not patch.commands:
            return StepOutcome(success=True, summary="no-op step (nothing to change)")

        for op in patch.operations:
            try:
                if op.type in ("update_file", "search_replace"):
                    try:
                        old_content = await self.file_manager.read_file(op.path)
                    except Exception:
                        old_content = ""
                else:
                    old_content = ""
                if op.type == "search_replace":
                    new_content = apply_search_replace_text(old_content, op.search or "", op.replace or "")
                else:
                    new_content = op.content or ""
                if not self.safety.confirm_file_op(op.type, op.path, old_content, new_content):
                    continue
                if op.type == "create_file":
                    await self.file_manager.create_file(op.path, new_content)
                    state.record_file_change(op.path, op.type)
                elif op.type in ("update_file", "search_replace"):
                    await self.file_manager.update_file(op.path, new_content)
                    state.record_file_change(op.path, op.type)
                elif op.type == "delete_file":
                    await self.file_manager.delete_file(op.path)
                    state.record_file_change(op.path, op.type)
            except Exception as e:
                logger.error(f"Incremental: failed to apply op {op.type} {op.path}: {e}")
                state.validation_results.append(ValidationResult(
                    stage="PATCH", success=False, detail=f"apply failed: {e}"))
                return StepOutcome(success=False, summary=f"apply failed: {e}")

        build_res = await self.build_validator.validate()
        lint_res = await self.lint_validator.validate()
        test_res = await self.test_validator.validate()
        for stage, res in (("BUILD", build_res), ("LINT", lint_res), ("TEST", test_res)):
            state.validation_results.append(ValidationResult(
                stage=stage, success=res.success, detail=res.stderr[:200] if not res.success else ""))
        ok = build_res.success and lint_res.success and test_res.success
        return StepOutcome(success=ok, summary="validated" if ok else "validation failed")
