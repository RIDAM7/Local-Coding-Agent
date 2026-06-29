from datetime import datetime
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
from agent.models.schemas import Task, Report, Plan, RetrievedContext, ValidationReport, RepairResult, RepairMetrics, RepairPatch, RepairScope, RoutingCause, ExecutionOutcome, CommandExecution, ValidationDiagnostic, RoleUsage, CostSummary
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

class Orchestrator:
    def __init__(self, workspace_path: Path = None, reports_dir: Path = None, memory_manager=None, reflection_enabled: bool = True, shadow_reflection: bool = False, claude_enabled: bool = True, claude_always_on: bool = False, budget_enforcement: bool = True, safety_mode: SafetyMode = None, safety_controller: SafetyController = None):
        ws_path = workspace_path if workspace_path else settings.get_workspace_path()
        self.reflection_enabled = reflection_enabled
        self.shadow_reflection = shadow_reflection
        self.claude_enabled = claude_enabled
        self.claude_always_on = claude_always_on
        self.memory_manager = memory_manager
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

    async def run(self, task_description: str, injected_memories: list = None) -> str:
        logger.info(f"Starting new task: {task_description}")
        
        original_task = Task(description=task_description)
        task = Task(description=task_description)

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
            extraction_res = await self.constraint_extractor.extract(task.description)
            if not extraction_res.success:
                msg = "Task failed: Constraint extraction failed and explicit constraint language was found. Failing closed."
                logger.error(msg)
                return msg
                
            constraints = extraction_res.constraints
            
            plan = await self.planner.create_plan(task)
            context = await self.retrieval_manager.search_context(task_description, plan)
            
            MAX_CLAUDE_REPAIR_CYCLES = 1
            claude_cycles = 0
            external_review_report = None
            
            while claude_cycles <= MAX_CLAUDE_REPAIR_CYCLES:
                patch = await self.coder.generate_patch(task, plan, context)
                
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
                    patch_val_result = self.patch_validator.validate_and_repair(patch)
                    
                    validation_report = ValidationReport()
                    validation_report.patch_validation = patch_val_result
                    
                    if not patch_val_result.is_valid:
                        logger.error("Patch validation failed. Aborting current attempt.")
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
                                        elif op.type in ("update_file", "search_replace"):
                                            await self.file_manager.update_file(op.path, new_content)
                                            files_modified.append(op.path)
                                    except Exception as e:
                                        logger.error(f"Failed to apply patch operation: {e}")
                                        
                                build_res = await self.build_validator.validate()
                                validation_report.build_result = build_res
                                
                                lint_res = await self.lint_validator.validate()
                                validation_report.lint_result = lint_res
                                
                                test_res = await self.test_validator.validate()
                                validation_report.test_result = test_res

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
        except Exception as e:
            execution_results = f"Unexpected error: {str(e)}"
            logger.exception(execution_results)
            
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
            timestamp=datetime.utcnow().isoformat(),
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
            claude_false_approval_count=1 if (arbitration_report and arbitration_report.decision == "REJECT_VALIDATION_FAILED" and "CLAUDE" in arbitration_report.overridden_systems) else 0
        )
        
        report_path = self.reporter.generate_report(report, constraints=constraints if 'constraints' in locals() else [], repair_scope=repair_scope if 'repair_scope' in locals() else None, rollback_results=rollback_results if 'rollback_results' in locals() else {}, refinement=refinement, raw_task=original_task.description)
        logger.info(f"Task finished with status {final_status}. Report: {report_path}")
        
        return report_path
