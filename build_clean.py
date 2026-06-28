import re

with open("agent/orchestrator.py", "r", encoding="utf-8") as f:
    text = f.read()

prefix = text.split("    async def run(")[0]

clean_run = """    async def run(self, task_description: str, injected_memories: list = None) -> str:
        logger.info(f"Starting new task: {task_description}")
        
        original_task = Task(description=task_description)
        task = Task(description=task_description)
        
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
            mem_text = "\\n\\nHistorical Context (Learnings from previous runs):\\n"
            for s in mem_summaries:
                mem_text += f"- {s}\\n"
            task.description += mem_text
            
        plan = None
        context = None
        files_modified = []
        commands_executed = []
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
                            critiques_str = "\\n".join([f"- {c.category.value}: {c.explanation} (Severity: {c.severity})" for c in reflection_report.critiques])
                            repair_task_desc = f"{task.description}\\n\\nREFLECTION REVIEW FAILED:\\n{reflection_report.summary}\\nCritiques:\\n{critiques_str}\\n\\nPlease generate a new replacement patch that addresses these issues."
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
                                        if op.type == "create_file":
                                            await self.file_manager.create_file(op.path, op.content or "")
                                            files_modified.append(op.path)
                                            self.rollback_manager.track_new_file(op.path)
                                        elif op.type == "update_file":
                                            await self.file_manager.update_file(op.path, op.content or "")
                                            files_modified.append(op.path)
                                    except Exception as e:
                                        logger.error(f"Failed to apply patch operation: {e}")
                                        
                                build_res = await self.build_validator.validate()
                                validation_report.build_result = build_res
                                
                                lint_res = await self.lint_validator.validate()
                                validation_report.lint_result = lint_res
                                
                                test_res = await self.test_validator.validate()
                                validation_report.test_result = test_res
                                
                                all_success = build_res.success and lint_res.success and test_res.success
                                
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
                
                if review_decision == ReviewDecision.APPROVE:
                    logger.info("Confidence score >= 95. Bypassing Claude.")
                    break
                    
                logger.info(f"Review decision is {review_decision.value}. Calling ClaudeReviewer...")
                external_review_report = await self.claude_reviewer.review(
                    task=task,
                    plan=plan,
                    patch=patch,
                    validation_report=validation_report,
                    reflection_report=reflection_report
                )
                
                if review_decision == ReviewDecision.MANDATORY_REVIEW and claude_cycles < MAX_CLAUDE_REPAIR_CYCLES:
                    if external_review_report.issues:
                        logger.warning("Claude found issues during MANDATORY_REVIEW. Triggering 1 local repair cycle.")
                        claude_cycles += 1
                        issues_str = "\\n".join([f"- {i.category.value} ({i.severity.value}): {i.description}" for i in external_review_report.issues])
                        repair_task_desc = f"{original_task.description}\\n\\nCLAUDE EXTERNAL REVIEW FAILED:\\nSummary: {external_review_report.summary}\\nIssues:\\n{issues_str}\\n\\nPlease generate a new replacement patch addressing these issues."
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
            logger.error(execution_results)
            
        # Report
        retrieved_files = []
        retrieved_symbols = []
        if context:
            retrieved_files = [res.file for res in context.results]
            for res in context.results:
                retrieved_symbols.extend(list(set([sym.name for sym in res.matched_symbols])))
                
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
            execution_results=execution_results,
            final_status=final_status,
            timestamp=datetime.utcnow().isoformat(),
            confidence_report=confidence_report if 'confidence_report' in locals() else None,
            review_decision=review_decision.value if 'review_decision' in locals() else None,
            reflection_report=reflection_report if 'reflection_report' in locals() else None,
            reflection_triggered=reflection_triggered if 'reflection_triggered' in locals() else False,
            reflection_retry_used=reflection_retry_used if 'reflection_retry_used' in locals() else False,
            reflection_result=reflection_report.result.value if ('reflection_report' in locals() and reflection_report) else None,
            external_review_report=external_review_report if 'external_review_report' in locals() else None
        )
        
        report_path = self.reporter.generate_report(report, constraints=constraints if 'constraints' in locals() else [], repair_scope=repair_scope if 'repair_scope' in locals() else None, rollback_results=rollback_results if 'rollback_results' in locals() else {})
        logger.info(f"Task finished with status {final_status}. Report: {report_path}")
        
        return report_path
"""

with open("agent/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(prefix + clean_run)
