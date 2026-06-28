import re

with open("agent/orchestrator.py", "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

plan_idx = -1
for i, line in enumerate(lines):
    if "# 1. Plan" in line:
        plan_idx = i
        break

report_end_idx = -1
for i, line in enumerate(lines):
    if "return report_path" in line:
        report_end_idx = i
        break

if plan_idx != -1 and report_end_idx != -1:
    prefix = lines[:plan_idx]
    body = lines[plan_idx:report_end_idx]
    suffix = lines[report_end_idx:]
    
    loop_init = [
        "            MAX_CLAUDE_REPAIR_CYCLES = 1",
        "            claude_cycles = 0",
        "            external_review_report = None",
        "",
        "            while claude_cycles <= MAX_CLAUDE_REPAIR_CYCLES:"
    ]
    
    indented_body = []
    for line in body:
        if line.strip():
            indented_body.append("    " + line)
        else:
            indented_body.append(line)
            
    # Inject Claude logic right before report path is generated, wait no, right before the loop ends
    claude_logic = """
                if review_decision.value == ReviewDecision.APPROVE.value:
                    logger.info("Confidence score >= 95. Bypassing Claude.")
                    break
                    
                logger.info(f"Review decision is {review_decision.value}. Calling ClaudeReviewer...")
                external_review_report = await self.claude_reviewer.review(
                    task=task,
                    plan=plan,
                    patch=patch if 'patch' in locals() else None,
                    validation_report=validation_report if 'validation_report' in locals() else None,
                    reflection_report=reflection_report if 'reflection_report' in locals() else None
                )
                
                # Update the generated report with Claude results by re-saving it
                report.external_review_report = external_review_report
                report_path = self.reporter.generate_report(report, constraints=constraints if 'constraints' in locals() else [], repair_scope=repair_scope if 'repair_scope' in locals() else None, rollback_results=rollback_results if 'rollback_results' in locals() else {})
                
                if review_decision.value == ReviewDecision.MANDATORY_REVIEW.value and claude_cycles < MAX_CLAUDE_REPAIR_CYCLES:
                    if external_review_report.issues:
                        logger.warning("Claude found issues during MANDATORY_REVIEW. Triggering 1 local repair cycle.")
                        claude_cycles += 1
                        issues_str = "\\n".join([f"- {i.category.value} ({i.severity.value}): {i.description}" for i in external_review_report.issues])
                        repair_task_desc = f"{task.description}\\n\\nCLAUDE EXTERNAL REVIEW FAILED:\\nSummary: {external_review_report.summary}\\nIssues:\\n{issues_str}\\n\\nPlease generate a new replacement patch addressing these issues."
                        task = Task(description=repair_task_desc)
                        continue
                    else:
                        logger.info("Claude found NO issues during MANDATORY_REVIEW. Proceeding.")
                        break
                else:
                    break
"""
    
    new_lines = prefix + loop_init + indented_body + claude_logic.split('\n') + suffix
    
    with open("agent/orchestrator.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
    print("Patched correctly!")
else:
    print("Could not find bounds")
