import re

with open("agent/orchestrator.py", "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

# Find bounds
start_idx = -1
for i, line in enumerate(lines):
    if "patch = await self.coder.generate_patch(task, plan, context)" in line and "if" not in line and start_idx == -1:
        start_idx = i

end_idx = -1
for i, line in enumerate(lines):
    if "review_decision = self.review_router.route(confidence_report)" in line:
        end_idx = i

if start_idx != -1 and end_idx != -1:
    print(f"Start: {start_idx}, End: {end_idx}")
    
    new_lines = lines[:start_idx]
    
    loop_init = [
        "            MAX_CLAUDE_REPAIR_CYCLES = 1",
        "            claude_cycles = 0",
        "            external_review_report = None",
        "",
        "            while claude_cycles <= MAX_CLAUDE_REPAIR_CYCLES:"
    ]
    
    new_lines.extend(loop_init)
    
    # Indent body
    for line in lines[start_idx:end_idx+1]:
        if line.strip() == "":
            new_lines.append(line)
        else:
            new_lines.append("    " + line)
            
    # Add claude logic
    claude_logic = [
        "                if review_decision == ReviewDecision.APPROVE:",
        "                    logger.info('Confidence score >= 95. Bypassing Claude.')",
        "                    break",
        "                ",
        "                logger.info(f'Review decision is {review_decision.value}. Calling ClaudeReviewer...')",
        "                external_review_report = await self.claude_reviewer.review(",
        "                    task=task,",
        "                    plan=plan,",
        "                    patch=patch,",
        "                    validation_report=validation_report if 'validation_report' in locals() else None,",
        "                    reflection_report=reflection_report if 'reflection_report' in locals() else None",
        "                )",
        "                ",
        "                if review_decision == ReviewDecision.MANDATORY_REVIEW and claude_cycles < MAX_CLAUDE_REPAIR_CYCLES:",
        "                    if external_review_report.issues:",
        "                        logger.warning('Claude found issues during MANDATORY_REVIEW. Triggering 1 local repair cycle.')",
        "                        claude_cycles += 1",
        "                        ",
        "                        issues_str = '\\n'.join([f'- {i.category.value} ({i.severity.value}): {i.description}' for i in external_review_report.issues])",
        "                        repair_task_desc = f'{task.description}\\n\\nCLAUDE EXTERNAL REVIEW FAILED:\\nSummary: {external_review_report.summary}\\nIssues:\\n{issues_str}\\n\\nPlease generate a new replacement patch addressing these issues.'",
        "                        task = Task(description=repair_task_desc)",
        "                        continue",
        "                    else:",
        "                        logger.info('Claude found NO issues during MANDATORY_REVIEW. Proceeding.')",
        "                        break",
        "                else:",
        "                    break"
    ]
    
    new_lines.extend(claude_logic)
    
    # Append the rest
    new_lines.extend(lines[end_idx+1:])
    
    with open("agent/orchestrator.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
    print("Patched successfully")
else:
    print("Could not find start/end")
