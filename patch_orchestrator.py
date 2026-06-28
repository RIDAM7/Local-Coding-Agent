import re

with open("agent/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

# I need to wrap from "patch = await self.coder.generate_patch(task, plan, context)"
# all the way down to "review_decision = self.review_router.route(confidence_report)"

# Wait, instead of a massive indent, I can just write the while loop correctly.
# Let's read the lines.
lines = content.split('\n')

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "patch = await self.coder.generate_patch(task, plan, context)" in line and start_idx == -1:
        start_idx = i
    if "review_decision = self.review_router.route(confidence_report)" in line:
        end_idx = i

if start_idx != -1 and end_idx != -1:
    print(f"Start: {start_idx}, End: {end_idx}")
    
    # We will insert loop initialization before start_idx
    prefix = lines[:start_idx]
    
    loop_init = [
        "            MAX_CLAUDE_REPAIR_CYCLES = 1",
        "            claude_cycles = 0",
        "            external_review_report = None",
        "            ",
        "            while claude_cycles <= MAX_CLAUDE_REPAIR_CYCLES:"
    ]
    
    # Indent the body
    body = []
    for line in lines[start_idx:end_idx+1]:
        if line.strip() == "":
            body.append(line)
        else:
            body.append("    " + line)
            
    # Add the Claude Review check at the end of the loop
    review_logic = [
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
    
    suffix = lines[end_idx+1:]
    
    # Update report generation variables to pull from `external_review_report`
    for i, line in enumerate(suffix):
        if "external_review_report=external_review_report" in line:
            break
        if "reflection_result=" in line:
            suffix.insert(i+1, "            external_review_report=external_review_report,")
            break
            
    new_lines = prefix + loop_init + body + review_logic + suffix
    
    with open("agent/orchestrator.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
    print("Patched successfully!")
else:
    print("Could not find boundaries")
