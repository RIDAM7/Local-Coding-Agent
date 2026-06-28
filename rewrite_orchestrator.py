import re

with open("agent/orchestrator.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Find everything before Code patch generation
start_pattern = r"(# 3\. Code\s+patch = await self\.coder\.generate_patch\(task, plan, context\))"
parts = re.split(start_pattern, text, 1)

if len(parts) == 3:
    prefix = parts[0]
    rest = parts[1] + parts[2]
    
    # 2. In `rest`, find where the `except AgentError as e:` begins
    except_pattern = r"(        except AgentError as e:)"
    rest_parts = re.split(except_pattern, rest, 1)
    
    if len(rest_parts) == 3:
        inner_body = rest_parts[0]
        suffix = rest_parts[1] + rest_parts[2]
        
        # 3. In `suffix`, find the confidence calculation and router logic
        # and move it to `inner_body`!
        conf_start = suffix.find("val_rep = validation_report")
        conf_end = suffix.find("        report = Report(")
        
        if conf_start != -1 and conf_end != -1:
            conf_logic = suffix[conf_start:conf_end]
            suffix = suffix[:conf_start] + suffix[conf_end:]
            
            # Now `inner_body` gets indented 4 spaces and wrapped in `while` loop
            inner_lines = inner_body.split('\n')
            conf_lines = conf_logic.split('\n')
            
            indented_inner = []
            for line in inner_lines:
                if line.strip(): indented_inner.append("    " + line)
                else: indented_inner.append(line)
                
            indented_conf = []
            for line in conf_lines:
                if line.strip(): indented_conf.append("    " + line)
                else: indented_conf.append(line)
            
            loop_header = """        MAX_CLAUDE_REPAIR_CYCLES = 1
        claude_cycles = 0
        external_review_report = None
        
        while claude_cycles <= MAX_CLAUDE_REPAIR_CYCLES:
"""
            
            claude_logic = """
            if review_decision == ReviewDecision.APPROVE:
                logger.info("Confidence score >= 95. Bypassing Claude.")
                break
            
            logger.info(f"Review decision is {review_decision.value}. Calling ClaudeReviewer...")
            external_review_report = await self.claude_reviewer.review(
                task=task,
                plan=plan,
                patch=patch,
                validation_report=validation_report if 'validation_report' in locals() else None,
                reflection_report=reflection_report if 'reflection_report' in locals() else None
            )
            
            if review_decision == ReviewDecision.MANDATORY_REVIEW and claude_cycles < MAX_CLAUDE_REPAIR_CYCLES:
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
            
            final_text = prefix + loop_header + "\n".join(indented_inner) + "\n".join(indented_conf) + claude_logic + "\n" + suffix
            
            with open("agent/orchestrator.py", "w", encoding="utf-8") as out:
                out.write(final_text)
            print("Successfully rewritten orchestrator.py!")
        else:
            print("Could not find confidence logic")
    else:
        print("Could not find except block")
else:
    print("Could not find start pattern")
