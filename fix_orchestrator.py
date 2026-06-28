import re

with open("agent/orchestrator.py", "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

# Find the loop start
start_idx = -1
for i, line in enumerate(lines):
    if "MAX_CLAUDE_REPAIR_CYCLES = 1" in line:
        start_idx = i
        break

if start_idx != -1:
    # Delete the loop init
    # The loop init is 5 lines
    new_lines = lines[:start_idx]
    
    # We outdent the body
    # The body goes until the if review_decision == ReviewDecision.APPROVE:
    end_idx = -1
    for i in range(start_idx + 5, len(lines)):
        if "if review_decision == ReviewDecision.APPROVE:" in lines[i]:
            end_idx = i
            break
            
    if end_idx != -1:
        # Outdent body
        for line in lines[start_idx+5:end_idx]:
            if line.startswith("    "):
                new_lines.append(line[4:])
            else:
                new_lines.append(line)
                
        # Skip the claude logic we added
        skip_end = end_idx
        for i in range(end_idx, len(lines)):
            if "else:" in lines[i] and "break" in lines[i+1]:
                skip_end = i + 2
                break
                
        new_lines.extend(lines[skip_end:])
        
        with open("agent/orchestrator.py", "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
        print("Reverted successfully")
    else:
        print("Could not find end")
else:
    print("Could not find start")
