from pathlib import Path
import os
from datetime import datetime, timezone
from agent.models.schemas import Report
from agent.config import logger, settings
from agent.safety.redact import redact

class Reporter:
    def __init__(self, reports_dir: Path = None):
        self.reports_dir = str(reports_dir) if reports_dir else "reports"
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_report(self, report: Report, constraints: list = None, repair_scope = None, rollback_results: dict = None, refinement = None, raw_task: str = None, plan_evolution: str = None) -> str:
        logger.info("Generating markdown report...")
        
        report_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.reports_dir, f"report_{report_id}.md")
        json_filepath = os.path.join(self.reports_dir, f"report_{report_id}.json")
        
        # Save JSON (Phase 5: redact secrets so nothing sensitive ever lands in a report).
        with open(json_filepath, 'w', encoding='utf-8') as f:
            f.write(redact(report.model_dump_json(indent=2)))
            
        content = [
            f"# Execution Report: {report_id}\n",
            f"**Task:** {report.task}\n",
            f"**Status:** {report.final_status}\n",
            f"**Execution Results:** {report.execution_results}\n\n"
        ]

        # Phase 3: when the refiner ran, show the raw prompt alongside the refined
        # rewrite so the user can audit exactly what was changed before planning.
        if refinement:
            content.append("## Prompt Refinement\n")
            content.append("**Raw Prompt:**\n")
            content.append(f"> {raw_task}\n\n" if raw_task else "> (unavailable)\n\n")
            content.append(f"**Refined Task:**\n> {refinement.refined_task}\n\n")
            content.append(f"**Clarified Goal:** {refinement.clarified_goal}\n\n")
            if refinement.assumptions:
                content.append("**Assumptions:**\n")
                for a in refinement.assumptions:
                    content.append(f"- {a}\n")
                content.append("\n")
            if refinement.acceptance_criteria:
                content.append("**Acceptance Criteria:**\n")
                for c in refinement.acceptance_criteria:
                    content.append(f"- {c}\n")
                content.append("\n")
            if refinement.open_questions:
                content.append("**Open Questions:**\n")
                for q in refinement.open_questions:
                    content.append(f"- {q}\n")
                content.append("\n")

        if constraints:
            content.append("## Task Constraints\n")
            for c in constraints:
                content.append(f"- **{c.type}**: `{', '.join(c.patterns) if c.patterns else ''}`\n")
            content.append("\n")
            
        if repair_scope:
            content.append("## Repair Scope\n")
            for p in repair_scope.allowed_paths:
                content.append(f"- `{p}`\n")
            content.append("\n")
        
        if report.plan:
            content.append(f"## Plan\n**Goal:** {report.plan.goal}\n\n**Summary:** {report.plan.summary}\n\n### Steps\n")
            for step in report.plan.steps:
                content.append(f"{step.id}. {step.description}\n   *Expected Output:* {step.expected_output}\n")
            content.append("\n")

        # Phase 11: incremental planning — show the plan evolution (original plan
        # -> revisions). Only present when the incremental path ran; the Round 1
        # report is byte-for-byte unchanged when this is None.
        if plan_evolution:
            content.append(plan_evolution)
            content.append("\n\n")

        if settings.observability_enabled and report.observability:
            try:
                from agent.observability import render_report_view
                content.append(render_report_view(report.observability))
                content.append("\n")
            except Exception as e:
                logger.warning(f"Observability report rendering failed: {e}")

        content.append("## Retrieved Context\n")
        if report.retrieved_files:
            content.append("**Files:**\n")
            for f in report.retrieved_files:
                content.append(f"- {f}\n")
            content.append("\n**Symbols:**\n")
            for s in set(report.retrieved_symbols):
                content.append(f"- {s}\n")
        else:
            content.append("No context retrieved.\n")
        content.append("\n")
            
        if report.validation_report:
            vr = report.validation_report
            content.append("## Validation Results\n")
            
            if vr.patch_validation:
                content.append("### Patch Validation\n")
                content.append(f"**Valid:** {vr.patch_validation.is_valid}\n")
                if vr.patch_validation.errors:
                    content.append("**Errors:**\n")
                    for e in vr.patch_validation.errors:
                        content.append(f"- {e}\n")
                if vr.patch_validation.warnings:
                    content.append("**Warnings (Auto-repaired):**\n")
                    for w in vr.patch_validation.warnings:
                        content.append(f"- {w}\n")
                content.append("\n")
                
            for stage_name, res in [("Build", vr.build_result), ("Lint", vr.lint_result), ("Test", vr.test_result)]:
                if res:
                    content.append(f"### {stage_name} Validation\n")
                    content.append(f"**Command:** `{res.command}`\n")
                    content.append(f"**Status:** {'SUCCESS' if res.success else 'FAILED'}\n")
                    content.append(f"**Duration:** {res.duration:.2f}s\n")
                    if res.stdout and not res.stdout.startswith("Skipped"):
                        content.append(f"\n**Stdout:**\n```\n{res.stdout}\n```\n")
                    elif res.stdout.startswith("Skipped"):
                        content.append(f"\n{res.stdout}\n")
                    if res.stderr:
                        content.append(f"\n**Stderr:**\n```\n{res.stderr}\n```\n")
                    content.append("\n")
                    
        if report.repair_metrics and report.repair_metrics.total_attempts > 0:
            rm = report.repair_metrics
            content.append("## Repair Summary\n")
            content.append(f"**Total Attempts:** {rm.total_attempts}\n")
            content.append(f"**Resolved In Attempt:** {rm.resolved_in_attempt if rm.resolved_in_attempt else 'N/A'}\n")
            content.append(f"**Rollback Triggered:** {rm.rollback_triggered}\n\n")
            
            for i, result in enumerate(report.repair_history, 1):
                content.append(f"### Attempt {result.attempt_number}\n")
                content.append(f"**Classification:** `{result.classification}`\n")
                if result.patch_applied:
                    content.append(f"**Confidence:** {result.patch_applied.confidence}\n")
                    content.append(f"**Explanation:** {result.patch_applied.explanation}\n")
                content.append(f"**Outcome:** {'SUCCESS' if result.success else 'FAILED'}\n\n")
            
        content.append("## Files Modified\n")
        if report.files_modified:
            for f in report.files_modified:
                content.append(f"- {f}\n")
        else:
            content.append("No files modified.\n")
        content.append("\n")
        
        content.append("## Commands\n")
        if report.commands_executed:
            content.append("**Executed:**\n\n")
            for cmd in report.commands_executed:
                content.append(f"### `{cmd.command}`\n")
                content.append(f"- **Exit Code:** {cmd.exit_code}\n")
                content.append(f"- **Duration:** {cmd.duration}s\n")
                if cmd.stdout:
                    content.append(f"**Stdout:**\n```\n{cmd.stdout}\n```\n")
                if cmd.stderr:
                    content.append(f"**Stderr:**\n```\n{cmd.stderr}\n```\n")
        elif report.proposed_commands:
            # EXECUTE_COMMANDS is off: surface what the coder proposed but make it
            # unambiguous that nothing was run.
            content.append("**Proposed (NOT executed — set `EXECUTE_COMMANDS=true` to run):**\n")
            for cmd in report.proposed_commands:
                content.append(f"- `{cmd}`\n")
        else:
            content.append("No commands.\n")
        # Phase 5: commands refused by the hard safety denylist (never run).
        if report.blocked_commands:
            content.append("\n**Blocked by safety denylist (NOT executed):**\n")
            for cmd in report.blocked_commands:
                content.append(f"- `{cmd}`\n")
        content.append("\n")
        
        # Phase 7C: per-role token usage + estimated cost.
        if report.cost_summary and report.cost_summary.per_role:
            cs = report.cost_summary
            content.append("## Cost & Token Usage\n")
            content.append("| Role | Provider | Model | Input | Output | Est. Cost (USD) |\n")
            content.append("|------|----------|-------|------:|-------:|----------------:|\n")
            for r in cs.per_role:
                content.append(f"| {r.role} | {r.provider} | {r.model} | {r.input_tokens} | {r.output_tokens} | ${r.est_cost:.6f} |\n")
            content.append(f"| **Total** |  |  | **{cs.total_input_tokens}** | **{cs.total_output_tokens}** | **${cs.total_est_cost:.6f}** |\n\n")

        # Phase 7B: git integration outcome.
        if report.git_branch:
            content.append("## Git\n")
            content.append(f"- **Branch:** `{report.git_branch}`\n")
            if report.git_commit:
                content.append(f"- **Commit:** `{report.git_commit}`\n")
            else:
                content.append("- **Commit:** none (no changes committed)\n")
            content.append("\n")

        content.append(f"## Execution Results\n{report.execution_results}\n")
        
        if rollback_results:
            content.append("\n## Rollback Verification\n")
            for fp, success in rollback_results.items():
                icon = "✓ Restored" if success else "✗ Hash mismatch"
                content.append(f"- `{fp}`: {icon}\n")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(redact("".join(content)))
            logger.info(f"Report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to write report to {filepath}: {e}")
            
        return filepath
