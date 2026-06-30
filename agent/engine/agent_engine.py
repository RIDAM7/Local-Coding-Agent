"""Phase 10 — Agent strategy (governed Reason -> Action -> Observe loop).

The agent loop asks a tool-calling/structured-output model for the next action,
dispatches it through the tool executor (every write routes through safety + jail),
observes the result on the shared :class:`AgentState`, and repeats until the model
signals done or the :class:`Governor` stops the run. Completion uses the Round 1
confidence engine; the provider-agnostic :class:`Reviewer` runs ONLY as a
last-resort escalation when the governor exhausts iterations with low confidence
and ``REVIEWER_ENABLED`` is set.

10B (stubbed, not built here): stronger Thought->Action format adherence + retries,
streaming, smarter heuristics, a richer tool set. The seams are clean: swap the
policy client / extend the registry without touching the loop.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from agent.config import logger, settings
from agent.engine import governor as gov
from agent.engine.base import ExecutionEngine
from agent.engine.governor import Governor
from agent.execution.core import Executor
from agent.files.core import FileManager
from agent.git.core import GitManager
from agent.review.confidence import ConfidenceEngine
from agent.safety.controller import SafetyController, SafetyMode
from agent.state.agent_state import AgentState
from agent.tools import ToolExecutor, build_default_registry
from agent.validation import BuildValidator, LintValidator, TestValidator
from agent.retrieval import RipgrepSearch, SymbolIndex, TreeSitterIndexer


class AgentAction(BaseModel):
    """The structured action the policy model returns each turn."""
    thought: str = ""
    tool: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    final_summary: str = ""


class AgentEngine(ExecutionEngine):
    name = "agent"

    def __init__(self, workspace_path=None, *, safety_mode: SafetyMode = None,
                 safety_controller: SafetyController = None, policy_client=None,
                 memory_manager=None, incremental: bool = False):
        self.workspace = Path(workspace_path) if workspace_path else settings.get_workspace_path()
        self.safety = safety_controller or SafetyController(safety_mode or SafetyMode(auto_approve=True))
        self._policy_client = policy_client          # injected in tests; built lazily otherwise
        self.memory_manager = memory_manager
        # Phase 11: incremental planning (plan -> execute step -> observe -> replan).
        # Defaults False here so the Phase 10 free-form loop (and its tests) are
        # unchanged on direct construction; the CLI passes settings.incremental_planning.
        self.incremental = incremental

        # Wire the tools to the existing Round 1 modules (no logic copied).
        self.file_manager = FileManager(self.workspace)
        self.executor = Executor(self.workspace, dry_run=self.safety.mode.dry_run)
        self.git_manager = GitManager(self.workspace)
        self.rg = RipgrepSearch()
        self.sym_idx = SymbolIndex(TreeSitterIndexer())
        self.index_dir = str(self.workspace / "index")
        self.build_validator = BuildValidator(self.executor)
        self.lint_validator = LintValidator(self.executor)
        self.test_validator = TestValidator(self.executor)
        self.confidence_engine = ConfidenceEngine()

        self.registry = build_default_registry(
            file_manager=self.file_manager, executor=self.executor, safety=self.safety,
            rg=self.rg, sym_idx=self.sym_idx, index_dir=self.index_dir, workspace=self.workspace,
            git_manager=self.git_manager, build_validator=self.build_validator,
            lint_validator=self.lint_validator, test_validator=self.test_validator,
            memory_manager=self.memory_manager,
        )
        self.tool_executor = ToolExecutor(self.registry)

    @property
    def policy_client(self):
        if self._policy_client is None:
            from agent.llm.factory import build_client
            self._policy_client = build_client("planner")
        return self._policy_client

    @property
    def model(self) -> str:
        return getattr(self.policy_client, "model", None) or settings.planner_model

    # --- main loop ----------------------------------------------------------

    async def execute(self, state: AgentState) -> AgentState:
        state.execution_mode = "agent"
        if self.incremental:
            return await self._execute_incremental(state)
        state.add_timeline("engine", "agent strategy selected (Think->Act->Observe)")
        governor = Governor.configure(state)

        while True:
            reason = governor.check_before_step()
            if reason:
                break
            governor.mark_step()
            try:
                stop = await asyncio.wait_for(self._run_one_step(state, governor),
                                              timeout=governor.step_timeout)
            except asyncio.TimeoutError:
                governor.stop(gov.STOP_STEP_TIMEOUT)
                break
            except Exception as e:
                # A policy/model failure ends the run cleanly (no crash). Robust
                # retries/backoff are a 10B enhancement.
                logger.warning(f"AgentEngine: step failed, stopping: {e}")
                state.add_observation(f"step error: {type(e).__name__}: {e}")
                governor.stop(gov.STOP_ERROR)
                break
            if stop:
                break

        self._finalize(state)
        await self._maybe_escalate(state)
        return state

    async def _run_one_step(self, state: AgentState, governor: Governor) -> bool:
        """Run one Reason->Act->Observe step. Returns True if the loop should stop."""
        action = await self._next_action(state)
        state.add_observation(f"thought: {action.thought}" if action.thought else "thought: (none)")

        if action.done or not action.tool:
            state.final_outputs.summary = action.final_summary or "agent reported done"
            governor.stop(gov.STOP_DONE)
            return True

        # Oscillation guard on identical repeated actions.
        fingerprint = f"{action.tool}:{json.dumps(action.args, sort_keys=True, default=str)}"
        if governor.note_progress(fingerprint):
            return True

        result = await self.tool_executor.dispatch(state, action.tool, action.args)
        state.add_observation(f"{action.tool} -> [{result.status}] {result.summary}")
        return False

    async def _next_action(self, state: AgentState) -> AgentAction:
        tool_lines = "\n".join(f"- {s.name}: {s.description}" for s in self.registry.specs())
        recent = "\n".join(o.note for o in state.observations[-8:])
        prompt = (
            "You are an autonomous coding agent. Decide the SINGLE next action.\n"
            f"User request:\n{state.user_request}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            f"Recent observations:\n{recent or '(none yet)'}\n\n"
            "Return JSON: {\"thought\": str, \"tool\": str, \"args\": object, "
            "\"done\": bool, \"final_summary\": str}. Set done=true when the task is "
            "complete; otherwise pick exactly one tool with its args."
        )
        result = await self.policy_client.generate_structured(self.model, prompt, AgentAction)
        return result.data

    # --- incremental planning (Phase 11) -----------------------------------

    async def _execute_incremental(self, state: AgentState) -> AgentState:
        """Run the shared plan -> execute step -> observe -> replan loop, with each
        step executed via this engine's governed tool loop."""
        from agent.planning import IncrementalPlanner, Replanner, StepPlanner

        state.add_timeline("engine", "agent strategy selected (incremental planning)")
        governor = Governor.configure(state)
        inc = IncrementalPlanner(
            step_planner=StepPlanner(client=self.policy_client),
            replanner=Replanner(client=self.policy_client) if settings.replan_on_failure else None,
        )
        await inc.run(state, self._step_executor, governor=governor,
                      context_bundle=state.loaded_context)
        self._score_confidence(state)
        state.add_timeline("engine", f"agent finished: status={state.final_outputs.status} "
                                     f"confidence={state.confidence} stop={state.governor.stop_reason}")
        await self._maybe_escalate(state)
        return state

    async def _step_executor(self, state: AgentState, step) -> "StepOutcome":
        """Execute ONE plan step via a bounded inner tool loop driven by the
        policy model. Returns the step outcome for the Observer to evaluate."""
        from agent.planning import StepOutcome

        # Per-step tool-call cap keeps a single step from running away; the global
        # governor still bounds the whole run (steps / cost / timeouts).
        max_inner = max(1, settings.max_steps)
        for _ in range(max_inner):
            stop = self.state_governor_check(state)
            if stop:
                return StepOutcome(success=False, summary=f"governor stop: {stop}")
            action = await self._next_step_action(state, step)
            if action.thought:
                state.add_observation(f"thought: {action.thought}")
            if action.done or not action.tool:
                return StepOutcome(success=True,
                                   summary=action.final_summary or f"step {step.index} complete")
            result = await self.tool_executor.dispatch(state, action.tool, action.args)
            state.add_observation(f"{action.tool} -> [{result.status}] {result.summary}")
            if result.status in ("error", "blocked"):
                return StepOutcome(success=False, summary=f"{action.tool} {result.status}: {result.summary}")
        return StepOutcome(success=True, summary=f"step {step.index}: reached per-step cap")

    @staticmethod
    def state_governor_check(state: AgentState) -> Optional[str]:
        g = state.governor
        if g.tool_call_budget and g.tool_calls_used >= g.tool_call_budget:
            return gov.STOP_TOOL_BUDGET
        return None

    async def _next_step_action(self, state: AgentState, step) -> AgentAction:
        tool_lines = "\n".join(f"- {s.name}: {s.description}" for s in self.registry.specs())
        recent = "\n".join(o.note for o in state.observations[-8:])
        prompt = (
            "You are an autonomous coding agent working through a plan ONE step at a time.\n"
            f"Overall request:\n{state.user_request}\n\n"
            f"CURRENT STEP (do only this):\n{step.description}\n"
            f"Acceptance check for this step:\n{step.acceptance or '(none)'}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            f"Recent observations:\n{recent or '(none yet)'}\n\n"
            "Return JSON: {\"thought\": str, \"tool\": str, \"args\": object, "
            "\"done\": bool, \"final_summary\": str}. Set done=true ONLY when THIS step "
            "is complete; otherwise pick exactly one tool with its args to advance it."
        )
        result = await self.policy_client.generate_structured(self.model, prompt, AgentAction)
        return result.data

    # --- completion + escalation -------------------------------------------

    def _score_confidence(self, state: AgentState) -> None:
        """Compute the Round 1 confidence score from the signals on ``state``.
        Shared by the free-form loop and the incremental loop."""
        def last(stage: str) -> Optional[bool]:
            for v in reversed(state.validation_results):
                if v.stage == stage:
                    return v.success
            return None

        build_ok = last("BUILD")
        test_ok = last("TEST")
        lint_ok = last("LINT")
        report = self.confidence_engine.evaluate(
            build_success=True if build_ok is None else build_ok,
            test_success=True if test_ok is None else test_ok,
            lint_success=True if lint_ok is None else lint_ok,
            repair_attempts=len(state.repair_attempts),
            memory_count=len(state.memory_refs.summaries),
            files_modified_count=len(state.files_modified),
            plan_step_count=len(state.plan.steps),
            constraint_violations=0,
        )
        state.confidence = round(report.confidence_score / 100.0, 4)

    def _finalize(self, state: AgentState) -> None:
        self._score_confidence(state)

        stopped_clean = state.governor.stop_reason == gov.STOP_DONE
        any_failure = any(not v.success for v in state.validation_results)
        state.final_outputs.status = "SUCCESS" if (stopped_clean and not any_failure) else "FAILURE"
        state.final_outputs.applied_changes = [f.path for f in state.files_modified]
        state.add_timeline("engine", f"agent finished: status={state.final_outputs.status} "
                                     f"confidence={state.confidence} stop={state.governor.stop_reason}")

    async def _maybe_escalate(self, state: AgentState) -> None:
        """Last-resort guided reviewer pass (replaces the old Claude/arbitration)."""
        exhausted = state.governor.stop_reason not in (gov.STOP_DONE, None)
        if not (exhausted and settings.reviewer_enabled
                and state.confidence < settings.confidence_threshold):
            return
        try:
            from agent.reviewers.reviewer import Reviewer
            reviewer = Reviewer()
            report = await reviewer.review(
                task=type("T", (), {"description": state.user_request}),
                plan=None, patch=None, validation_report=state.validation_results,
                reflection_report=None)
            state.add_observation(f"reviewer escalation: {report.summary}")
            state.add_timeline("engine", f"reviewer escalation ({report.review_status})")
        except Exception as e:
            logger.warning(f"AgentEngine: reviewer escalation failed: {e}")
