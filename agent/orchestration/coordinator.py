"""Phase 16 — Coordinator (MVP).

Owns all orchestration for a multi-part task: decomposition, worker assignment,
sequential scheduling, dependency ordering, result merging, and post-merge validation.
Workers remain stateless — the Coordinator holds the parent AgentState.
"""

from __future__ import annotations

from typing import List

from agent.config import logger, settings
from agent.engine.governor import Governor
from agent.orchestration.decomposer import decompose
from agent.orchestration.merger import Merger
from agent.orchestration.worker import Worker, WorkerResult, WorkerSpec
from agent.safety.controller import SafetyMode
from agent.session.checkpointer import Checkpointer
from agent.state.agent_state import AgentState, ValidationResult


class Coordinator:
    """Central orchestration owner for multi-concern tasks.

    Usage::

        coordinator = Coordinator(orchestrator=orch, safety_mode=mode)
        state = await coordinator.run(state)
    """

    def __init__(self, orchestrator=None, *, safety_mode: SafetyMode = None):
        self._orchestrator = orchestrator
        self._safety_mode = safety_mode or SafetyMode(auto_approve=True)

    async def run(self, state: AgentState) -> AgentState:
        """Decompose, schedule workers sequentially, merge, and validate."""
        if not settings.orchestration_enabled:
            state.add_timeline("orchestration", "orchestration disabled — skipping coordinator")
            return state

        state.add_timeline("orchestration", "coordinator started")
        governor = Governor.configure(state)
        self._apply_orchestration_budget(state)

        checkpointer = Checkpointer()
        if checkpointer.enabled:
            try:
                checkpointer._store.create(
                    task_description=state.user_request,
                    session_id=state.task.id,
                )
            except Exception as e:
                logger.warning(f"Coordinator: session seed failed (non-fatal): {e}")

        specs = self._assign(decompose(state))
        roles = [s.role for s in specs]
        state.add_timeline("orchestration", f"decomposed into {len(specs)} worker(s): {roles}")

        results = await self._schedule(specs, state, governor, checkpointer)
        validation_ok = await self._validate(state)
        self._finalize(state, results, validation_ok)

        if checkpointer.enabled:
            checkpointer.checkpoint_step(state, len(results))

        state.add_timeline(
            "orchestration",
            f"coordinator finished: status={state.final_outputs.status}",
        )
        return state

    @staticmethod
    def _apply_orchestration_budget(state: AgentState) -> None:
        """Apply ORCHESTRATION_BUDGET_USD when set (shares the governor leash)."""
        budget = settings.orchestration_budget_usd
        if budget and budget > 0:
            g = state.governor
            if g.run_budget_usd == 0 or budget < g.run_budget_usd:
                g.run_budget_usd = budget

    @staticmethod
    def _assign(specs: List[WorkerSpec]) -> List[WorkerSpec]:
        """Worker assignment hook — MVP passes specs through unchanged."""
        return specs

    async def _schedule(
        self,
        specs: List[WorkerSpec],
        state: AgentState,
        governor: Governor,
        checkpointer: Checkpointer,
    ) -> List[WorkerResult]:
        """Run workers one at a time in dependency order (sequential MVP)."""
        if settings.max_parallel_workers > 1:
            logger.info(
                "MAX_PARALLEL_WORKERS=%s requested; MVP uses sequential scheduling only",
                settings.max_parallel_workers,
            )

        results: List[WorkerResult] = []
        for index, spec in enumerate(specs):
            stop = governor.check_before_step()
            if stop:
                state.add_timeline(
                    "orchestration",
                    f"global budget halt before worker [{spec.role}]: {stop}",
                )
                break

            if checkpointer.enabled:
                checkpointer.checkpoint_step(state, index)

            logger.info("Coordinator: dispatching worker [%s]", spec.role)
            state.add_timeline("orchestration", f"worker started: {spec.role}")
            result = await Worker.run(
                spec,
                safety_mode=self._safety_mode,
                incremental=settings.incremental_planning,
            )
            results.append(result)
            Merger.merge(state, [result])
            status = "ok" if result.success else "FAILED"
            state.add_timeline(
                "orchestration",
                f"worker [{spec.role}] finished — {status}",
            )
            if not result.success:
                logger.warning("Coordinator: worker [%s] failed: %s", spec.role, result.summary)

        return results

    async def _validate(self, state: AgentState) -> bool:
        """Run existing build/lint/test validators over the merged result."""
        orch = self._orchestrator
        if orch is None:
            state.add_timeline("orchestration", "validation skipped (no orchestrator)")
            return True

        try:
            build_res = await orch.build_validator.validate()
            lint_res = await orch.lint_validator.validate()
            test_res = await orch.test_validator.validate()
        except Exception as e:
            logger.warning("Coordinator: validation failed: %s", e)
            state.validation_results.append(
                ValidationResult(stage="BUILD", success=False, detail=str(e))
            )
            state.add_timeline("validation", f"orchestration validation error: {type(e).__name__}")
            return False

        ok = True
        for stage, res in (("BUILD", build_res), ("LINT", lint_res), ("TEST", test_res)):
            success = res.success
            detail = res.stderr[:200] if not success and getattr(res, "stderr", None) else ""
            state.validation_results.append(
                ValidationResult(stage=stage, success=success, detail=detail)
            )
            if not success:
                ok = False

        state.add_timeline(
            "validation",
            f"orchestration validation: {'passed' if ok else 'failed'}",
        )
        return ok

    @staticmethod
    def _finalize(
        state: AgentState,
        results: List[WorkerResult],
        validation_ok: bool,
    ) -> None:
        """Set final_outputs from worker + validation outcomes."""
        workers_ok = bool(results) and all(r.success for r in results)
        success = workers_ok and validation_ok
        summaries = [r.summary for r in results if r.summary]
        summary = "; ".join(summaries[:3]) if summaries else "orchestration complete"
        if not validation_ok:
            summary = f"{summary} (validation failed)" if summary else "validation failed"

        state.final_outputs.status = "SUCCESS" if success else "FAILURE"
        state.final_outputs.summary = summary
        state.final_outputs.applied_changes = [fc.path for fc in state.files_modified]
