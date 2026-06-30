"""Phase 10 — Execution Governor (the shared leash for BOTH strategies).

Enforces the run's safety caps:
  - max steps / iterations
  - tool-call budget
  - per-run cost budget (reuses the Phase 7 cost telemetry values)
  - per-step + per-run timeouts
  - oscillation / no-progress detection (repeated-action fingerprints)

State lives on ``state.governor`` (:class:`GovernorState`) with a single
``stop_reason``. The agent loop calls :meth:`check_before_step` each iteration and
wraps each step in :meth:`step_timeout_seconds`; the pipeline engine reuses the
same cost/step accounting so one leash governs both engines.
"""

from __future__ import annotations

import time
from typing import Optional

from agent.config import logger, settings
from agent.state.agent_state import AgentState

# How many times an identical action fingerprint may repeat before we call it
# oscillation. A 10B enhancement can make this adaptive / progress-aware.
OSCILLATION_REPEAT = 3

# Stop-reason constants (stable identifiers used by tests + reports).
STOP_MAX_STEPS = "max_steps"
STOP_TOOL_BUDGET = "tool_call_budget"
STOP_COST_BUDGET = "cost_budget"
STOP_RUN_TIMEOUT = "run_timeout"
STOP_STEP_TIMEOUT = "step_timeout"
STOP_OSCILLATION = "oscillation"
STOP_MAX_REPLANS = "max_replans"
STOP_DONE = "done"
STOP_ERROR = "policy_error"


class Governor:
    def __init__(self, state: AgentState):
        self.state = state
        self._start = time.monotonic()

    @classmethod
    def configure(cls, state: AgentState) -> "Governor":
        """Apply the configured caps onto ``state.governor`` and return a governor."""
        g = state.governor
        g.max_steps = settings.max_steps
        g.tool_call_budget = settings.tool_call_budget
        g.run_budget_usd = settings.run_budget_usd
        g.step_timeout_seconds = settings.step_timeout_seconds
        g.run_timeout_seconds = settings.run_timeout_seconds
        g.max_replans = settings.max_replans
        return cls(state)

    @property
    def step_timeout(self) -> Optional[float]:
        t = self.state.governor.step_timeout_seconds
        return float(t) if t and t > 0 else None

    # --- caps ---------------------------------------------------------------

    def check_before_step(self) -> Optional[str]:
        """Return a stop reason if any cap is already exceeded, else None."""
        g = self.state.governor
        if g.stopped:
            return g.stop_reason
        if g.max_steps and g.steps_used >= g.max_steps:
            return self.stop(STOP_MAX_STEPS)
        if g.tool_call_budget and g.tool_calls_used >= g.tool_call_budget:
            return self.stop(STOP_TOOL_BUDGET)
        if g.run_budget_usd and g.cost_used_usd >= g.run_budget_usd:
            return self.stop(STOP_COST_BUDGET)
        if g.run_timeout_seconds and (time.monotonic() - self._start) >= g.run_timeout_seconds:
            return self.stop(STOP_RUN_TIMEOUT)
        return None

    def mark_step(self) -> None:
        self.state.governor.steps_used += 1

    def add_cost(self, usd: float) -> Optional[str]:
        g = self.state.governor
        g.cost_used_usd = round(g.cost_used_usd + max(0.0, usd), 6)
        if g.run_budget_usd and g.cost_used_usd >= g.run_budget_usd:
            return self.stop(STOP_COST_BUDGET)
        return None

    def note_replan(self) -> Optional[str]:
        """Account one replan against the governed bound (Phase 11).

        Returns a stop reason when the replan budget is exhausted (so the loop
        halts instead of replanning forever), else None and the attempt counts.
        ``max_replans == 0`` disables the cap (unbounded replanning)."""
        g = self.state.governor
        if g.max_replans and g.replans_used >= g.max_replans:
            return self.stop(STOP_MAX_REPLANS)
        g.replans_used += 1
        return None

    def note_progress(self, fingerprint: str) -> Optional[str]:
        """Record an action fingerprint; flag oscillation if it repeats too often."""
        g = self.state.governor
        g.progress_fingerprints.append(fingerprint)
        recent = g.progress_fingerprints[-OSCILLATION_REPEAT:]
        if len(recent) == OSCILLATION_REPEAT and len(set(recent)) == 1:
            return self.stop(STOP_OSCILLATION)
        return None

    def stop(self, reason: str) -> str:
        g = self.state.governor
        if not g.stopped:
            g.stopped = True
            g.stop_reason = reason
            logger.info(f"Governor: stopping run — reason={reason}")
            self.state.add_timeline("governor", f"stop: {reason}")
        return reason
