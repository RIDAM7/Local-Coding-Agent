"""Phase 16 — Merger.

Reconciles worker results back into the parent AgentState. The Coordinator
calls ``merge`` after all workers complete to combine files_modified, evidence,
observations, validation_results, and governor cost/usage from each worker
result into the parent state.
"""

from __future__ import annotations

from typing import List

from agent.orchestration.worker import WorkerResult
from agent.state.agent_state import AgentState, Evidence


class Merger:
    """Reconciles worker results into the parent AgentState.

    Each worker result slice is appended to the parent state. Duplicate file
    paths are deduplicated (last writer wins). Governor cost/usage is summed
    across all workers.
    """

    @staticmethod
    def merge(state: AgentState, results: List[WorkerResult]) -> AgentState:
        """Merge all worker results into the parent state."""
        file_map = {fc.path: fc for fc in state.files_modified}
        for result in results:
            # Files modified (last writer wins per path).
            for fc in result.files_modified:
                file_map[fc.path] = fc
        state.files_modified = list(file_map.values())

        for result in results:
            # Evidence.
            for ev in result.evidence:
                if isinstance(ev, dict):
                    state.evidence.append(Evidence(**ev))
                else:
                    state.evidence.append(ev)

            # Observations.
            for obs in result.observations:
                state.observations.append(obs)

            # Validation results.
            for vr in result.validation_results:
                state.validation_results.append(vr)

            # Governor cost/usage (summed).
            state.governor.steps_used += result.steps_used
            state.governor.tool_calls_used += result.tool_calls_used
            state.governor.cost_used_usd = round(
                state.governor.cost_used_usd + result.cost_used_usd, 6
            )
            state.governor.replans_used += result.replans_used

            # Timeline: record each worker's outcome.
            state.add_timeline(
                "orchestration",
                f"worker finished: {result.status} — {result.summary[:120]}",
            )

        return state
