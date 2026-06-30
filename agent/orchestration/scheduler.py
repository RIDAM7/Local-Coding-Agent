"""Phase 16 — Sequential Scheduler (MVP).

Default scheduling is sequential: one worker runs at a time, in dependency order.
This is the offline-safe default — no parallelism. The scheduler exists as a
seam for the 16B enhancement (bounded-parallel scheduling) without changing the
Coordinator.
"""

from __future__ import annotations

from typing import List

from agent.config import logger
from agent.orchestration.worker import WorkerSpec, WorkerResult, Worker


class Scheduler:
    """Sequential scheduler: runs workers one at a time in the given order."""

    def __init__(self, parallel: bool = False):
        self._parallel = parallel

    async def run(self, specs: List[WorkerSpec]) -> List[WorkerResult]:
        """Execute all specs sequentially in order. Returns results in the same order.

        Each worker runs to completion (or governor stop) before the next
        worker begins. A failed worker does not block subsequent workers
        unless the Coordinator chooses to abort.
        """
        results: List[WorkerResult] = []
        for spec in specs:
            logger.info(f"Scheduler: running worker [{spec.role}]")
            result = await Worker.run(spec)
            results.append(result)
            status = "ok" if result.success else "FAILED"
            logger.info(f"Scheduler: worker [{spec.role}] finished — {status}")
        return results
