"""Phase 16 — Decomposer (MVP).

Analyzes a task and its context to produce an ordered list of WorkerSpecs.
In the MVP, decomposition is based on keywords and repository structure
detected by the Context Engine (Phase 9) and Repository Graph (Phase 13).
A future enhancement will use the planner model for semantic decomposition.
"""

from __future__ import annotations

from typing import Dict, List

from agent.orchestration.worker import WorkerSpec
from agent.state.agent_state import AgentState


# Keywords that hint a task involves specific concerns.
_ROLE_KEYWORDS: Dict[str, List[str]] = {
    "backend": ["api", "endpoint", "route", "server", "database", "db", "migration",
                 "sql", "query", "model", "schema", "graphql", "rest"],
    "frontend": ["ui", "frontend", "component", "react", "vue", "angular", "page",
                  "style", "css", "html", "template", "view"],
    "test": ["test", "spec", "assertion", "pytest", "unittest", "jest", "mocha",
              "coverage", "testing"],
    "docs": ["doc", "readme", "documentation", "comment", "api doc", "swagger",
              "markdown"],
}


def _detect_roles(task_description: str) -> List[str]:
    """Detect which roles a task touches, in priority order."""
    lower = task_description.lower()
    detected = []
    for role, keywords in _ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                detected.append(role)
                break
    # Default: if nothing specific, treat as backend.
    return detected or ["backend"]


def _build_sub_tasks(task_description: str, roles: List[str]) -> List[str]:
    """Build scoped sub-task descriptions from the original task and detected roles.

    The MVP splits by concern; a future enhancement will use the planner model
    for semantic decomposition.
    """
    if len(roles) <= 1:
        return [task_description]
    # Split into one sub-task per role, scoping each.
    sub_tasks = []
    for role in roles:
        if role == "test":
            sub_tasks.append(
                f"{task_description}\n\nFocus on writing and running tests. "
                f"Only add test code."
            )
        elif role == "frontend":
            sub_tasks.append(
                f"{task_description}\n\nFocus on UI and front-end code. "
                f"Only add frontend components and styles."
            )
        elif role == "docs":
            sub_tasks.append(
                f"{task_description}\n\nFocus on documentation. "
                f"Only add docstrings, comments, and markdown files."
            )
        else:  # backend (or default)
            sub_tasks.append(
                f"{task_description}\n\nFocus on backend logic, APIs, and data."
            )
    return sub_tasks


def decompose(state: AgentState) -> List[WorkerSpec]:
    """Decompose the current task into an ordered list of WorkerSpecs.

    Uses the task description, loaded context, and repository graph evidence
    to determine the sub-task decomposition. Returns specs in dependency order
    (e.g. backend before frontend, backend before tests).
    """
    task = state.objective or state.user_request
    roles = _detect_roles(task)
    sub_tasks = _build_sub_tasks(task, roles)

    # Dependency order: backend -> frontend -> test -> docs
    _ORDER = {"backend": 0, "frontend": 1, "test": 2, "docs": 3}
    ordered = sorted(zip(roles, sub_tasks), key=lambda x: _ORDER.get(x[0], 99))
    roles, sub_tasks = zip(*ordered) if ordered else (["backend"], [task])

    specs = []
    for role, sub in zip(roles, sub_tasks):
        context_bundle = state.loaded_context
        scoped_evidence = [e for e in state.evidence if e.kind in ("graph_impact", "search_hit")]
        spec = WorkerSpec(
            role=role,
            sub_task=sub,
            user_request=state.user_request,
            scoped_context=context_bundle,
            scoped_evidence=[e.model_dump() if hasattr(e, "model_dump") else e for e in scoped_evidence],
            parent_session_id=state.task.id,
            execution_mode=state.execution_mode,
        )
        specs.append(spec)

    return specs
