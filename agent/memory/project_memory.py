"""Phase 12 - local, human-readable project memory.

This module is the markdown memory layer described in Phase 12. It sits beside
the existing vector memory; it does not use embeddings, cloud services, or an
external database. The store is intentionally plain markdown so users can review
and edit what the agent learns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from agent.config import logger, settings
from agent.context.schemas import ContextBundle
from agent.safety.redact import redact
from agent.state.agent_state import AgentState


MEMORY_VERSION = "localcli-project-memory:v1"
DEFAULT_MEMORY_DIR = ".localcli/memory"


@dataclass(frozen=True)
class ProjectMemoryType:
    """A structured markdown memory kind.

    The registry is data-driven so later phases or downstream projects can add a
    new memory type without changing the storage/update machinery.
    """

    key: str
    filename: str
    title: str
    description: str


DEFAULT_MEMORY_TYPES: tuple[ProjectMemoryType, ...] = (
    ProjectMemoryType(
        "learned_patterns",
        "learned_patterns.md",
        "Learned Patterns",
        "Durable coding patterns, repository conventions, and successful approaches.",
    ),
    ProjectMemoryType(
        "developer_preferences",
        "developer_preferences.md",
        "Developer Preferences",
        "Stable preferences inferred from explicit instructions or successful runs.",
    ),
    ProjectMemoryType(
        "architecture",
        "architecture.md",
        "Architecture Notes",
        "Long-lived architecture observations and project terminology.",
    ),
    ProjectMemoryType(
        "commands",
        "commands.md",
        "Project Commands",
        "Commands that were useful for build, lint, test, or project workflows.",
    ),
    ProjectMemoryType(
        "mistakes",
        "mistakes.md",
        "Mistakes And Fixes",
        "Repeated mistakes, failed assumptions, and fixes that worked.",
    ),
)


class ProjectMemoryBundle(BaseModel):
    """Loaded markdown memory, ready to inject into planning context."""

    files: Dict[str, str] = Field(default_factory=dict)
    used_files: List[str] = Field(default_factory=list)
    summaries: List[str] = Field(default_factory=list)
    recovered_files: List[str] = Field(default_factory=list)

    def render_for_planner(self, *, max_chars_per_file: int = 1200) -> str:
        if not self.files:
            return ""
        lines = ["Project Memory (local, user-editable):"]
        for rel in sorted(self.files):
            text = self.files[rel].strip()
            if not text:
                continue
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file].rstrip() + "\n..."
            lines.append(f"\n[{rel}]")
            lines.append(text)
        return "\n".join(lines).strip()


class MemoryUpdateResult(BaseModel):
    files_updated: List[str] = Field(default_factory=list)
    entries_added: int = 0
    entries_skipped: int = 0


def _resolve_memory_dir(workspace: Path, memory_dir: str | Path | None = None) -> Path:
    raw = Path(memory_dir or settings.project_memory_dir or DEFAULT_MEMORY_DIR)
    return raw if raw.is_absolute() else workspace / raw


def _scrub_memory_text(text: str, *, workspace: Optional[Path] = None) -> str:
    """Redact secrets and scrub absolute paths before writing memory."""
    scrubbed = redact(str(text or ""))
    if workspace is not None:
        try:
            scrubbed = scrubbed.replace(str(workspace.resolve()), "[WORKSPACE]")
        except Exception:
            pass
    # Windows paths like C:\Users\...
    scrubbed = re.sub(r"[a-zA-Z]:\\[^\s\"'`<>]+", "[SCRUBBED_PATH]", scrubbed)
    # Unix absolute paths. Keep markdown list markers and URLs intact.
    scrubbed = re.sub(r"(?<![\w:])/[^\s\"'`<>]+", "[SCRUBBED_PATH]", scrubbed)
    return re.sub(r"\s+", " ", scrubbed).strip()


def _normalize_entry(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_durable(text: str) -> bool:
    lowered = text.lower()
    if len(lowered.strip()) < 12:
        return False
    transient_markers = (
        "temporary",
        "temp ",
        "scratch",
        "debug only",
        "one-off",
        "do not store",
        "ignore this",
    )
    return not any(marker in lowered for marker in transient_markers)


class ProjectMemoryManager:
    """Retriever + updater for local markdown project memory."""

    def __init__(
        self,
        workspace_path: Path | str | None = None,
        *,
        memory_dir: str | Path | None = None,
        memory_types: Sequence[ProjectMemoryType] = DEFAULT_MEMORY_TYPES,
        enabled: Optional[bool] = None,
    ):
        self.workspace = Path(workspace_path) if workspace_path else settings.get_workspace_path()
        self.workspace = self.workspace.resolve()
        self.memory_dir = _resolve_memory_dir(self.workspace, memory_dir)
        self.memory_types = tuple(memory_types)
        self.enabled = settings.project_memory_enabled if enabled is None else enabled

    @property
    def relative_dir(self) -> str:
        try:
            return self.memory_dir.relative_to(self.workspace).as_posix()
        except ValueError:
            return self.memory_dir.as_posix()

    def type_by_key(self) -> Dict[str, ProjectMemoryType]:
        return {t.key: t for t in self.memory_types}

    def ensure_files(self) -> None:
        if not self.enabled:
            return
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for mem_type in self.memory_types:
            path = self.memory_dir / mem_type.filename
            if not path.exists():
                path.write_text(self._empty_document(mem_type), encoding="utf-8")

    def load(self) -> ProjectMemoryBundle:
        """Load all known memory files. Missing files simply produce no memory."""
        if not self.enabled:
            return ProjectMemoryBundle()
        self.ensure_files()
        files: Dict[str, str] = {}
        used: List[str] = []
        summaries: List[str] = []
        recovered: List[str] = []
        for mem_type in self.memory_types:
            path = self.memory_dir / mem_type.filename
            rel = self._rel(path)
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"Project memory file is not UTF-8; recovering: {rel}")
                text = path.read_text(encoding="utf-8", errors="replace")
                path.write_text(self._with_header(mem_type, text), encoding="utf-8")
                recovered.append(rel)
            except OSError as e:
                logger.warning(f"Could not load project memory {rel}: {e}")
                continue
            if MEMORY_VERSION not in text:
                text = self._with_header(mem_type, text)
                path.write_text(text, encoding="utf-8")
                recovered.append(rel)
            files[rel] = text
            used.append(rel)
            summaries.extend(self._extract_entries(text)[:5])
        return ProjectMemoryBundle(files=files, used_files=used, summaries=summaries,
                                   recovered_files=recovered)

    def load_into_state(self, state: AgentState) -> ProjectMemoryBundle:
        bundle = self.load()
        if not self.enabled or not bundle.files:
            return bundle
        for rel in bundle.used_files:
            if rel not in state.memory_refs.markdown_files:
                state.memory_refs.markdown_files.append(rel)
        for summary in bundle.summaries:
            if summary not in state.memory_refs.summaries:
                state.memory_refs.summaries.append(summary)
        if state.loaded_context is not None:
            self.inject_into_context(state.loaded_context, bundle)
        state.add_observation(f"loaded project memory: {len(bundle.used_files)} file(s)")
        return bundle

    def inject_into_context(self, context: ContextBundle, bundle: ProjectMemoryBundle | None = None) -> ContextBundle:
        if not self.enabled:
            return context
        bundle = bundle or self.load()
        context.project_memory = dict(bundle.files)
        return context

    def render_for_planning(self) -> str:
        return self.load().render_for_planner()

    def update_from_state(self, state: AgentState) -> MemoryUpdateResult:
        if not self.enabled or state.final_outputs.status != "SUCCESS":
            return MemoryUpdateResult()
        candidates = self._extract_candidates(state)
        result = self.update(candidates)
        if result.entries_added:
            state.add_observation(
                f"project memory updated: {result.entries_added} entr"
                f"{'y' if result.entries_added == 1 else 'ies'}"
            )
        return result

    def update(self, candidates: Mapping[str, Iterable[str]]) -> MemoryUpdateResult:
        """Review, scrub, dedupe, then append durable candidates."""
        if not self.enabled:
            return MemoryUpdateResult()
        self.ensure_files()
        by_key = self.type_by_key()
        updated: List[str] = []
        added = 0
        skipped = 0

        for key, raw_entries in candidates.items():
            mem_type = by_key.get(key)
            if mem_type is None:
                logger.debug(f"Skipping unknown project memory type: {key}")
                continue
            path = self.memory_dir / mem_type.filename
            text = path.read_text(encoding="utf-8") if path.exists() else self._empty_document(mem_type)
            if MEMORY_VERSION not in text:
                text = self._with_header(mem_type, text)
            existing = {_normalize_entry(e) for e in self._extract_entries(text)}
            new_lines: List[str] = []
            for raw in raw_entries:
                entry = _scrub_memory_text(raw, workspace=self.workspace)
                key_norm = _normalize_entry(entry)
                if not key_norm or key_norm in existing or not _is_durable(entry):
                    skipped += 1
                    continue
                existing.add(key_norm)
                new_lines.append(f"- {entry}")
            if new_lines:
                if not text.endswith("\n"):
                    text += "\n"
                text += "\n".join(new_lines) + "\n"
                path.write_text(text, encoding="utf-8")
                updated.append(self._rel(path))
                added += len(new_lines)
        return MemoryUpdateResult(files_updated=updated, entries_added=added,
                                  entries_skipped=skipped)

    def _extract_candidates(self, state: AgentState) -> Dict[str, List[str]]:
        candidates: Dict[str, List[str]] = {t.key: [] for t in self.memory_types}

        for step in state.completed_steps:
            if step.status == "done" and step.summary:
                candidates["learned_patterns"].append(
                    f"Step succeeded: {step.description} -> {step.summary}"
                )
            elif step.status == "failed" and step.summary:
                candidates["mistakes"].append(
                    f"Step failed and required replanning: {step.description} -> {step.summary}"
                )

        for validation in state.validation_results:
            if not validation.success:
                candidates["mistakes"].append(
                    f"{validation.stage} failed with: {validation.detail}"
                )

        for repair in state.repair_attempts:
            if repair.classification:
                status = "worked" if repair.success else "failed"
                candidates["mistakes"].append(
                    f"Repair attempt {repair.attempt} ({repair.classification}) {status}."
                )

        for tool in state.tool_history:
            if tool.name == "run_command":
                command = str(tool.args.get("command", "")).strip()
                if command and tool.status == "ok":
                    candidates["commands"].append(f"Useful command: `{command}`")

        for note in state.memory_refs.summaries:
            lowered = note.lower()
            if any(word in lowered for word in ("prefer", "preference", "avoid", "style")):
                candidates["developer_preferences"].append(note)
            else:
                candidates["learned_patterns"].append(note)

        if state.loaded_context and state.loaded_context.architecture_summary:
            candidates["architecture"].append(state.loaded_context.architecture_summary)
        for evidence in state.evidence:
            if evidence.kind in {"architecture", "terminology"}:
                candidates["architecture"].append(f"{evidence.kind}: {evidence.detail}")

        return candidates

    def _empty_document(self, mem_type: ProjectMemoryType) -> str:
        return (
            f"# {mem_type.title}\n\n"
            f"<!-- {MEMORY_VERSION} type={mem_type.key} -->\n\n"
            f"{mem_type.description}\n\n"
            "## Entries\n"
        )

    def _with_header(self, mem_type: ProjectMemoryType, text: str) -> str:
        entries = "\n".join(f"- {e}" for e in self._extract_entries(text))
        if not entries and text.strip():
            entries = f"- {_scrub_memory_text(text, workspace=self.workspace)}"
        doc = self._empty_document(mem_type)
        return doc + (entries + "\n" if entries else "")

    @staticmethod
    def _extract_entries(text: str) -> List[str]:
        entries: List[str] = []
        for line in (text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                entries.append(stripped[2:].strip())
        return entries

    def _rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError:
            return path.as_posix()
