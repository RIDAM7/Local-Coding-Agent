"""Phase 9 — Context Engine data models.

These are the standalone artifacts the Context Engine returns. The headline type
is :class:`ContextBundle`: a durable, structured understanding of the *whole*
repository. It is intentionally serializable (pydantic) so that Phase 10 can later
assign it to ``state.loaded_context`` without a redesign — nothing here depends on
any global runtime state.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TechStack(BaseModel):
    """A detected ecosystem/manifest and the dependencies declared in it."""

    ecosystem: str                       # python | node | go | rust | java
    manifest: str                        # e.g. pyproject.toml, package.json
    dependencies: List[str] = Field(default_factory=list)


class EntryPoint(BaseModel):
    """A discovered program entry point (a file or a declared script)."""

    kind: str                            # main | cli | server | index | script
    target: str                          # file path or script command
    evidence: str = ""                   # why it was classified this way


class Conventions(BaseModel):
    """Lightweight inferred conventions (MVP). Richer inference is stubbed."""

    primary_language: str = ""
    naming_style: str = ""               # e.g. snake_case / camelCase / mixed
    test_layout: str = ""                # e.g. "tests/ directory", "*_test.go"
    has_lint_config: bool = False
    lint_tools: List[str] = Field(default_factory=list)
    source_layout: List[str] = Field(default_factory=list)  # top-level dirs
    # Enhancement hook — richer pattern inference lands later, kept empty for MVP.
    notes: List[str] = Field(default_factory=list)


class ContextBundle(BaseModel):
    """The Context Bundle — returned to the Planner / Execution Engine.

    Phase 9 returns this as a standalone artifact. Phase 10 will retrofit it into
    ``AgentState.loaded_context``; keeping it a clean, serializable return type is
    what makes that retrofit a no-op.
    """

    root: str                            # absolute workspace path that was scanned
    generated_at: str                    # ISO-8601 (UTC)
    fingerprint: str                     # cache key over the file set
    file_count: int = 0
    languages: Dict[str, int] = Field(default_factory=dict)  # ext -> count
    frameworks: List[str] = Field(default_factory=list)
    tech_stack: List[TechStack] = Field(default_factory=list)
    entry_points: List[EntryPoint] = Field(default_factory=list)
    conventions: Conventions = Field(default_factory=Conventions)
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    architecture_summary: Optional[str] = None  # None when no LLM was available
    symbol_count: int = 0

    def to_planner_block(self) -> str:
        """Render a compact, deterministic text block for the planner prompt.

        Deterministic (sorted, bounded) so the same repo yields the same block —
        important for reproducibility and for the pipeline-parity guarantee (the
        block is only injected when the engine is enabled).
        """
        lines: List[str] = []
        lines.append("Repository Context (auto-generated, local):")
        if self.frameworks:
            lines.append(f"- Frameworks/tech: {', '.join(sorted(self.frameworks))}")
        if self.tech_stack:
            ecos = sorted({t.ecosystem for t in self.tech_stack})
            lines.append(f"- Ecosystems: {', '.join(ecos)}")
        if self.languages:
            top = sorted(self.languages.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
            lines.append("- Languages: " + ", ".join(f"{ext}({n})" for ext, n in top))
        if self.entry_points:
            eps = [f"{e.kind}:{e.target}" for e in self.entry_points[:8]]
            lines.append("- Entry points: " + ", ".join(eps))
        conv = self.conventions
        if conv.primary_language or conv.test_layout or conv.lint_tools:
            bits = []
            if conv.primary_language:
                bits.append(f"language={conv.primary_language}")
            if conv.naming_style:
                bits.append(f"naming={conv.naming_style}")
            if conv.test_layout:
                bits.append(f"tests={conv.test_layout}")
            if conv.lint_tools:
                bits.append(f"lint={'/'.join(sorted(conv.lint_tools))}")
            lines.append("- Conventions: " + ", ".join(bits))
        if conv.source_layout:
            lines.append("- Top-level layout: " + ", ".join(sorted(conv.source_layout)[:10]))
        if self.architecture_summary:
            lines.append("- Architecture:")
            for ln in self.architecture_summary.strip().splitlines():
                lines.append(f"    {ln}")
        return "\n".join(lines)


class ScanResult(BaseModel):
    """Raw output of the repository scan, before detection/summarization."""

    root: str
    fingerprint: str
    files: List[str] = Field(default_factory=list)        # posix rel paths
    file_types: Dict[str, int] = Field(default_factory=dict)
    manifests: Dict[str, str] = Field(default_factory=dict)  # rel path -> content
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    symbol_count: int = 0


class ArchitectureSummary(BaseModel):
    """Structured shape requested from the planner model for architecture.md."""

    overview: str
    key_components: List[str] = Field(default_factory=list)
    notes: str = ""
