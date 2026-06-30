"""Phase 9 — architecture summarizer.

Uses the local **planner** model (via the injected ``BaseLLMClient``) to turn the
machine bundle into a short prose architecture overview. It must degrade
gracefully: when the client is ``None`` or the call fails/times out, the engine
still produces a fully usable machine-only bundle and a deterministic
``architecture.md``. No network call is ever made here beyond what the planner
client itself does (which is local Ollama by default).
"""

from __future__ import annotations

from typing import Optional

from agent.config import logger
from agent.context.schemas import ArchitectureSummary, ContextBundle


def _facts_block(bundle: ContextBundle) -> str:
    parts = [
        f"Files: {bundle.file_count}",
        f"Symbols: {bundle.symbol_count}",
    ]
    if bundle.frameworks:
        parts.append("Frameworks: " + ", ".join(bundle.frameworks))
    if bundle.tech_stack:
        for t in bundle.tech_stack:
            deps = ", ".join(t.dependencies[:15])
            parts.append(f"{t.ecosystem} ({t.manifest}): {deps}")
    if bundle.entry_points:
        parts.append("Entry points: " + ", ".join(
            f"{e.kind}:{e.target}" for e in bundle.entry_points[:10]))
    if bundle.conventions.source_layout:
        parts.append("Top-level dirs: " + ", ".join(bundle.conventions.source_layout[:12]))
    if bundle.languages:
        top = sorted(bundle.languages.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        parts.append("Languages: " + ", ".join(f"{e}={n}" for e, n in top))
    return "\n".join(parts)


async def summarize_architecture(bundle: ContextBundle, client) -> Optional[str]:
    """Ask the planner model for a short architecture overview.

    Returns the prose summary, or ``None`` when no client is available or the call
    fails — the caller treats ``None`` as "machine-only bundle".
    """
    if client is None:
        logger.info("Context engine: no LLM client; producing machine-only bundle.")
        return None

    model = getattr(client, "model", None) or "planner"
    prompt = (
        "You are a senior engineer onboarding to an unfamiliar repository.\n"
        "Given ONLY the structured facts below, write a concise architecture\n"
        "overview (3-6 sentences) plus the key components. Do not invent details\n"
        "that are not supported by the facts.\n\n"
        f"<facts>\n{_facts_block(bundle)}\n</facts>\n\n"
        "Return JSON matching: "
        '{"overview": "...", "key_components": ["..."], "notes": "..."}'
    )
    try:
        result = await client.generate_structured(model, prompt, ArchitectureSummary)
        summary: ArchitectureSummary = result.data
        lines = [summary.overview.strip()]
        if summary.key_components:
            lines.append("")
            lines.append("Key components:")
            for c in summary.key_components:
                lines.append(f"- {c}")
        if summary.notes.strip():
            lines.append("")
            lines.append(summary.notes.strip())
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Context engine: architecture summary skipped ({e!r}); "
                       "falling back to machine-only bundle.")
        return None


def render_architecture_md(bundle: ContextBundle) -> str:
    """Deterministic architecture.md — always usable, with or without the LLM."""
    lines = ["# Architecture", ""]
    lines.append(f"_Generated {bundle.generated_at} (local, Phase 9 Context Engine)._")
    lines.append("")
    if bundle.architecture_summary:
        lines.append("## Overview")
        lines.append("")
        lines.append(bundle.architecture_summary)
        lines.append("")
    lines.append("## Tech stack")
    lines.append("")
    if bundle.frameworks:
        lines.append(f"- **Frameworks:** {', '.join(bundle.frameworks)}")
    for t in bundle.tech_stack:
        deps = ", ".join(t.dependencies[:25]) or "(none declared)"
        lines.append(f"- **{t.ecosystem}** (`{t.manifest}`): {deps}")
    if not bundle.tech_stack and not bundle.frameworks:
        lines.append("- (no recognized manifests)")
    lines.append("")
    lines.append("## Entry points")
    lines.append("")
    if bundle.entry_points:
        for e in bundle.entry_points:
            lines.append(f"- `{e.target}` — {e.kind} ({e.evidence})")
    else:
        lines.append("- (none discovered)")
    lines.append("")
    lines.append("## Languages")
    lines.append("")
    for ext, n in sorted(bundle.languages.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{ext}`: {n}")
    lines.append("")
    return "\n".join(lines)


def render_conventions_md(bundle: ContextBundle) -> str:
    """Deterministic conventions.md."""
    c = bundle.conventions
    lines = ["# Conventions", ""]
    lines.append(f"- **Primary language:** {c.primary_language or 'unknown'}")
    lines.append(f"- **Naming style:** {c.naming_style or 'unknown'}")
    lines.append(f"- **Test layout:** {c.test_layout or 'unknown'}")
    lines.append(f"- **Lint/format tools:** {', '.join(c.lint_tools) or 'none detected'}")
    if c.source_layout:
        lines.append(f"- **Top-level layout:** {', '.join(c.source_layout)}")
    if c.notes:
        lines.append("")
        lines.append("## Notes")
        for n in c.notes:
            lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)
