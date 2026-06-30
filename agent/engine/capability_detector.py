"""Phase 10 — Capability Detector (no model-size heuristic, ever).

Answers *what can this provider+model actually do* — reliable structured output,
tool calling, function calling, streaming, context-window length, reasoning mode,
vision — never *how big is it*. `EXECUTION_MODE=auto` selects the **agent** strategy
only when the minimum capabilities are present (reliable structured output + tool
calling + adequate context); otherwise the **pipeline** strategy. So a small model
that is genuinely good at tool calling is treated as agent-capable, and a large one
that is not, is not.

Detection sources, cheapest first:
  1. a static capability table for known provider/models;
  2. provider metadata / model-list endpoints (stub — extension point for 10B);
  3. an optional one-time, cached **probe** (a tiny structured-output check) for
     unknown local/gateway models, gated by ``CAPABILITY_PROBE``.

Results are cached per provider+model in ``.localcli/capabilities.json`` so
detection runs at most once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from agent.config import logger, settings

# Minimum context window (tokens) we consider "adequate" for the agent loop.
MIN_CONTEXT = 8000


class Capabilities(BaseModel):
    provider: str = ""
    model: str = ""
    structured_output: bool = False
    tool_calling: bool = False
    function_calling: bool = False
    streaming: bool = False
    context_window: int = 0
    reasoning: bool = False
    vision: bool = False
    source: str = "default"          # table | metadata | probe | default

    def is_agent_capable(self) -> bool:
        return (self.structured_output and self.tool_calling
                and self.context_window >= MIN_CONTEXT)


class _ProbeSchema(BaseModel):
    ok: bool = Field(default=True)
    answer: str = ""


# --- static capability table -------------------------------------------------
# Keyed by provider -> list of (model substring, capability kwargs). First match
# wins. These are *capability* facts, not sizes.
_TABLE = {
    "anthropic": [
        ("claude", dict(structured_output=True, tool_calling=True, function_calling=True,
                        streaming=True, context_window=200000, reasoning=True, vision=True)),
    ],
    "openai": [
        ("gpt-4", dict(structured_output=True, tool_calling=True, function_calling=True,
                       streaming=True, context_window=128000, vision=True)),
        ("gpt-3.5", dict(structured_output=True, tool_calling=True, function_calling=True,
                         streaming=True, context_window=16000)),
        ("o1", dict(structured_output=True, tool_calling=True, function_calling=True,
                    streaming=False, context_window=128000, reasoning=True)),
    ],
    "google": [
        ("gemini", dict(structured_output=True, tool_calling=True, function_calling=True,
                        streaming=True, context_window=1000000, vision=True)),
    ],
    # Local Ollama models that reliably support tool calling + JSON mode.
    "ollama": [
        ("qwen2.5-coder", dict(structured_output=True, tool_calling=True, function_calling=True,
                               streaming=True, context_window=32000)),
        ("qwen2.5", dict(structured_output=True, tool_calling=True, function_calling=True,
                         streaming=True, context_window=32000)),
        ("llama3.1", dict(structured_output=True, tool_calling=True, function_calling=True,
                          streaming=True, context_window=128000)),
        ("llama3.2", dict(structured_output=True, tool_calling=True, function_calling=True,
                          streaming=True, context_window=128000)),
        ("mistral-nemo", dict(structured_output=True, tool_calling=True, function_calling=True,
                              streaming=True, context_window=128000)),
        ("command-r", dict(structured_output=True, tool_calling=True, function_calling=True,
                           streaming=True, context_window=128000)),
        ("firefunction", dict(structured_output=True, tool_calling=True, function_calling=True,
                              streaming=True, context_window=32000)),
    ],
}


def _table_lookup(provider: str, model: str) -> Optional[Capabilities]:
    entries = _TABLE.get(provider, [])
    low = (model or "").lower()
    for substr, caps in entries:
        if substr in low:
            return Capabilities(provider=provider, model=model, source="table", **caps)
    return None


class CapabilityDetector:
    def __init__(self, cache_path: str | Path | None = None):
        if cache_path is None:
            cache_path = settings.get_workspace_path() / ".localcli" / "capabilities.json"
        self.cache_path = Path(cache_path)

    # --- cache --------------------------------------------------------------

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache: dict) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as e:
            logger.debug(f"CapabilityDetector: could not write cache: {e}")

    # --- detection ----------------------------------------------------------

    async def detect(self, provider: str, model: str, *, client=None,
                     use_cache: bool = True) -> Capabilities:
        key = f"{provider}/{model}"

        if use_cache:
            cached = self._load_cache().get(key)
            if cached:
                return Capabilities(**cached)

        # (1) static table — cheapest, authoritative for known models.
        caps = _table_lookup(provider, model)

        # (2) provider metadata — extension point (10B). Not used offline.

        # (3) optional one-time probe for unknown models.
        if caps is None:
            if settings.capability_probe and client is not None:
                caps = await self._probe(provider, model, client)
            else:
                caps = Capabilities(provider=provider, model=model, source="default")
                logger.info(f"CapabilityDetector: {key} unknown and no probe — defaulting to pipeline.")

        cache = self._load_cache()
        cache[key] = caps.model_dump()
        self._save_cache(cache)
        return caps

    async def _probe(self, provider: str, model: str, client) -> Capabilities:
        """A tiny structured-output check. Confirms JSON reliability only — tool
        calling is left unconfirmed (so unknown models degrade to pipeline unless
        the static table proves otherwise). 10B can extend this to probe tools."""
        caps = Capabilities(provider=provider, model=model, source="probe")
        try:
            result = await client.generate_structured(
                model, "Return JSON {\"ok\": true, \"answer\": \"pong\"}.", _ProbeSchema)
            data = getattr(result, "data", None)
            caps.structured_output = bool(data is not None)
            caps.context_window = MIN_CONTEXT  # assume adequate if it answered
            logger.info(f"CapabilityDetector: probed {provider}/{model} "
                        f"structured_output={caps.structured_output}.")
        except Exception as e:
            logger.info(f"CapabilityDetector: probe failed for {provider}/{model} ({e!r}); "
                        "treating as not agent-capable.")
        return caps
