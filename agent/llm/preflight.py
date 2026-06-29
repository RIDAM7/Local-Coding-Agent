"""Pre-run validation of role -> provider routing (Phase 2).

Before a task runs, verify that every role that will actually be used is
configured and reachable:
  - Ollama roles: the host responds and the resolved model is pulled.
  - Cloud roles: the required API key (and base_url for OpenAI) is present.

Failures are aggregated into one clear, actionable :class:`PreflightError`.
Secrets are NEVER included in any message — only the env var name is named.
"""

import shutil

import aiohttp

from agent.config import settings, logger
from agent.exceptions.errors import PreflightError
from agent.llm.factory import resolve_provider, resolve_model

# Roles always exercised by the pipeline (the refiner is appended in
# ``preflight_check`` only when REFINER_ENABLED is true; the reviewer is
# budget-gated and degrades gracefully, so it is not hard-required).
DEFAULT_ROLES = ["planner", "coder", "constraint", "repair", "reflection"]

# provider -> (env var name that must be set) for cloud providers.
CLOUD_KEY_ENV = {
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    "google": ("google_api_key", "GOOGLE_API_KEY"),
}


async def _ollama_models(base_url: str) -> list[str]:
    """Return the list of pulled Ollama model names (raises on connection error)."""
    url = f"{base_url.rstrip('/')}/api/tags"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise PreflightError(
                    f"Ollama at {base_url} responded with status {response.status}: {text}"
                )
            data = await response.json()
    return [m.get("name", "") for m in (data.get("models") or [])]


def _model_present(model: str, available: list[str]) -> bool:
    if model in available:
        return True
    # Tolerate ":latest" omission and base-name matches (e.g. "qwen2.5" ~ "qwen2.5:latest").
    base = model.split(":")[0]
    for name in available:
        if name == f"{model}:latest" or name.split(":")[0] == base:
            return True
    return False


async def preflight_check(roles: list[str] | None = None) -> None:
    """Validate the given roles (default: the pipeline roles). Raises PreflightError."""
    if roles is None:
        roles = list(DEFAULT_ROLES)
        # The refiner is optional (Phase 3): only validate it when it will run.
        if settings.refiner_enabled:
            roles.append("refiner")
    problems: list[str] = []

    # Resolve once; only contact each distinct Ollama host a single time.
    ollama_cache: dict[str, list[str] | str] = {}

    for role in roles:
        provider = resolve_provider(role)
        model = resolve_model(role)

        if provider == "ollama":
            base_url = settings.ollama_base_url
            if base_url not in ollama_cache:
                try:
                    ollama_cache[base_url] = await _ollama_models(base_url)
                except PreflightError as e:
                    ollama_cache[base_url] = f"__ERROR__:{e}"
                except aiohttp.ClientError as e:
                    ollama_cache[base_url] = f"__ERROR__:Cannot reach Ollama at {base_url}: {e}"

            cached = ollama_cache[base_url]
            if isinstance(cached, str) and cached.startswith("__ERROR__:"):
                problems.append(
                    f"role '{role}' (ollama): {cached[len('__ERROR__:'):]} "
                    f"Is Ollama running? Start it or set OLLAMA_BASE_URL."
                )
            elif not _model_present(model, cached):
                problems.append(
                    f"role '{role}' (ollama): model '{model}' is not pulled. "
                    f"Run: ollama pull {model}"
                )

        elif provider in CLOUD_KEY_ENV:
            attr, env_name = CLOUD_KEY_ENV[provider]
            if not (getattr(settings, attr, "") or "").strip():
                problems.append(
                    f"role '{role}' ({provider}): missing API key. Set {env_name} in your .env."
                )
            if provider == "openai" and not (settings.openai_base_url or "").strip():
                problems.append(
                    f"role '{role}' (openai): OPENAI_BASE_URL is empty. "
                    f"Set it (default https://api.openai.com/v1)."
                )
        else:
            problems.append(
                f"role '{role}': unknown provider '{provider}'. "
                f"Supported: ollama, openai, anthropic, google."
            )

    if problems:
        message = "Preflight failed — fix the following before running:\n  - " + "\n  - ".join(problems)
        logger.error(message)
        raise PreflightError(message)

    logger.info(f"Preflight OK for roles: {', '.join(roles)}")


def tooling_check() -> None:
    """Verify external tooling the retrieval layer needs is available (Phase 4b).

    Checks that Ripgrep (``rg``) is on PATH and that the tree-sitter language
    grammars actually load. Fails fast with an actionable message instead of a
    deep stack trace mid-run.
    """
    problems: list[str] = []

    if shutil.which("rg") is None:
        problems.append(
            "Ripgrep ('rg') was not found on PATH. The retrieval layer needs it. "
            "Install it (Windows: 'winget install BurntSushi.ripgrep.MSVC'; "
            "macOS: 'brew install ripgrep'; Debian/Ubuntu: 'apt install ripgrep') "
            "or see https://github.com/BurntSushi/ripgrep#installation."
        )

    try:
        from agent.retrieval.tree_sitter_indexer import TreeSitterIndexer
        if not TreeSitterIndexer().get_supported_extensions():
            problems.append(
                "Tree-sitter language grammars failed to load. Reinstall the "
                "Python deps: pip install -r requirements.txt"
            )
    except Exception as e:
        problems.append(
            f"Tree-sitter is not available ({type(e).__name__}: {e}). Reinstall "
            f"the Python deps: pip install -r requirements.txt"
        )

    if problems:
        message = "Tooling preflight failed — fix the following before running:\n  - " + "\n  - ".join(problems)
        logger.error(message)
        raise PreflightError(message)

    logger.info("Tooling preflight OK (rg + tree-sitter present).")
