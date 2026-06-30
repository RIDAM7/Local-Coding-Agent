"""localcli — command-line interface (Phase 6).

Subcommands:
  localcli run "<task>"      run an end-to-end coding task (default; the bare
                             form `localcli "<task>"` is shorthand for `run`)
  localcli config check      preflight + tooling check, print the resolved
                             role->provider->model routing table and health
  localcli models            print the routing table only (no network)
  localcli index|search|symbols   retrieval/index utilities (unchanged behavior)

Phase 5 safety flags (--yes/--auto, --dry-run) are carried on `run`.
Secret values are NEVER printed — credentials are shown as "set"/"missing" only.
"""

import sys
import os
import asyncio
import argparse

from agent.config import settings
from agent.safety.controller import SafetyMode
from agent.llm.factory import resolve_provider, resolve_model
from agent.llm.preflight import (
    preflight_check, tooling_check, CLOUD_KEY_ENV, _ollama_models, _model_present,
)
from agent.exceptions.errors import PreflightError

# All roles the agent routes, in a sensible display order.
ROLES = ["planner", "coder", "refiner", "repair", "constraint", "reflection", "reviewer"]

# Subcommands argparse owns; anything else as the first token is treated as a
# bare-task shorthand for `run`.
KNOWN_SUBCOMMANDS = {"run", "config", "models", "index", "search", "symbols", "context"}


# --- Routing table -----------------------------------------------------------

def routing_rows():
    """Resolve (role, provider, model) for every role. No network, no secrets."""
    return [(role, resolve_provider(role), resolve_model(role)) for role in ROLES]


def _print_routing_table(rows):
    print(f"{'Role':<12} {'Provider':<12} Model")
    print("-" * 52)
    for role, provider, model in rows:
        print(f"{role:<12} {provider:<12} {model}")


# --- Subcommand handlers (return an int exit code) ---------------------------

def cmd_models(args):
    """Print the resolved routing table only (no network)."""
    _print_routing_table(routing_rows())
    return 0


async def cmd_config_check(args):
    """Print routing table + credential/health status; run preflight; exit code."""
    rows = routing_rows()
    print("Resolved routing:")
    _print_routing_table(rows)

    # Credentials — set/missing only, NEVER the value.
    print("\nCredentials:")
    for provider, (attr, env_name) in CLOUD_KEY_ENV.items():
        present = bool((getattr(settings, attr, "") or "").strip())
        print(f"  {env_name:<18} {'set' if present else 'missing'}")

    # Provider health.
    print("\nProvider health:")
    ollama_roles = [(r, m) for (r, p, m) in rows if p == "ollama"]
    if ollama_roles:
        base = settings.ollama_base_url
        try:
            available = await _ollama_models(base)
            print(f"  ollama @ {base}: reachable")
            for role, model in ollama_roles:
                mark = "pulled" if _model_present(model, available) else f"MISSING (run: ollama pull {model})"
                print(f"    {role:<11} {model:<26} {mark}")
        except Exception as e:
            print(f"  ollama @ {base}: UNREACHABLE ({e})")
    cloud_providers = sorted({p for (_, p, _) in rows if p in CLOUD_KEY_ENV})
    for provider in cloud_providers:
        attr, env_name = CLOUD_KEY_ENV[provider]
        present = bool((getattr(settings, attr, "") or "").strip())
        print(f"  {provider}: key {'present' if present else 'MISSING'} ({env_name})")

    # Authoritative pass/fail (also validates tooling). Secret-free messages.
    try:
        tooling_check()
        await preflight_check()
    except PreflightError as e:
        print(f"\n{e}")
        print("\nconfig check: FAILED")
        return 1

    print("\nconfig check: OK")
    return 0


async def cmd_run(args):
    """Run an end-to-end coding task through the orchestrator."""
    from agent.orchestrator import Orchestrator  # heavy import, only when running

    safety_mode = SafetyMode(auto_approve=args.auto, dry_run=args.dry_run)
    task_description = " ".join(args.task).strip() if args.task else ""

    print("=" * 50)
    print("Initializing Local Coding Agent...")
    print(f"Workspace: {settings.workspace_dir}")
    print("=" * 50)

    os.makedirs(settings.workspace_dir, exist_ok=True)
    orchestrator = Orchestrator(safety_mode=safety_mode)
    if safety_mode.dry_run:
        print("DRY RUN: previewing diffs + planned commands. Nothing will be written or executed.")

    if not task_description:
        print("\nEnter task (press Ctrl+D on Linux/Mac or Ctrl+Z on Windows to submit):")
        lines = []
        try:
            while True:
                lines.append(input("> "))
        except EOFError:
            pass
        task_description = "\n".join(lines).strip()

    if not task_description:
        print("\nNo task provided. Exiting.")
        return 0

    print(f"\nExecuting task:\n{task_description}")

    # Preflight: validate tooling (rg + tree-sitter) and each role's
    # provider/model/credentials before any work. Secret-free messages.
    try:
        tooling_check()
        await preflight_check()
    except PreflightError as e:
        print(f"\n{e}")
        return 1

    _ensure_index(orchestrator)

    # Phase 10: resolve the execution strategy. `pipeline` is byte-for-byte the
    # Round 1 path (orchestrator.run unchanged); `agent` runs the governed loop.
    from agent.engine.selector import resolve_mode, build_engine
    mode, caps = await resolve_mode()
    print(f"Execution mode: {mode}" + (f" (auto via capabilities: {caps.source})" if caps else ""))

    if mode == "agent":
        from agent.state.agent_state import AgentState, TaskMetadata
        from agent.context import build_context_bundle
        state = AgentState(user_request=task_description,
                           task=TaskMetadata(description=task_description))
        try:
            state.loaded_context = await build_context_bundle(settings.get_workspace_path())
        except Exception:
            pass
        if caps:
            state.capabilities = caps.model_dump()
        # Phase 11: incremental planning is on by default (settings); the agent
        # loop then runs plan -> execute step -> observe -> replan.
        engine = build_engine("agent", safety_mode=safety_mode,
                              memory_manager=orchestrator.memory_manager,
                              incremental=settings.incremental_planning)
        state = await engine.execute(state)
        print(f"\nAgent run complete: status={state.final_outputs.status}, "
              f"confidence={state.confidence}, steps={state.governor.steps_used}, "
              f"stop={state.governor.stop_reason}")
        print(f"Files modified: {[f.path for f in state.files_modified] or 'none'}")
        if settings.incremental_planning and state.plan.revisions:
            from agent.planning import render_plan_evolution
            print("\n" + render_plan_evolution(state))
        return 0

    # Phase 11: pipeline strategy with step-wise execution + replanning when
    # INCREMENTAL_PLANNING is on. When off, orchestrator.run is byte-for-byte the
    # Round 1 plan-once-execute-all path (parity).
    if settings.incremental_planning:
        report_path = await orchestrator.run_incremental(task_description)
    else:
        report_path = await orchestrator.run(task_description)
    print(f"\nExecution complete. Report generated at: {report_path}")
    return 0


def _ensure_index(orchestrator):
    idx_dir = orchestrator.retrieval_manager.index_dir
    ws = str(orchestrator.retrieval_manager.workspace)
    if not os.path.exists(os.path.join(idx_dir, "metadata.json")):
        print("No index found. Building index first...")
        orchestrator.retrieval_manager.sym_idx.build_index(ws, idx_dir)
        map_data = orchestrator.retrieval_manager.repo_map.generate(ws)
        orchestrator.retrieval_manager.repo_map.save(map_data, idx_dir)
    else:
        orchestrator.retrieval_manager.sym_idx.incremental_update(ws, idx_dir)


async def cmd_context(args):
    """Phase 9: generate/refresh the repository Context Bundle and print a summary.

    100% local. Uses the local planner model for the architecture summary when
    reachable, and degrades gracefully to a machine-only bundle otherwise.
    """
    from agent.context import ContextEngine

    if not settings.context_engine_enabled:
        print("Context engine is disabled (CONTEXT_ENGINE_ENABLED=false).")
        return 0

    engine = ContextEngine(settings.get_workspace_path())
    # The architecture summary uses the local planner model; if it is None or
    # unreachable, the bundle is still produced (machine-only).
    llm = None
    try:
        from agent.llm.factory import build_client
        llm = build_client("planner")
    except Exception:
        llm = None

    print(f"Scanning workspace: {settings.workspace_dir}")
    bundle = await engine.build(force=args.refresh, llm_client=llm)

    print("\nRepository Context")
    print("-" * 52)
    print(f"Files scanned   : {bundle.file_count}")
    print(f"Symbols indexed : {bundle.symbol_count}")
    if bundle.tech_stack:
        print(f"Ecosystems      : {', '.join(sorted({t.ecosystem for t in bundle.tech_stack}))}")
    if bundle.frameworks:
        print(f"Frameworks      : {', '.join(bundle.frameworks)}")
    if bundle.entry_points:
        eps = ", ".join(f"{e.kind}:{e.target}" for e in bundle.entry_points[:6])
        print(f"Entry points    : {eps}")
    if bundle.conventions.primary_language:
        print(f"Primary language: {bundle.conventions.primary_language}")
    if bundle.conventions.test_layout:
        print(f"Test layout     : {bundle.conventions.test_layout}")
    print(f"Architecture    : {'LLM summary' if bundle.architecture_summary else 'machine-only (no LLM)'}")
    print(f"\nCached at: {engine.context_dir}")
    print("  repo_context.json, architecture.md, conventions.md, dependency_graph.json")
    return 0


async def cmd_index(args):
    from agent.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    print("\nRunning indexing...")
    idx_dir = orchestrator.retrieval_manager.index_dir
    ws = str(orchestrator.retrieval_manager.workspace)
    if args.reindex:
        orchestrator.retrieval_manager.sym_idx.build_index(ws, idx_dir)
    else:
        orchestrator.retrieval_manager.sym_idx.incremental_update(ws, idx_dir)
    map_data = orchestrator.retrieval_manager.repo_map.generate(ws)
    orchestrator.retrieval_manager.repo_map.save(map_data, idx_dir)
    print("Indexing complete.")
    return 0


async def cmd_search(args):
    from agent.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    query = " ".join(args.query)
    print(f"\nSearching context for: {query}")
    context = await orchestrator.retrieval_manager.search_context(query, None)
    print(f"\nFound {context.total_files} relevant files:")
    for res in context.results:
        print(f"- {res.file} (Score: {res.score})")
        print(f"  Evidence: {res.evidence}")
    return 0


async def cmd_symbols(args):
    from agent.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    print("\nExtracted Symbols:")
    idx_dir = orchestrator.retrieval_manager.index_dir
    symbols = orchestrator.retrieval_manager.sym_idx.load(idx_dir)
    if not symbols:
        print("No symbols found. Run 'localcli index' first.")
        return 0
    for s in symbols:
        print(f"[{s.type}] {s.name} in {s.file}:{s.line_start}")
    return 0


# --- Parser ------------------------------------------------------------------

def _add_safety_flags(p):
    p.add_argument("--yes", "--auto", dest="auto", action="store_true",
                   help="Skip confirmation prompts (for CI). The hard safety denylist still applies.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Preview file diffs and proposed commands; write nothing and run nothing.")


def build_parser():
    parser = argparse.ArgumentParser(prog="localcli", description="Local-first multi-stage coding agent.")
    sub = parser.add_subparsers(dest="subcommand")

    p_run = sub.add_parser("run", help="Run an end-to-end coding task")
    p_run.add_argument("task", nargs="*", help="The task description")
    p_run.add_argument("--reindex", action="store_true", help="Force full rebuild of the index first")
    _add_safety_flags(p_run)

    p_index = sub.add_parser("index", help="Build/update the workspace index")
    p_index.add_argument("--reindex", action="store_true", help="Force full rebuild")

    p_search = sub.add_parser("search", help="Exercise retrieval without invoking the LLM")
    p_search.add_argument("query", nargs="+", help="Search query")

    sub.add_parser("symbols", help="Print parsed Tree-sitter symbols")

    p_context = sub.add_parser("context", help="Generate/refresh the repository Context Bundle (Phase 9, local)")
    p_context.add_argument("--refresh", action="store_true", help="Force a full rescan, ignoring the cache")

    p_config = sub.add_parser("config", help="Configuration utilities")
    config_sub = p_config.add_subparsers(dest="config_action")
    config_sub.add_parser("check", help="Run preflight and print the resolved routing table + health")

    sub.add_parser("models", help="Print the resolved role->provider->model routing table (no network)")

    return parser


def _normalize_argv(argv):
    """Treat a leading non-subcommand token as the bare-task shorthand for `run`."""
    if not argv:
        return ["run"]
    first = argv[0]
    if first in KNOWN_SUBCOMMANDS or first in ("-h", "--help"):
        return argv
    return ["run"] + argv


async def _dispatch(args):
    if args.subcommand == "models":
        return cmd_models(args)
    if args.subcommand == "config":
        if getattr(args, "config_action", None) == "check":
            return await cmd_config_check(args)
        print("Usage: localcli config check")
        return 2
    if args.subcommand == "index":
        return await cmd_index(args)
    if args.subcommand == "search":
        return await cmd_search(args)
    if args.subcommand == "symbols":
        return await cmd_symbols(args)
    if args.subcommand == "context":
        return await cmd_context(args)
    # default: run
    return await cmd_run(args)


def main(argv=None):
    """Console entry point. Returns an int exit code."""
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(raw))
    try:
        return asyncio.run(_dispatch(args))
    except KeyboardInterrupt:
        print("\nAgent execution cancelled by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
