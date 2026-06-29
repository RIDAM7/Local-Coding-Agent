import sys
import asyncio
import os
import argparse
from agent.orchestrator import Orchestrator
from agent.config import settings
from agent.llm.preflight import preflight_check, tooling_check
from agent.exceptions.errors import PreflightError

async def main():
    # Use argparse for CLI commands
    parser = argparse.ArgumentParser(description="Local Coding Agent Phase 2")
    parser.add_argument("command", nargs="?", default="", help="Command to run: index, search, symbols, or the task string itself")
    parser.add_argument("args", nargs="*", help="Arguments for the command or rest of the task string")
    parser.add_argument("--reindex", action="store_true", help="Force full rebuild of index")
    
    parsed_args = parser.parse_args()

    print("=" * 50)
    print("Initializing Local Coding Agent (Phase 2 - Repository Intelligence)...")
    print(f"Workspace: {settings.workspace_dir}")
    print("=" * 50)
    
    os.makedirs(settings.workspace_dir, exist_ok=True)
    orchestrator = Orchestrator()
    
    cmd = parsed_args.command.lower()
    known_commands = ["index", "search", "symbols"]

    if cmd == "index":
        print("\nRunning indexing...")
        idx_dir = orchestrator.retrieval_manager.index_dir
        ws = str(orchestrator.retrieval_manager.workspace)
        if parsed_args.reindex:
            orchestrator.retrieval_manager.sym_idx.build_index(ws, idx_dir)
        else:
            orchestrator.retrieval_manager.sym_idx.incremental_update(ws, idx_dir)
        
        map_data = orchestrator.retrieval_manager.repo_map.generate(ws)
        orchestrator.retrieval_manager.repo_map.save(map_data, idx_dir)
        print("Indexing complete.")
        return

    elif cmd == "search":
        if not parsed_args.args:
            print("Please provide a search query. Example: python main.py search 'jwt'")
            return
        query = " ".join(parsed_args.args)
        print(f"\nSearching context for: {query}")
        context = await orchestrator.retrieval_manager.search_context(query, None)
        print(f"\nFound {context.total_files} relevant files:")
        for res in context.results:
            print(f"- {res.file} (Score: {res.score})")
            print(f"  Evidence: {res.evidence}")
        return

    elif cmd == "symbols":
        print("\nExtracted Symbols:")
        idx_dir = orchestrator.retrieval_manager.index_dir
        symbols = orchestrator.retrieval_manager.sym_idx.load(idx_dir)
        if not symbols:
            print("No symbols found. Run 'python main.py index' first.")
            return
        for s in symbols:
            print(f"[{s.type}] {s.name} in {s.file}:{s.line_start}")
        return

    # If it's not a known command, treat everything as the task description
    task_description = ""
    if cmd and cmd not in known_commands:
        task_description = parsed_args.command + " " + " ".join(parsed_args.args)
        
    if not task_description:
        print("\nEnter task (press Ctrl+D on Linux/Mac or Ctrl+Z on Windows to submit):")
        lines = []
        try:
            while True:
                line = input("> ")
                lines.append(line)
        except EOFError:
            pass
        task_description = "\n".join(lines).strip()
        
    if not task_description:
        print("\nNo task provided. Exiting.")
        return

    print(f"\nExecuting task:\n{task_description}")

    # Preflight: validate tooling (rg + tree-sitter) and each role's
    # provider/model/credentials before any work. Fails fast with a clear,
    # secret-free message instead of a deep stack trace.
    try:
        tooling_check()
        await preflight_check()
    except PreflightError as e:
        print(f"\n{e}")
        return

    # Ensure index exists before running task
    idx_dir = orchestrator.retrieval_manager.index_dir
    ws = str(orchestrator.retrieval_manager.workspace)
    if not os.path.exists(os.path.join(idx_dir, "metadata.json")):
        print("No index found. Building index first...")
        orchestrator.retrieval_manager.sym_idx.build_index(ws, idx_dir)
        map_data = orchestrator.retrieval_manager.repo_map.generate(ws)
        orchestrator.retrieval_manager.repo_map.save(map_data, idx_dir)
    else:
        # Run incremental update invisibly
        orchestrator.retrieval_manager.sym_idx.incremental_update(ws, idx_dir)
        
    report_path = await orchestrator.run(task_description)
    print(f"\nExecution complete. Report generated at: {report_path}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent execution cancelled by user.")
