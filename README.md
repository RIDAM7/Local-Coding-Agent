# Local Coding Agent

A **local-first**, autonomous coding agent powered by [Ollama](https://ollama.com/) and Qwen models. It combines a repository-intelligence retrieval layer (Ripgrep + Tree-sitter) with planning, code generation, validation, and self-healing repair loops — all running entirely on your own machine, with no external API calls.

## Features

- **Retrieval Layer** — Dynamically indexes the repository to extract symbols, routes, and components, letting the agent locate precise file context instead of stuffing the whole repo into the prompt.
- **Planner** — Uses a mid-size model (e.g. `qwen3:14b`) to break a task into a structured implementation plan.
- **Coder** — Uses a dedicated coding model (e.g. `qwen2.5-coder`) to generate structured file changes based *only* on the relevant retrieved context.
- **Validation Layer** — Evaluates file operations before applying, auto-corrects path issues, and runs configurable build / lint / test commands after modifications to catch execution errors.
- **Repair Layer (Self-Healing)** — Catches validation failures, classifies errors (Build / Lint / Test), retrieves error-relevant context, generates repair patches, and rolls back if it exceeds the max attempts.
- **Execution** — Runs terminal commands asynchronously and captures their output.
- **Reporting** — Generates a detailed Markdown report summarizing retrieved context, task execution, and modifications.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) running locally
- [`rg` (Ripgrep)](https://github.com/BurntSushi/ripgrep) installed and available on your `PATH`

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment.** Copy the example file and edit it to match your models and validation commands:
   ```bash
   cp .env.example .env
   ```
   ```env
   OLLAMA_BASE_URL=http://localhost:11434

   PLANNER_MODEL=qwen3:14b
   CODER_MODEL=qwen2.5-coder:7b-instruct

   WORKSPACE_DIR=./workspace
   LOG_LEVEL=INFO

   BUILD_COMMAND=python -m compileall .
   LINT_COMMAND=
   TEST_COMMAND=python -m unittest discover

   MAX_REPAIR_ATTEMPTS=3
   ```

3. **Pull the required Ollama models:**
   ```bash
   ollama pull qwen3:14b
   ollama pull qwen2.5-coder:7b-instruct
   ```

## CLI Usage

The agent supports explicit indexing, searching, symbol inspection, and end-to-end task processing.

**1. Indexing** — Builds the repository map and symbol index under `workspace/index/`.
```bash
python main.py index
# Force a full rebuild instead of an incremental update:
python main.py index --reindex
```

**2. Testing Search** — Exercise the retrieval manager and scoring logic directly, without invoking the LLM coder.
```bash
python main.py search "jwt authentication"
```

**3. Viewing Symbols** — Print all parsed Tree-sitter symbols in the workspace.
```bash
python main.py symbols
```

**4. Running a Task** — Run an end-to-end task. The index is incrementally updated automatically before the run.
```bash
python main.py "Create calculator.py that prints Hello World and run it."
```
> Run `python main.py` with no arguments to enter interactive multiline mode (submit with `Ctrl+Z` on Windows, `Ctrl+D` on Linux/Mac).

## Project Structure

```
agent/
├── orchestrator.py   # Coordinates the full plan → code → validate → repair loop
├── config.py         # Environment-driven settings
├── retrieval/        # Ripgrep + Tree-sitter indexing and context search
├── planner/          # Task decomposition
├── coder/            # Structured code generation
├── validation/       # Pre-apply checks + build/lint/test execution
├── repair/           # Self-healing repair patches
├── execution/        # Async terminal command runner
├── reporting/        # Markdown run reports
└── ...               # llm, memory, models, review, reflection, evaluation
main.py               # CLI entry point
```

## License

This project is provided as-is for local and educational use.
