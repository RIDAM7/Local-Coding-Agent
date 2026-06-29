# Local Coding Agent

A **local-first**, autonomous coding agent. It runs fully on [Ollama](https://ollama.com/) with
zero configuration, and can optionally route any role to a cloud model (OpenAI / OpenAI-compatible
gateways, Anthropic, Google). It pairs a repository-intelligence retrieval layer
(Ripgrep + Tree-sitter) with a multi-stage pipeline: optional prompt refinement, planning, code
generation, self-critique (reflection), validation, self-healing repair, confidence scoring, an
optional external (Claude) review, and arbitration — then writes a detailed Markdown report.

> Default behavior is **100% local**: with no API keys and no provider overrides, every role runs on
> Ollama and no external calls are made.

## Architecture (end-to-end flow)

```
task
 └─ prompt refiner (optional, REFINER_ENABLED)   # rewrite vague task into a clearer instruction
 └─ memory retrieval (optional)                   # inject learnings from previous runs
 └─ constraint extraction                         # protected paths / no-delete / allowlist
 └─ planner                                       # structured implementation plan
 └─ retrieval (Ripgrep + Tree-sitter)             # locate precise file/symbol context
 └─ coder                                         # structured patch (file ops + proposed commands)
 └─ reflection (self-critique, up to 2 passes)    # catch scope drift / risks before applying
 └─ patch + constraint validation                 # path safety, auto-repair, constraint checks
 └─ apply files                                    # create/update files in the workspace
 └─ build / lint / test                           # configurable validation commands
 └─ command execution (optional, EXECUTE_COMMANDS) # run the coder's proposed commands
 └─ self-healing repair loop                       # classify → retrieve → repair → rollback on fail
 └─ confidence engine → review router              # score the result, decide if review is needed
 └─ Claude reviewer (optional, budget-gated)       # external review when confidence is low
 └─ arbitration                                    # reconcile validation vs. reviewers (validation wins)
 └─ Markdown + JSON report
```

## Providers & roles

Each **role** independently resolves a **provider** and a **model**, so you can mix local and cloud
freely (e.g. a local planner with a cloud coder).

Roles: `planner`, `coder`, `refiner`, `constraint`, `repair`, `reflection`, `reviewer`.

- **Provider** comes from `<ROLE>_PROVIDER` (one of `ollama`, `openai`, `anthropic`, `google`).
  Base roles (`planner`, `coder`) default to `ollama`. Derived roles inherit when left blank:
  `constraint ← planner`, `repair`/`reflection ← coder`, `refiner`/`reviewer ← planner`.
- **Model** comes from `<ROLE>_MODEL`, with the same inheritance. Legacy `PLANNER_MODEL` /
  `CODER_MODEL` still drive everything.
- **Credentials** are only required when a role actually points at that provider. The `openai`
  provider also covers any OpenAI-compatible gateway (OpenRouter, Groq, Together, DeepSeek,
  Fireworks, local vLLM / LM Studio) — just change `OPENAI_BASE_URL` + key + model.

A **preflight check** runs before each task: it verifies Ripgrep and the Tree-sitter grammars are
available, that local models are pulled, and that any selected cloud role has its key set — failing
fast with a clear, secret-free message. API keys are never logged or written to reports.

## Key features

- **Provider abstraction** — per-role provider+model routing with sensible inheritance; local by
  default, cloud when you opt in.
- **Optional prompt refiner** — rewrites a raw task into a structured instruction (clarified goal,
  assumptions, acceptance criteria) before planning; off by default and fails open to the raw prompt.
- **Retrieval layer** — indexes the repo (symbols, components) with Ripgrep + Tree-sitter to feed the
  coder precise context instead of the whole repo.
- **Reflection** — a self-critique pass (up to two) that can regenerate a patch before it is applied.
- **Validation + self-healing repair** — runs build/lint/test, classifies failures, retrieves
  error-relevant context, generates repair patches, and rolls back if it exceeds the attempt budget.
- **Optional command execution** — runs the coder's proposed shell commands (off by default); a
  non-zero exit feeds the same repair loop.
- **Confidence, review routing & arbitration** — scores each run, escalates low-confidence results to
  an optional budget-gated external (Claude) reviewer, and lets validation override a hallucinated
  approval.
- **Reporting** — a Markdown + JSON report covering refinement (if used), plan, retrieved context,
  validation, repair history, executed/proposed commands, review, and arbitration.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) running locally (for the default local setup)
- [`rg` (Ripgrep)](https://github.com/BurntSushi/ripgrep) installed and on your `PATH`

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment.** Copy the reference file and edit it. Everything has a working default;
   with nothing set, the agent runs fully local.
   ```bash
   cp .env.example .env
   ```
   A minimal local `.env`:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   PLANNER_MODEL=qwen2.5:14b
   CODER_MODEL=qwen2.5-coder:32b

   WORKSPACE_DIR=./workspace
   LOG_LEVEL=INFO

   BUILD_COMMAND=python -m compileall .
   TEST_COMMAND=python -m unittest discover
   MAX_REPAIR_ATTEMPTS=3

   # Off by default — the agent proposes commands but does not run them.
   EXECUTE_COMMANDS=false
   ```
   See [.env.example](.env.example) for the complete, commented reference (per-role providers/models,
   cloud credentials + base_url, refiner, timeouts, retries).

3. **Pull the Ollama models** you configured, e.g.:
   ```bash
   ollama pull qwen2.5:14b
   ollama pull qwen2.5-coder:32b
   ```

### Using a cloud model for a role (optional)

```env
# Local planner, cloud coder via OpenRouter (an OpenAI-compatible gateway):
CODER_PROVIDER=openai
CODER_MODEL=anthropic/claude-3.5-sonnet
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-...
```

## CLI usage

```bash
# Index the workspace (symbol + repo map under workspace/index/)
python main.py index
python main.py index --reindex        # force a full rebuild

# Exercise retrieval/scoring without invoking the LLM
python main.py search "jwt authentication"

# Print parsed Tree-sitter symbols
python main.py symbols

# Run an end-to-end task (index is updated automatically first)
python main.py "Create calculator.py with add and subtract functions."
```
> Run `python main.py` with no arguments for interactive multiline input (submit with `Ctrl+Z` on
> Windows, `Ctrl+D` on Linux/Mac).

## Project structure

```
agent/
├── orchestrator.py   # Coordinates the full pipeline above
├── config.py         # Environment-driven settings
├── llm/              # Provider abstraction, factory (role→provider+model), preflight
│   └── providers/    # ollama, openai (+ compatible gateways), anthropic, google
├── refiner/          # Optional prompt refinement stage
├── retrieval/        # Ripgrep + Tree-sitter indexing and context search
├── planner/          # Task decomposition
├── coder/            # Structured code generation
├── validation/       # Pre-apply checks + build/lint/test execution
├── repair/           # Self-healing repair patches + rollback
├── execution/        # Async terminal command runner
├── review/           # Confidence engine, review router, arbitration
├── reflection/       # Self-critique stage
├── reviewers/        # External (Claude) reviewer
├── reporting/        # Markdown + JSON run reports
└── ...               # memory, models, evaluation
main.py               # CLI entry point
```

## Testing

```bash
./venv/Scripts/python.exe -m pytest -q     # Windows venv
# or: python -m pytest -q
```

## License

Provided as-is for local and educational use.
