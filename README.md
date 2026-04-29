# Euler Agent CLI

Euler is a local-first coding agent CLI inspired by tools like Claude Code.
It runs in your terminal, uses your own API key or local endpoint, and supports OpenAI, Anthropic, Gemini, Ollama, and OpenAI-compatible local servers.

## Install

```bash
pip install Euler-Agent
```

For local development:

```bash
pip install -e .
```

## Run

```bash
# default is already Gemini — just set your key:
Euler config set --provider gemini --model gemini-2.5-flash
Euler

# Ollama local model (example: qwen/gemma served by Ollama):
Euler config set --provider ollama --model qwen2.5-coder:7b

# Generic local OpenAI-compatible server (LM Studio/vLLM/etc):
Euler config set --provider local --model kimi-k2-instruct --base-url http://localhost:1234/v1
```

You can also run one-shot tasks:

```bash
Euler run "Build a FastAPI todo service with tests"
```

Autonomous build/fix loop:

```bash
Euler autopilot "Fix failing tests and refactor auth module" --verify-command "pytest -q"
Euler autopilot "Refactor auth module" --policy-profile safe
Euler autopilot "Implement feature X" --policy-profile aggressive --auto-approve-risky
```

Convert a file to another language:

```bash
Euler convert app.py typescript --output app.ts
```

Build semantic index and search code:

```bash
Euler index
Euler index --full
Euler search-code "where is auth token validation implemented"
Euler graph
Euler knowledge-graph
Euler knowledge-graph src/
Euler role-map
Euler role-map --list
```

## Features in this version

- Provider/model selection: OpenAI, Anthropic, Gemini, Ollama, Local OpenAI-compatible
- API key managed locally in `~/.euler_agent/config.json`
- Interactive REPL with:
  - free-form coding prompts
  - SQL generation via `/sql`
  - selected code replacement via `/replace <file> <start> <end> <instruction>`
- **Default provider: Gemini (`gemini-2.5-flash`)**
- 8-specialist parallel agent swarm: architect, coder, tester, security, devops, db, documenter, refactor
- Production-grade code enforcement in all agents and autopilot writes
- Cross-language file conversion: `Euler convert <file> <lang>` and REPL `/convert`
- Autonomous execution loop with retries and command observations
- Guardrailed autopilot with command allowlist, workdir sandbox, and mutation limits
- Multi-agent orchestration with specialist roles (planner -> architect/coder/tester in parallel -> arbitrator -> reviewer)
- Support for project instruction memory in `./Euler-Knowledge/*.md`
- Persistent long-term memory for previous project goals/results (`Euler memory "<query>"`)
- Embedding-backed semantic memory retrieval (local sparse TF-IDF; zero native deps)
- AST-aware patch safety guards for Python edits in autonomous mode
- Repo semantic indexing for codebase-wide natural language retrieval (`Euler index`, `Euler search-code`)
- Incremental semantic index updates with optional full rebuild (`Euler index`, `Euler index --full`)
- Cross-language graph extraction for Python + TS/JS + SQL symbols/relations (`Euler graph`)
- Knowledge-graph build to `./Euler-Knowledge/knowledge_graph*.json` for root or selected folder (`Euler knowledge-graph [folder]`, REPL `/knowledge-graph [folder]`)
- Knowledge file responsibility mapping via `Euler role-map` with listing via `Euler role-map --list`
- Graph-aware retrieval expansion (uses `code_graph.json` neighbors to improve context precision)
- Semantic response cache with approximate query matching for near-identical prompts
- Memory lesson cards (compact summaries) to reduce noisy long-history prompt injection
- Adaptive specialist early-exit for low-risk focused tasks
- Prompt-delta autopilot context to limit multi-round context bloat
- Transactional round snapshots with rollback on command/verification failure in autopilot
- Policy profiles (`safe`, `normal`, `aggressive`) for autopilot guardrails
- Approval-gated risky commands/actions with explicit auto-approve flag
- Structured audit logs for each autopilot run in `.euler/audit/*.jsonl`
- Avatar/banner shown when CLI activates

## Project instruction memory

Initialize instruction docs:

```bash
Euler init
```

This creates `Euler-Knowledge/project.md` in your current repo. Add architecture notes, coding rules, or domain constraints there. Euler injects those docs into agent context automatically.

## Remaining roadmap toward full Claude-Code parity

The current release is a foundation and architecture scaffold. To reach full parity, next milestones are:

- richer tool/plugin protocol compatibility with external MCP-style tools
- higher autonomy for full app build/deploy loops with explicit approvals
- remote team audit sinks and signed execution logs