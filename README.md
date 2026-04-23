# Euler Agent CLI

Euler is a local-first coding agent CLI inspired by tools like Claude Code.
It runs in your terminal, uses your own API key, and supports OpenAI, Anthropic, and Gemini.

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
Euler config set --provider openai --model gpt-4o-mini
Euler
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

Build semantic index and search code:

```bash
Euler index
Euler index --full
Euler search-code "where is auth token validation implemented"
Euler graph
```

## Features in this version

- Provider/model selection: OpenAI, Anthropic, Gemini
- API key managed locally in `~/.euler_agent/config.json`
- Interactive REPL with:
  - free-form coding prompts
  - SQL generation via `/sql`
  - selected code replacement via `/replace <file> <start> <end> <instruction>`
- Autonomous execution loop with retries and command observations
- Guardrailed autopilot with command allowlist, workdir sandbox, and mutation limits
- Multi-agent orchestration with specialist roles (planner -> architect/coder/tester in parallel -> arbitrator -> reviewer)
- Support for project instruction memory in `./Euler/*.md`
- Persistent long-term memory for previous project goals/results (`Euler memory "<query>"`)
- Embedding-backed semantic memory retrieval (local vector search via FastEmbed)
- AST-aware patch safety guards for Python edits in autonomous mode
- Repo semantic indexing for codebase-wide natural language retrieval (`Euler index`, `Euler search-code`)
- Incremental semantic index updates with optional full rebuild (`Euler index`, `Euler index --full`)
- Cross-language graph extraction for Python + TS/JS + SQL symbols/relations (`Euler graph`)
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

This creates `Euler/project.md` in your current repo. Add architecture notes, coding rules, or domain constraints there. Euler injects those docs into agent context automatically.

## Remaining roadmap toward full Claude-Code parity

The current release is a foundation and architecture scaffold. To reach full parity, next milestones are:

- richer tool/plugin protocol compatibility with external MCP-style tools
- higher autonomy for full app build/deploy loops with explicit approvals
- remote team audit sinks and signed execution logs