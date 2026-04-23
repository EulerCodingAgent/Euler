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

## Features in this version

- Provider/model selection: OpenAI, Anthropic, Gemini
- API key managed locally in `~/.euler_agent/config.json`
- Interactive REPL with:
  - free-form coding prompts
  - SQL generation via `/sql`
  - selected code replacement via `/replace <file> <start> <end> <instruction>`
- Multi-agent orchestration scaffold (planner -> parallel workers -> reviewer) powered by LangGraph
- Support for project instruction memory in `./Euler/*.md`
- Avatar/banner shown when CLI activates

## Project instruction memory

Initialize instruction docs:

```bash
Euler init
```

This creates `Euler/project.md` in your current repo. Add architecture notes, coding rules, or domain constraints there. Euler injects those docs into agent context automatically.

## Roadmap toward full Claude-Code parity

The current release is a foundation and architecture scaffold. To reach full parity, next milestones are:

- deeper file-aware tool calling and patch planning
- autonomous command execution loops with retries
- stronger long-term memory + embeddings
- richer specialized agents (architect/coder/tester/reviewer with conflict handling)
- advanced code intelligence (repo graph, semantic indexing, refactor primitives)