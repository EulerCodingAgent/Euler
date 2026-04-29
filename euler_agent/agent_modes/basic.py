"""Default heuristic REPL mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="basic",
    summary="Default smart routing mode.",
    responsibility=(
        "Auto-route user requests based on intent: quick Q&A for questions, "
        "patch flow for file-referenced edits, and full swarm run for broad tasks."
    ),
    prompt_preamble=(
        "Mode: BASIC. Enforce skill workflow PLAN -> FIND -> EXECUTE. PLAN defines "
        "ordered production steps, FIND maps exact files/symbols/constraints, EXECUTE "
        "returns compact implementation-grade output with line-by-line code only when "
        "code is required. No placeholders, no examples, no unnecessary prose."
    ),
    examples=(
        "explain @auth.py:10-40",
        "fix @euler_agent/repl.py autocomplete bug",
        "implement retry logic for API client",
    ),
    strategy="heuristic",
)
