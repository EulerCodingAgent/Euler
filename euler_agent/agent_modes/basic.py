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
        "Mode: BASIC. Use concise, high-signal output. Prefer direct answers for "
        "questions and patch-ready code blocks for concrete edits."
    ),
    examples=(
        "explain @auth.py:10-40",
        "fix @euler_agent/repl.py autocomplete bug",
        "implement retry logic for API client",
    ),
    strategy="heuristic",
)
