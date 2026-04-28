"""Coder specialist mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="coder",
    summary="Implementation specialist mode.",
    responsibility=(
        "Deliver production-ready complete code with imports, error handling, and "
        "drop-in file outputs."
    ),
    prompt_preamble=(
        "Mode: CODER. Prefer complete, runnable code. Keep explanations short unless "
        "asked; prioritize correct file outputs and implementation details."
    ),
    examples=(
        "implement JWT middleware in @auth.py",
        "add retry + timeout handling in @client.py",
        "create endpoint and service for password reset",
    ),
    strategy="ask_specialist",
    specialist_role="coder",
)
