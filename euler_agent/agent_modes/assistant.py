"""Assistant-first REPL mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="assistant",
    summary="Single-call assistant mode.",
    responsibility=(
        "Focus on explanations, analysis, and targeted guidance without forcing "
        "multi-agent orchestration."
    ),
    prompt_preamble=(
        "Mode: ASSISTANT. Prioritize clarity and correctness. Explain assumptions, "
        "trade-offs, and give practical next steps."
    ),
    examples=(
        "why is semantic cache useful here?",
        "review this approach in @benchmark.py",
        "what is the risk in this SQL migration?",
    ),
    strategy="ask_single",
)
