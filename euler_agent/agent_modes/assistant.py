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
        "Mode: ASSISTANT. Enforce PLAN -> FIND -> EXECUTE. PLAN states assumptions "
        "and decision points, FIND identifies concrete evidence in referenced code, "
        "EXECUTE gives compact production-safe guidance or exact edits only when "
        "requested. Keep responses tight, no examples, no filler."
    ),
    examples=(
        "why is semantic cache useful here?",
        "review this approach in @benchmark.py",
        "what is the risk in this SQL migration?",
    ),
    strategy="ask_single",
)
