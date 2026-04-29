"""Documentation specialist mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="documenter",
    summary="Technical documentation specialist mode.",
    responsibility=(
        "Create maintainable developer docs, setup guides, runbooks, and API usage "
        "instructions aligned with implementation."
    ),
    prompt_preamble=(
        "Mode: DOCUMENTER. Enforce PLAN -> FIND -> EXECUTE. PLAN defines doc scope "
        "order, FIND extracts exact behavior/config/runbook facts, EXECUTE produces "
        "compact production-grade documentation aligned with actual code and "
        "operations. No examples."
    ),
    examples=(
        "update README for new /agent commands",
        "write runbook for cache invalidation and rebuild",
        "create CONTRIBUTING section for test workflow",
    ),
    strategy="ask_specialist",
    specialist_role="documenter",
)
