"""Architect specialist mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="architect",
    summary="Systems architecture specialist mode.",
    responsibility=(
        "Design module boundaries, interfaces, data flow, dependency graph, and "
        "cross-cutting concerns before implementation."
    ),
    prompt_preamble=(
        "Mode: ARCHITECT. Enforce PLAN -> FIND -> EXECUTE. PLAN defines system-level "
        "sequencing, FIND maps current module boundaries and constraints, EXECUTE "
        "outputs production-grade architecture decisions, contracts, and risk controls "
        "with deterministic wording only."
    ),
    examples=(
        "design auth service boundaries and API contracts",
        "propose module structure for payment + invoicing",
        "create dependency graph for refactor plan",
    ),
    strategy="ask_specialist",
    specialist_role="architect",
)
