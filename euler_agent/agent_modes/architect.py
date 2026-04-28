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
        "Mode: ARCHITECT. Produce architecture-first output: modules, contracts, "
        "trade-offs, and risk notes. Avoid low-level implementation detail unless required."
    ),
    examples=(
        "design auth service boundaries and API contracts",
        "propose module structure for payment + invoicing",
        "create dependency graph for refactor plan",
    ),
    strategy="ask_specialist",
    specialist_role="architect",
)
