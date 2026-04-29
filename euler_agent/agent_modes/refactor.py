"""Refactor specialist mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="refactor",
    summary="Code quality/refactor specialist mode.",
    responsibility=(
        "Improve maintainability and clarity without changing external behavior, "
        "including duplication removal, complexity reduction, and naming improvements."
    ),
    prompt_preamble=(
        "Mode: REFACTOR. Enforce PLAN -> FIND -> EXECUTE. PLAN defines "
        "behavior-preserving change order, FIND identifies duplication and complexity "
        "hotspots, EXECUTE returns production-grade refactors with unchanged external "
        "behavior and clear structural improvements."
    ),
    examples=(
        "refactor @euler_agent/repl.py into smaller handlers",
        "remove duplication in cache read/write paths",
        "simplify complex branching in autopilot loop",
    ),
    strategy="ask_specialist",
    specialist_role="refactor",
)
