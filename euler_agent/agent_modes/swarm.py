"""Full multi-agent orchestration mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="swarm",
    summary="Always run full multi-agent pipeline.",
    responsibility=(
        "Use planner + specialists + reviewer orchestration for complex implementation "
        "work that benefits from parallel role decomposition."
    ),
    prompt_preamble=(
        "Mode: SWARM. Enforce PLAN -> FIND -> EXECUTE across planner, specialists, "
        "arbitrator, and reviewer. Require production-grade compact outputs with "
        "deterministic file-level actions and line-by-line code when implementation "
        "is needed."
    ),
    examples=(
        "build auth module with RBAC and tests",
        "refactor data layer and add migrations",
        "design and implement audit logging end-to-end",
    ),
    strategy="run_graph",
)
