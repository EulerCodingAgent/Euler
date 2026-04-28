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
        "Mode: SWARM. Produce implementation-grade output with explicit files, tests, "
        "and deployment considerations."
    ),
    examples=(
        "build auth module with RBAC and tests",
        "refactor data layer and add migrations",
        "design and implement audit logging end-to-end",
    ),
    strategy="run_graph",
)
