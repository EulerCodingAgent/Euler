"""Tester specialist mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="tester",
    summary="Test engineering specialist mode.",
    responsibility=(
        "Create robust unit/integration tests, edge-case coverage, and failure-path "
        "validation based on project conventions."
    ),
    prompt_preamble=(
        "Mode: TESTER. Output executable tests with meaningful assertions, fixtures, "
        "and clear scenario names."
    ),
    examples=(
        "write tests for @euler_agent/optimization/token_optimizer.py",
        "add regression tests for cache key collision bug",
        "add integration tests for /agent mode commands",
    ),
    strategy="ask_specialist",
    specialist_role="tester",
)
