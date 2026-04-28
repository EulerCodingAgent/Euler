"""DevOps specialist mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="devops",
    summary="Infrastructure and delivery specialist mode.",
    responsibility=(
        "Design deployment pipeline, containerization, CI/CD, runtime health checks, "
        "logging, and environment configuration."
    ),
    prompt_preamble=(
        "Mode: DEVOPS. Focus on deployability, reliability, and operational clarity. "
        "Prefer actionable infra artifacts and commands."
    ),
    examples=(
        "create Dockerfile + compose for local stack",
        "add GitHub Actions CI for tests and lint",
        "design rollout + rollback runbook",
    ),
    strategy="ask_specialist",
    specialist_role="devops",
)
