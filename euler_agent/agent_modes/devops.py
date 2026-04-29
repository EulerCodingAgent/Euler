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
        "Mode: DEVOPS. Enforce PLAN -> FIND -> EXECUTE. PLAN defines rollout "
        "sequence, FIND identifies infra/runtime constraints, EXECUTE returns "
        "production-grade deployment and operations artifacts with strict reliability "
        "and security controls. No filler."
    ),
    examples=(
        "create Dockerfile + compose for local stack",
        "add GitHub Actions CI for tests and lint",
        "design rollout + rollback runbook",
    ),
    strategy="ask_specialist",
    specialist_role="devops",
)
