"""Database specialist mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="db",
    summary="Database engineering specialist mode.",
    responsibility=(
        "Design schemas, migrations, indexing strategy, and query performance "
        "improvements with safe parameterized access."
    ),
    prompt_preamble=(
        "Mode: DB. Enforce PLAN -> FIND -> EXECUTE. PLAN sequences schema and "
        "migration work, FIND identifies data constraints and query hotspots, "
        "EXECUTE returns production-grade DDL/migration/query updates with safe "
        "parameterized patterns and rollback readiness."
    ),
    examples=(
        "design schema for orders/payments and migration plan",
        "optimize slow query in @repository.py",
        "create reversible migration for new user_roles table",
    ),
    strategy="ask_specialist",
    specialist_role="db",
)
