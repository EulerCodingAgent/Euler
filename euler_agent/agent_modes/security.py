"""Security specialist mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="security",
    summary="Security review specialist mode.",
    responsibility=(
        "Identify vulnerabilities, classify severity, and provide concrete remediation "
        "steps/code for auth, input handling, secrets, and abuse resistance."
    ),
    prompt_preamble=(
        "Mode: SECURITY. Enforce PLAN -> FIND -> EXECUTE. PLAN defines threat-check "
        "order, FIND identifies concrete attack surfaces and risky paths, EXECUTE "
        "returns severity-ranked remediations with production-safe fixes. Keep it "
        "concise, actionable, and example-free."
    ),
    examples=(
        "audit auth flow in @api/auth.py for OWASP issues",
        "review SQL handling for injection risks",
        "check secret management and token leakage paths",
    ),
    strategy="ask_specialist",
    specialist_role="security",
)
