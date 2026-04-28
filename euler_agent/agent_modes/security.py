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
        "Mode: SECURITY. Prioritize threat modeling and exploitability. Output findings "
        "with severity and concrete fixes."
    ),
    examples=(
        "audit auth flow in @api/auth.py for OWASP issues",
        "review SQL handling for injection risks",
        "check secret management and token leakage paths",
    ),
    strategy="ask_specialist",
    specialist_role="security",
)
