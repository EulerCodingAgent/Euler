"""Autopilot guardrails for command and file safety."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_ALLOWED_COMMAND_PREFIXES = (
    "python",
    "pytest",
    "pip",
    "uv",
    "poetry",
    "npm",
    "pnpm",
    "yarn",
    "git status",
    "git diff",
)

BLOCKED_COMMAND_TOKENS = ("&&", ";", "|", ">", "<", "\n", "\r", "rm ", "rmdir ", "del ", "format ", "`")
RISKY_COMMAND_HINTS = (
    "git push",
    "git commit",
    "git checkout",
    "git reset",
    "git clean",
    "docker",
    "kubectl",
    "terraform",
)

POLICY_PROFILES = ("safe", "normal", "aggressive")


@dataclass
class AutopilotPolicy:
    max_file_mutations: int = 25
    max_actions_per_round: int = 8
    allowed_command_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_COMMAND_PREFIXES
    profile: str = "normal"
    require_approval_for_risky: bool = True


def build_policy(
    profile: str = "normal",
    max_file_mutations: int = 25,
    require_approval_for_risky: bool = True,
) -> AutopilotPolicy:
    normalized = profile.lower().strip()
    if normalized not in POLICY_PROFILES:
        normalized = "normal"

    if normalized == "safe":
        return AutopilotPolicy(
            max_file_mutations=min(max_file_mutations, 10),
            max_actions_per_round=5,
            allowed_command_prefixes=("python", "pytest", "uv", "pip", "git status", "git diff"),
            profile=normalized,
            require_approval_for_risky=True,
        )
    if normalized == "aggressive":
        return AutopilotPolicy(
            max_file_mutations=max(max_file_mutations, 60),
            max_actions_per_round=12,
            allowed_command_prefixes=(
                "python",
                "pytest",
                "pip",
                "uv",
                "poetry",
                "npm",
                "pnpm",
                "yarn",
                "git status",
                "git diff",
                "git add",
                "git commit",
            ),
            profile=normalized,
            require_approval_for_risky=require_approval_for_risky,
        )
    return AutopilotPolicy(
        max_file_mutations=max_file_mutations,
        max_actions_per_round=8,
        allowed_command_prefixes=DEFAULT_ALLOWED_COMMAND_PREFIXES,
        profile=normalized,
        require_approval_for_risky=require_approval_for_risky,
    )


def ensure_inside_workdir(workdir: str, target: str) -> bool:
    root = Path(workdir).resolve()
    resolved = Path(target).resolve()
    return root == resolved or root in resolved.parents


def is_command_allowed(command: str, policy: AutopilotPolicy) -> tuple[bool, str]:
    normalized = command.strip()
    lowered = normalized.lower()
    for token in BLOCKED_COMMAND_TOKENS:
        if token in lowered:
            return False, f"Blocked token detected: {token!r}"
    if not normalized:
        return False, "Empty command is not allowed."
    if not any(lowered.startswith(prefix.lower()) for prefix in policy.allowed_command_prefixes):
        return False, (
            "Command prefix is not in allowlist. "
            f"Allowed prefixes: {', '.join(policy.allowed_command_prefixes)}"
        )
    return True, ""


def is_risky_command(command: str) -> bool:
    lowered = command.strip().lower()
    return any(lowered.startswith(hint) for hint in RISKY_COMMAND_HINTS)
