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

BLOCKED_COMMAND_TOKENS = ("&&", ";", "|", ">", "<", "rm ", "rmdir ", "del ", "format ")


@dataclass
class AutopilotPolicy:
    max_file_mutations: int = 25
    max_actions_per_round: int = 8
    allowed_command_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_COMMAND_PREFIXES


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
