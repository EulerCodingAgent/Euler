"""Shared types for REPL agent-mode definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentModeSpec:
    name: str
    summary: str
    responsibility: str
    prompt_preamble: str
    examples: tuple[str, ...]
    strategy: str
    specialist_role: str | None = None
