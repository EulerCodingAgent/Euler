"""Compact skill contracts for production-grade agent execution."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SkillStep(str, Enum):
    PLAN = "plan"
    FIND = "find"
    EXECUTE = "execute"


class SkillContract(BaseModel):
    """Minimal contract every role follows for deterministic execution."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    steps: tuple[SkillStep, ...] = (SkillStep.PLAN, SkillStep.FIND, SkillStep.EXECUTE)
    production_only: bool = True
    line_by_line_code: bool = True


def render_skill_protocol(role: str, objective: str) -> str:
    """Render strict workflow instructions injected into prompts."""
    contract = SkillContract(role=role, objective=objective)
    ordered_steps = " -> ".join(step.value.upper() for step in contract.steps)
    return (
        "Skill Contract (mandatory):\n"
        f"- Role: {contract.role}\n"
        f"- Objective: {contract.objective}\n"
        f"- Steps: {ordered_steps}\n"
        "- PLAN: list exact actions in execution order.\n"
        "- FIND: identify concrete files/functions/config and constraints.\n"
        "- EXECUTE: apply production-grade output line-by-line, no placeholders.\n"
        "- Output must be compact, deterministic, and implementation-first.\n"
        "- Do not include examples, filler, or speculative alternatives."
    )
