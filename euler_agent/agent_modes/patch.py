"""Patch-first REPL mode."""

from __future__ import annotations

from euler_agent.agent_modes.types import AgentModeSpec

SPEC = AgentModeSpec(
    name="patch",
    summary="Single-call patch generation mode.",
    responsibility=(
        "Generate patch-ready complete file outputs optimized for direct review/apply, "
        "especially for tasks with @file references."
    ),
    prompt_preamble=(
        "Mode: PATCH. Enforce PLAN -> FIND -> EXECUTE. PLAN states exact patch "
        "intent, FIND confirms touched files and constraints, EXECUTE returns "
        "complete updated production-grade file content line by line for direct "
        "apply. Avoid long prose and examples."
    ),
    examples=(
        "fix @euler_agent/repl.py command parsing",
        "add tests in @tests/test_cache.py",
        "update @README.md feature docs",
    ),
    strategy="ask_patch",
)
