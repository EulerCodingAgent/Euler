"""Project instruction and repository context helpers."""

from __future__ import annotations

from pathlib import Path


def load_euler_instruction_docs(workdir: Path) -> str:
    """
    Load markdown files from ./Euler/ as supporting project instructions.
    """
    instruction_dir = workdir / "Euler"
    if not instruction_dir.exists() or not instruction_dir.is_dir():
        return ""

    blocks: list[str] = []
    for md_file in sorted(instruction_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        blocks.append(f"# {md_file.name}\n{content}")

    return "\n\n".join(blocks)
