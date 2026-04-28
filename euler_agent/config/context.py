"""Project instruction and repository context helpers."""

from __future__ import annotations

from pathlib import Path


def load_euler_instruction_docs(workdir: Path) -> str:
    """
    Load instruction files from ./Euler/ as supporting project instructions.

    Supported:
      - Markdown (*.md)
      - JSON (*.json)
    """
    instruction_dir = workdir / "Euler"
    if not instruction_dir.exists() or not instruction_dir.is_dir():
        return ""

    blocks: list[str] = []
    patterns = ("*.md", "*.json")
    files = sorted(
        f
        for pattern in patterns
        for f in instruction_dir.glob(pattern)
        if f.is_file()
    )
    for file_path in files:
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        blocks.append(f"# {file_path.name}\n{content}")

    return "\n\n".join(blocks)
