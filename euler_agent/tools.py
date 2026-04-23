"""Basic local tools used by the coding agent."""

from __future__ import annotations

import subprocess
from pathlib import Path


def read_file(path: str) -> str:
    target = Path(path)
    return target.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {target}"


def replace_range(path: str, start_line: int, end_line: int, replacement: str) -> str:
    """
    Replace a selected line range in a file (1-indexed, inclusive).
    """
    target = Path(path)
    lines = target.read_text(encoding="utf-8").splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    new_lines = lines[:start_idx] + replacement.splitlines() + lines[end_idx:]
    target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return f"Replaced lines {start_line}-{end_line} in {target}"


def run_terminal_command(command: str, cwd: str | None = None) -> str:
    """
    Execute shell command and capture both output streams.
    """
    completed = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return (
        f"exit_code={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
