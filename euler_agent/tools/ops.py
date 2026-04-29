"""Basic local tools used by the coding agent."""

from __future__ import annotations

import subprocess
import shlex
from ast import parse as ast_parse
from pathlib import Path
from typing import Iterable


def read_file(path: str) -> str:
    target = Path(path)
    return target.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {target}"


def append_file(path: str, content: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(content)
    return f"Appended to {target}"


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


def replace_in_files(paths: Iterable[str], search: str, replacement: str) -> str:
    """
    Perform a simple text replacement across multiple files.
    """
    changed: list[str] = []
    rejected: list[str] = []
    for raw_path in paths:
        target = Path(raw_path)
        if not target.exists():
            continue
        original = target.read_text(encoding="utf-8")
        updated = original.replace(search, replacement)
        if updated != original:
            if target.suffix == ".py":
                try:
                    ast_parse(updated)
                except SyntaxError:
                    rejected.append(str(target))
                    continue
            target.write_text(updated, encoding="utf-8")
            changed.append(str(target))
    if not changed and not rejected:
        return "No files changed."
    parts: list[str] = []
    if changed:
        parts.append(f"Updated {len(changed)} files: {', '.join(changed)}")
    if rejected:
        parts.append(f"Rejected {len(rejected)} files due to Python syntax risk: {', '.join(rejected)}")
    return " | ".join(parts)


def run_terminal_command(command: str, cwd: str | None = None) -> str:
    """
    Execute command safely without invoking a shell.
    """
    parsed = shlex.split(command, posix=False)
    if not parsed:
        return "exit_code=1\nstdout:\n\nstderr:\nEmpty command."
    completed = subprocess.run(
        parsed,
        shell=False,
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
