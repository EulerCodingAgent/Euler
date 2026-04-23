"""Safety checks and rollback guards for code edits."""

from __future__ import annotations

import ast
from pathlib import Path


def validate_python_syntax(content: str) -> tuple[bool, str]:
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} at line {exc.lineno}"


def guarded_write(path: str, new_content: str) -> str:
    target = Path(path)
    previous = target.read_text(encoding="utf-8") if target.exists() else None
    if target.suffix == ".py":
        ok, message = validate_python_syntax(new_content)
        if not ok:
            return f"Rejected write for {target}: invalid Python syntax ({message})"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new_content, encoding="utf-8")
    return f"Wrote {target}"


def guarded_replace_range(path: str, start_line: int, end_line: int, replacement: str) -> str:
    target = Path(path)
    original_lines = target.read_text(encoding="utf-8").splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(original_lines), end_line)
    merged = original_lines[:start_idx] + replacement.splitlines() + original_lines[end_idx:]
    merged_content = "\n".join(merged) + "\n"
    if target.suffix == ".py":
        ok, message = validate_python_syntax(merged_content)
        if not ok:
            return (
                f"Rejected replace for {target} lines {start_line}-{end_line}: "
                f"invalid Python syntax ({message})"
            )
    target.write_text(merged_content, encoding="utf-8")
    return f"Replaced lines {start_line}-{end_line} in {target}"
