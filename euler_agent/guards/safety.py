"""Safety checks and rollback guards for code edits."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


def validate_python_syntax(content: str) -> tuple[bool, str]:
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} at line {exc.lineno}"


def _balanced_symbols(content: str, pairs: tuple[tuple[str, str], ...]) -> bool:
    stack: list[str] = []
    open_to_close = {o: c for o, c in pairs}
    close_to_open = {c: o for o, c in pairs}
    for ch in content:
        if ch in open_to_close:
            stack.append(ch)
        elif ch in close_to_open:
            if not stack or stack[-1] != close_to_open[ch]:
                return False
            stack.pop()
    return not stack


def validate_by_extension(path: Path, content: str) -> tuple[bool, str]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return validate_python_syntax(content)
    if suffix in {".json", ".jsonc"}:
        try:
            cleaned = re.sub(r"//.*?$|/\*.*?\*/", "", content, flags=re.MULTILINE | re.DOTALL)
            json.loads(cleaned)
            return True, ""
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON ({exc.msg})"
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        if not _balanced_symbols(content, (("{", "}"), ("(", ")"), ("[", "]"))):
            return False, "unbalanced braces/brackets/parentheses"
        return True, ""
    if suffix == ".sql":
        if not _balanced_symbols(content, (("(", ")"),)):
            return False, "unbalanced SQL parentheses"
        if content.count("'") % 2 != 0:
            return False, "unbalanced SQL single quotes"
        return True, ""
    return True, ""


def guarded_write(path: str, new_content: str) -> str:
    target = Path(path)
    previous = target.read_text(encoding="utf-8") if target.exists() else None
    ok, message = validate_by_extension(target, new_content)
    if not ok:
        return f"Rejected write for {target}: validation failed ({message})"
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
    ok, message = validate_by_extension(target, merged_content)
    if not ok:
        return (
            f"Rejected replace for {target} lines {start_line}-{end_line}: "
            f"validation failed ({message})"
        )
    target.write_text(merged_content, encoding="utf-8")
    return f"Replaced lines {start_line}-{end_line} in {target}"
