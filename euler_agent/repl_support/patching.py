"""Patch extraction and transactional apply utilities for the REPL."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

from euler_agent.guards.safety import validate_by_extension
from euler_agent.repl_support.constants import CODE_BLOCK_RE, FILE_PATH_COMMENT_RE, JSON_BLOCK_RE
from euler_agent.repl_support.models import PatchEnvelope, PatchTuple


def is_allowed_derived_write_path(candidate: Path, allowed_files: set[Path], workdir: Path) -> bool:
    if not allowed_files:
        return True
    try:
        rel = candidate.relative_to(workdir)
    except ValueError:
        return False
    if candidate.exists():
        return any(candidate == af for af in allowed_files)
    rel_norm = str(rel).replace("\\", "/")
    for af in allowed_files:
        try:
            af_rel = af.relative_to(workdir)
        except ValueError:
            continue
        af_rel_norm = str(af_rel).replace("\\", "/")
        if rel.parent == af_rel.parent and rel.stem == af_rel.stem:
            return True
        if candidate.stem == af.stem:
            return True
        if rel_norm == af_rel_norm:
            return True
    return False


def extract_patches(response: str, console: Console, workdir: Path, allowed_files: set[Path]) -> list[PatchTuple]:
    patches: list[PatchTuple] = []
    seen: set[Path] = set()
    for block_match in CODE_BLOCK_RE.finditer(response):
        fenced_block = response[block_match.start() : block_match.end()]
        if fenced_block.lstrip().lower().startswith("```json"):
            continue
        body = block_match.group(1)
        lines = body.splitlines()
        if not lines:
            continue
        path_match = FILE_PATH_COMMENT_RE.match(lines[0].strip())
        raw_path: str | None = path_match.group("path") if path_match else None
        code_lines = lines[1:] if raw_path else lines
        if raw_path is None:
            if len(allowed_files) == 1:
                candidate = next(iter(allowed_files))
            elif allowed_files:
                candidate = None
                header = "\n".join(lines[:5]).lower()
                for af in allowed_files:
                    if af.name.lower() in header:
                        candidate = af
                        break
                if candidate is None:
                    continue
            else:
                continue
        else:
            parsed = Path(raw_path)
            candidate = None
            for af in allowed_files:
                norm_raw = raw_path.replace("\\", "/")
                norm_af = str(af).replace("\\", "/")
                if af.name == parsed.name or norm_af.endswith("/" + norm_raw):
                    candidate = af
                    break
            if candidate is None:
                if allowed_files:
                    console.print(f"[yellow]Skipped {escape(raw_path)} — not part of attached files[/yellow]")
                    continue
                candidate = parsed if parsed.is_absolute() else (workdir / parsed).resolve()
        try:
            candidate.relative_to(workdir)
        except ValueError:
            console.print(f"[yellow]Skipped {escape(str(candidate))} — outside workdir[/yellow]")
            continue
        if not is_allowed_derived_write_path(candidate, allowed_files, workdir):
            console.print(f"[yellow]Skipped {escape(str(candidate))} — not an allowed derived target[/yellow]")
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        code = "\n".join(code_lines).strip() + "\n"
        if not code.strip():
            continue
        ok, message = validate_by_extension(candidate, code)
        if not ok:
            console.print(f"[yellow]Skipped {escape(candidate.name)} — validation failed: {escape(message)}[/yellow]")
            continue
        patches.append((candidate, code, "write"))
    return patches


def extract_json_protocol_patches(
    response: str, console: Console, workdir: Path, allowed_files: set[Path]
) -> list[PatchTuple] | None:
    match = JSON_BLOCK_RE.search(response)
    if not match:
        return []
    raw = match.group(1).strip()
    try:
        payload = PatchEnvelope.model_validate_json(raw)
    except ValidationError as exc:
        try:
            payload = PatchEnvelope.model_validate(parse_relaxed_patch_payload(raw))
        except Exception:
            try:
                payload = PatchEnvelope.model_validate(_salvage_patch_payload(raw))
            except Exception:
                console.print(f"[yellow]Invalid JSON patch payload: {escape(str(exc))}[/yellow]")
                return None
    except Exception as exc:
        try:
            payload = PatchEnvelope.model_validate(parse_relaxed_patch_payload(raw))
        except Exception:
            try:
                payload = PatchEnvelope.model_validate(_salvage_patch_payload(raw))
            except Exception:
                console.print(f"[yellow]Failed to parse JSON patch payload: {escape(str(exc))}[/yellow]")
                return None
    patches: list[PatchTuple] = []
    seen: set[Path] = set()
    for edit in payload.edits:
        raw_path = edit.path.strip()
        parsed = Path(raw_path)
        candidate: Path | None = None
        for af in allowed_files:
            norm_raw = raw_path.replace("\\", "/")
            norm_af = str(af).replace("\\", "/")
            if af.name == parsed.name or norm_af.endswith("/" + norm_raw):
                candidate = af
                break
        if candidate is None:
            candidate = parsed if parsed.is_absolute() else (workdir / parsed).resolve()
        try:
            candidate.relative_to(workdir)
        except ValueError:
            console.print(f"[yellow]Skipped {escape(str(candidate))} — outside workdir[/yellow]")
            continue
        if edit.operation == "write" and not is_allowed_derived_write_path(candidate, allowed_files, workdir):
            console.print(f"[yellow]Skipped {escape(str(candidate))} — not an allowed derived target[/yellow]")
            continue
        if edit.operation == "delete" and allowed_files and candidate not in allowed_files:
            console.print(f"[yellow]Skipped {escape(str(candidate))} — delete allowed only for attached files[/yellow]")
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if edit.operation == "delete":
            patches.append((candidate, None, "delete"))
            continue
        if edit.content is None:
            console.print(f"[yellow]Skipped {escape(candidate.name)} — missing content for write operation[/yellow]")
            continue
        content = edit.content if edit.content.endswith("\n") else (edit.content + "\n")
        ok, message = validate_by_extension(candidate, content)
        if not ok:
            console.print(f"[yellow]Skipped {escape(candidate.name)} — validation failed: {escape(message)}[/yellow]")
            continue
        patches.append((candidate, content, "write"))
    return patches


def parse_relaxed_patch_payload(raw: str) -> dict[str, Any]:
    i = 0
    n = len(raw)

    def skip_ws() -> None:
        nonlocal i
        while i < n and raw[i] in " \t\r\n":
            i += 1

    def parse_string() -> str:
        nonlocal i
        if i >= n or raw[i] != '"':
            raise ValueError("expected string")
        i += 1
        out: list[str] = []
        while i < n:
            ch = raw[i]
            if ch == "\\":
                i += 1
                if i >= n:
                    break
                esc = raw[i]
                out.append({"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}.get(esc, esc))
                i += 1
                continue
            if ch == '"':
                i += 1
                return "".join(out)
            out.append(ch)
            i += 1
        raise ValueError("unterminated string")

    def parse_literal() -> Any:
        nonlocal i
        if raw.startswith("null", i):
            i += 4
            return None
        if raw.startswith("true", i):
            i += 4
            return True
        if raw.startswith("false", i):
            i += 5
            return False
        raise ValueError("unsupported literal")

    def parse_value() -> Any:
        nonlocal i
        skip_ws()
        if i >= n:
            raise ValueError("unexpected end")
        if raw[i] == '"':
            return parse_string()
        if raw[i] == "{":
            return parse_object()
        if raw[i] == "[":
            return parse_array()
        return parse_literal()

    def parse_object() -> dict[str, Any]:
        nonlocal i
        if raw[i] != "{":
            raise ValueError("expected object")
        i += 1
        obj: dict[str, Any] = {}
        skip_ws()
        if i < n and raw[i] == "}":
            i += 1
            return obj
        while i < n:
            skip_ws()
            key = parse_string()
            skip_ws()
            if i >= n or raw[i] != ":":
                raise ValueError("expected ':'")
            i += 1
            obj[key] = parse_value()
            skip_ws()
            if i < n and raw[i] == ",":
                i += 1
                continue
            if i < n and raw[i] == "}":
                i += 1
                return obj
            raise ValueError("expected ',' or '}'")
        raise ValueError("unterminated object")

    def parse_array() -> list[Any]:
        nonlocal i
        if raw[i] != "[":
            raise ValueError("expected array")
        i += 1
        arr: list[Any] = []
        skip_ws()
        if i < n and raw[i] == "]":
            i += 1
            return arr
        while i < n:
            arr.append(parse_value())
            skip_ws()
            if i < n and raw[i] == ",":
                i += 1
                continue
            if i < n and raw[i] == "]":
                i += 1
                return arr
            raise ValueError("expected ',' or ']'")
        raise ValueError("unterminated array")

    skip_ws()
    payload = parse_object()
    skip_ws()
    if i != n:
        raise ValueError("trailing content")
    return payload


def _decode_json_like_string(raw: str) -> str:
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "r":
                out.append("\r")
            elif nxt == "t":
                out.append("\t")
            elif nxt == '"':
                out.append('"')
            elif nxt == "\\":
                out.append("\\")
            else:
                out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _salvage_patch_payload(raw: str) -> dict[str, Any]:
    """
    Best-effort salvage for near-JSON patch payloads.

    Handles model outputs where the top-level shape is preserved but `content`
    includes JSON-breaking text. We recover edits by scanning structural anchors.
    """
    edits: list[dict[str, Any]] = []
    cursor = 0
    while True:
        path_key = raw.find('"path":"', cursor)
        if path_key == -1:
            break
        path_start = path_key + len('"path":"')
        path_end = raw.find('"', path_start)
        if path_end == -1:
            break
        path_value = raw[path_start:path_end]

        op_key = raw.find('"operation":"', path_end)
        if op_key == -1:
            break
        op_start = op_key + len('"operation":"')
        op_end = raw.find('"', op_start)
        if op_end == -1:
            break
        operation = raw[op_start:op_end]

        if operation == "delete":
            edits.append({"path": path_value, "operation": "delete"})
            cursor = op_end + 1
            continue

        if operation != "write":
            cursor = op_end + 1
            continue

        content_key = raw.find('"content":"', op_end)
        if content_key == -1:
            edits.append({"path": path_value, "operation": "write", "content": ""})
            cursor = op_end + 1
            continue
        content_start = content_key + len('"content":"')

        next_edit_marker = raw.find('},{"path":"', content_start)
        end_marker = raw.find('"}]}', content_start)
        candidates = [idx for idx in [next_edit_marker, end_marker] if idx != -1]
        if not candidates:
            content_end = len(raw)
            cursor = len(raw)
        else:
            content_end = min(candidates)
            cursor = content_end + 1

        content_raw = raw[content_start:content_end]
        content = _decode_json_like_string(content_raw)
        edits.append({"path": path_value, "operation": "write", "content": content})

    if not edits:
        raise ValueError("Unable to salvage patch payload.")
    return {"edits": edits}


def _show_diff(console: Console, path: Path, old: str, new: str) -> None:
    diff_lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            lineterm="",
        )
    )
    if not diff_lines:
        console.print("  [dim](no textual changes)[/dim]")
        return
    console.print(Syntax("\n".join(diff_lines), "diff", theme="monokai", line_numbers=False))


def review_and_apply(patches: list[PatchTuple], console: Console, workdir: Path) -> list[Path]:
    if not patches:
        return []
    pending_ops: list[PatchTuple] = []
    apply_all = False
    for path, new_content, operation in patches:
        old_content = path.read_text(encoding="utf-8") if path.exists() else ""
        target_content = "" if operation == "delete" else (new_content or "")
        if old_content == target_content:
            console.print(f"[dim]{path.name}: no changes[/dim]")
            continue
        console.print()
        if operation == "delete":
            console.print(Panel(f"[bold red]Delete file:[/bold red] {escape(path.name)}", padding=(0, 1)))
        elif not path.exists():
            console.print(Panel(f"[bold cyan]New file:[/bold cyan] {escape(path.name)}", padding=(0, 1)))
        else:
            try:
                rel = path.relative_to(workdir)
            except ValueError:
                rel = path
            console.print(Panel(f"[bold cyan]Proposed changes:[/bold cyan] {escape(str(rel))}", padding=(0, 1)))
        _show_diff(console, path, old_content, target_content)
        if apply_all:
            pending_ops.append((path, new_content, operation))
            continue
        try:
            choice = console.input("\nApply? [y]es / [n]o / [a]ll / [q]uit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Cancelled.[/yellow]")
            break
        if choice in {"a", "all"}:
            apply_all = True
            choice = "y"
        if choice in {"y", "yes"}:
            pending_ops.append((path, new_content, operation))
        elif choice in {"q", "quit"}:
            console.print("[yellow]Stopped — remaining changes discarded.[/yellow]")
            break
    if not pending_ops:
        return []
    backups: dict[Path, str | None] = {}
    written: list[Path] = []
    try:
        for path, _, _ in pending_ops:
            backups[path] = path.read_text(encoding="utf-8") if path.exists() else None
        for path, new_content, operation in pending_ops:
            if operation == "delete":
                if path.exists():
                    path.unlink()
                written.append(path)
                continue
            payload = new_content or ""
            ok, message = validate_by_extension(path, payload)
            if not ok:
                raise RuntimeError(f"{path}: {message}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            written.append(path)
        return written
    except Exception as exc:
        for path, old in backups.items():
            if old is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(old, encoding="utf-8")
        console.print(f"[red]Patch transaction failed; rolled back all writes: {escape(str(exc))}[/red]")
        return []

