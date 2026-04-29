"""Interactive REPL shell for Euler agent."""

from __future__ import annotations

import difflib
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from pydantic import BaseModel, ValidationError

from euler_agent.agent_modes import MODE_BY_NAME, MODE_SPECS
from euler_agent.core.agent import EulerAgent
from euler_agent.core.autopilot import run_autopilot
from euler_agent.analysis.code_graph import build_code_graph
from euler_agent.memory.store import search_memory
from euler_agent.analysis.semantic_index import index_path, search_index
from euler_agent.guards.safety import validate_by_extension
from euler_agent.tools.ops import read_file, replace_range, write_file

# ── prompt_toolkit (optional — graceful fallback to plain input) ──────────────
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    _PT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PT_AVAILABLE = False


# ── patterns & constants ──────────────────────────────────────────────────────

_FILE_REF_PATTERN = re.compile(r"@([^\s]+)")
_URL_PATTERN = re.compile(r"\bhttps?://[^\s<>()\"']+")

# File extensions treated as readable code/text when attaching a folder
_CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".r", ".sql", ".sh", ".bash", ".zsh", ".ps1",
    ".yaml", ".yml", ".toml", ".json", ".jsonc",
    ".md", ".txt", ".env", ".cfg", ".ini", ".conf",
    ".html", ".css", ".scss", ".sass", ".less",
    ".xml", ".proto", ".graphql",
})

# Directories to skip when attaching a folder
_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".hg", ".svn",
    "node_modules", ".venv", "venv", ".env",
    "dist", "build", "out", "target", ".cache",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
    ".tox", "coverage", ".ruff_cache", "site-packages",
})

# Maximum files to attach from a single folder reference
_FOLDER_FILE_LIMIT = 40

# Words that signal the user wants to DELETE the referenced files, not patch them
_DELETE_WORDS = frozenset({
    "delete", "remove", "rm", "erase", "wipe", "unlink",
    "get rid", "trash", "clean up", "cleanup",
})
_RANGED_FILE_REF_PATTERN = re.compile(r"^(?P<path>.+):(?P<start>\d+)-(?P<end>\d+)$")
_WEB_FETCH_TIMEOUT_SEC = 10
_WEB_CONTENT_CHAR_LIMIT = 8_000

# Matches ```lang\n<body>\n``` (captures body)
_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Detects a file-path comment as the first line of a code block
_FILE_PATH_COMMENT_RE = re.compile(
    r"^(?:#|//|/\*)\s*(?:file:\s*)?(?P<path>[^\s*]+\.\w+)"
)


class _PatchEdit(BaseModel):
    path: str
    operation: Literal["write", "delete"] = "write"
    content: str | None = None


class _PatchEnvelope(BaseModel):
    edits: list[_PatchEdit]


PatchTuple = tuple[Path, str | None, Literal["write", "delete"]]

# Verbs that imply an action → full pipeline or patch path
_ACTION_VERBS = frozenset({
    "fix", "refactor", "implement", "build", "write", "create",
    "update", "change", "add", "remove", "delete", "rename",
    "convert", "generate", "deploy", "patch", "rewrite", "optimize",
    "migrate", "scaffold", "test", "improve", "correct", "repair",
    "edit", "modify", "clean", "format", "lint", "upgrade", "extend",
    "complete", "finish", "solve", "debug",
})
_NON_DELETE_ACTION_VERBS = frozenset(v for v in _ACTION_VERBS if v not in {"delete", "remove", "rm"})

# First words that mark a question → fast agent.ask()
_QUESTION_FIRST_WORDS = frozenset({
    "explain", "what", "why", "how", "describe", "summarize",
    "tell", "show", "is", "are", "does", "can", "could",
    "should", "would", "hi", "hello", "hey",
})

_AGENT_MODES: tuple[str, ...] = tuple(spec.name for spec in MODE_SPECS)
_SPECIALIST_MODES: frozenset[str] = frozenset(
    spec.name for spec in MODE_SPECS if spec.specialist_role is not None
)


# ── patch extraction & approval ────────────────────────────────────────────────

def _is_allowed_derived_write_path(candidate: Path, allowed_files: set[Path], workdir: Path) -> bool:
    """
    Allow safe creation of new files derived from attached files.

    Example: attached `test/card.ts` can produce `test/card.py`.
    """
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
        # same directory + same stem (card.ts -> card.py)
        if rel.parent == af_rel.parent and rel.stem == af_rel.stem:
            return True
        # same stem anywhere in repo path
        if candidate.stem == af.stem:
            return True
        # exact match remains allowed
        if rel_norm == af_rel_norm:
            return True
    return False


def _extract_patches(
    response: str,
    console: Console,
    workdir: Path,
    allowed_files: set[Path],
) -> list[PatchTuple]:
    """
    Scan LLM response for fenced code blocks and return (path, new_content) pairs.
    Nothing is written to disk here — writing happens only after user approval.
    """
    patches: list[PatchTuple] = []
    seen: set[Path] = set()

    for block_match in _CODE_BLOCK_RE.finditer(response):
        fenced_block = response[block_match.start() : block_match.end()]
        if fenced_block.lstrip().lower().startswith("```json"):
            continue
        body = block_match.group(1)
        lines = body.splitlines()
        if not lines:
            continue

        # Try to get file path from first-line comment (e.g. "# ema.py")
        path_match = _FILE_PATH_COMMENT_RE.match(lines[0].strip())
        raw_path: str | None = path_match.group("path") if path_match else None
        code_lines = lines[1:] if raw_path else lines

        if raw_path is None:
            # Fallback 1: lone allowed file → use it unambiguously
            if len(allowed_files) == 1:
                candidate = next(iter(allowed_files))
            # Fallback 2: match by filename appearing in the block header
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
            # Priority: match against the @-referenced files by name/relative path.
            # This ensures "# ema.py" resolves to the exact file the user attached,
            # not a guessed workdir-relative path.
            candidate = None
            for af in allowed_files:
                norm_raw = raw_path.replace("\\", "/")
                norm_af  = str(af).replace("\\", "/")
                if af.name == parsed.name or norm_af.endswith("/" + norm_raw):
                    candidate = af
                    break
            if candidate is None:
                # When files were explicitly attached, do not allow model-invented
                # paths outside that attachment set.
                if allowed_files:
                    console.print(
                        f"[yellow]Skipped {escape(raw_path)} — not part of attached files[/yellow]"
                    )
                    continue
                candidate = parsed if parsed.is_absolute() else (workdir / parsed).resolve()

        # Safety: must stay inside workdir
        try:
            candidate.relative_to(workdir)
        except ValueError:
            console.print(
                f"[yellow]Skipped {escape(str(candidate))} — outside workdir[/yellow]"
            )
            continue

        if not _is_allowed_derived_write_path(candidate, allowed_files, workdir):
            console.print(
                f"[yellow]Skipped {escape(str(candidate))} — not an allowed derived target[/yellow]"
            )
            continue

        if candidate in seen:
            continue
        seen.add(candidate)

        code = "\n".join(code_lines).strip() + "\n"
        if not code.strip():
            continue

        ok, message = validate_by_extension(candidate, code)
        if not ok:
            console.print(
                f"[yellow]Skipped {escape(candidate.name)} — validation failed: "
                f"{escape(message)}[/yellow]"
            )
            continue

        patches.append((candidate, code, "write"))

    return patches


def _extract_json_protocol_patches(
    response: str,
    console: Console,
    workdir: Path,
    allowed_files: set[Path],
) -> list[PatchTuple] | None:
    """
    Extract strict JSON patch protocol payload:
      {"edits":[{"path":"relative/or/absolute","content":"complete file"}]}
    """
    match = _JSON_BLOCK_RE.search(response)
    if not match:
        return []

    raw = match.group(1).strip()
    try:
        payload = _PatchEnvelope.model_validate_json(raw)
    except ValidationError as exc:
        try:
            payload_dict = _parse_relaxed_patch_payload(raw)
            payload = _PatchEnvelope.model_validate(payload_dict)
        except Exception:
            console.print(f"[yellow]Invalid JSON patch payload: {escape(str(exc))}[/yellow]")
            return None
    except Exception as exc:
        try:
            payload_dict = _parse_relaxed_patch_payload(raw)
            payload = _PatchEnvelope.model_validate(payload_dict)
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
            if allowed_files:
                candidate = parsed if parsed.is_absolute() else (workdir / parsed).resolve()
            else:
                candidate = parsed if parsed.is_absolute() else (workdir / parsed).resolve()
        try:
            candidate.relative_to(workdir)
        except ValueError:
            console.print(f"[yellow]Skipped {escape(str(candidate))} — outside workdir[/yellow]")
            continue
        if edit.operation == "write" and not _is_allowed_derived_write_path(candidate, allowed_files, workdir):
            console.print(
                f"[yellow]Skipped {escape(str(candidate))} — not an allowed derived target[/yellow]"
            )
            continue
        if edit.operation == "delete" and allowed_files and candidate not in allowed_files:
            console.print(
                f"[yellow]Skipped {escape(str(candidate))} — delete allowed only for attached files[/yellow]"
            )
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if edit.operation == "delete":
            patches.append((candidate, None, "delete"))
            continue
        if edit.content is None:
            console.print(
                f"[yellow]Skipped {escape(candidate.name)} — missing content for write operation[/yellow]"
            )
            continue
        content = edit.content if edit.content.endswith("\n") else (edit.content + "\n")
        ok, message = validate_by_extension(candidate, content)
        if not ok:
            console.print(
                f"[yellow]Skipped {escape(candidate.name)} — validation failed: "
                f"{escape(message)}[/yellow]"
            )
            continue
        patches.append((candidate, content, "write"))
    return patches


def _parse_relaxed_patch_payload(raw: str) -> dict[str, Any]:
    """
    Parse a relaxed JSON payload for patch protocol.

    Accepts multiline (unescaped) string content values that strict JSON rejects.
    Supported structure:
      {"edits":[{"path":"...","operation":"write|delete","content":"..."}, ...]}
    """
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
                if esc == "n":
                    out.append("\n")
                elif esc == "r":
                    out.append("\r")
                elif esc == "t":
                    out.append("\t")
                elif esc == '"':
                    out.append('"')
                elif esc == "\\":
                    out.append("\\")
                else:
                    out.append(esc)
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
        ch = raw[i]
        if ch == '"':
            return parse_string()
        if ch == "{":
            return parse_object()
        if ch == "[":
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
            value = parse_value()
            obj[key] = value
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


def _show_diff(console: Console, path: Path, old: str, new: str) -> None:
    """Render a colored unified diff between old and new content."""
    diff_lines = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
        lineterm="",
    ))
    if not diff_lines:
        console.print(f"  [dim](no textual changes)[/dim]")
        return
    diff_text = "\n".join(diff_lines)
    console.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))


def _review_and_apply(
    patches: list[PatchTuple],
    console: Console,
    workdir: Path,
) -> list[Path]:
    """
    For each proposed patch, show a diff and ask for approval.
    Options: [y]es  [n]o  [a]ll  [q]uit
    Returns the list of files actually written.
    """
    if not patches:
        return []

    pending_ops: list[PatchTuple] = []
    apply_all = False

    for path, new_content, operation in patches:
        old_content = path.read_text(encoding="utf-8") if path.exists() else ""
        is_new = not path.exists()
        target_content = "" if operation == "delete" else (new_content or "")

        if old_content == target_content:
            console.print(f"[dim]{path.name}: no changes[/dim]")
            continue

        # ── show header + diff ────────────────────────────────────────────────
        console.print()
        if operation == "delete":
            console.print(Panel(f"[bold red]Delete file:[/bold red] {escape(path.name)}", padding=(0, 1)))
        elif is_new:
            console.print(
                Panel(f"[bold cyan]New file:[/bold cyan] {escape(path.name)}", padding=(0, 1))
            )
        else:
            try:
                rel = path.relative_to(workdir)
            except ValueError:
                rel = path
            console.print(
                Panel(
                    f"[bold cyan]Proposed changes:[/bold cyan] {escape(str(rel))}",
                    padding=(0, 1),
                )
            )
        _show_diff(console, path, old_content, target_content)

        if apply_all:
            pending_ops.append((path, new_content, operation))
            console.print(f"[bold green]APPLIED[/bold green] {escape(path.name)}")
            continue

        # ── prompt ────────────────────────────────────────────────────────────
        try:
            choice = console.input(
                "\nApply? [y]es / [n]o / [a]ll / [q]uit: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Cancelled.[/yellow]")
            break

        if choice in {"a", "all"}:
            apply_all = True
            choice = "y"

        if choice in {"y", "yes"}:
            pending_ops.append((path, new_content, operation))
            console.print(f"[bold green]APPLIED[/bold green] {escape(path.name)}")
        elif choice in {"q", "quit"}:
            console.print("[yellow]Stopped — remaining changes discarded.[/yellow]")
            break
        else:
            console.print(f"[yellow]SKIPPED[/yellow] {escape(path.name)}")

    if not pending_ops:
        return []

    # Apply all approved edits atomically with rollback on first failure.
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


# ── routing helpers ────────────────────────────────────────────────────────────

def _should_use_quick_ask(user_input: str) -> bool:
    """True → fast single-call agent.ask(); False → action path (writes files)."""
    text = user_input.strip().lower()
    if not text:
        return True
    # Action verbs win over first-word heuristic.
    # "Can you fix @ema.py" contains "fix" → action path, not Q&A.
    if any(v in text for v in _ACTION_VERBS):
        return False
    first_word = text.split()[0]
    if first_word in _QUESTION_FIRST_WORDS:
        return True
    return True


# ── folder attachment helper ──────────────────────────────────────────────────

def _collect_folder_files(folder: Path) -> list[Path]:
    """Walk a folder and return code files, skipping noise directories."""
    files: list[Path] = []
    for candidate in sorted(folder.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in _CODE_EXTENSIONS:
            continue
        # Skip if any path component is a noise directory
        if any(part in _SKIP_DIRS for part in candidate.parts):
            continue
        files.append(candidate)
        if len(files) >= _FOLDER_FILE_LIMIT:
            break
    return files


def _attach_folder(
    folder: Path,
    ref: str,
    notes: list[str],
    ref_paths: set[Path],
) -> str:
    """Build a multi-file context block for an entire folder reference."""
    files = _collect_folder_files(folder)
    if not files:
        notes.append(f"[yellow]No code files found in @{escape(ref)}[/yellow]")
        return f"@{ref}"

    truncated = len(files) >= _FOLDER_FILE_LIMIT

    blocks: list[str] = []
    attached: list[str] = []
    for f in files:
        try:
            content = read_file(str(f))
        except Exception as exc:
            notes.append(f"[yellow]Skipped {escape(f.name)}: {escape(str(exc))}[/yellow]")
            continue
        try:
            rel = f.relative_to(folder.parent)
        except ValueError:
            rel = f
        blocks.append(
            f"### {rel}\n"
            f"[Attached file: {f}]\n"
            f"```text\n{content}\n```"
        )
        ref_paths.add(f)
        attached.append(f.name)

    suffix = f" (first {_FOLDER_FILE_LIMIT})" if truncated else ""
    notes.append(
        f"[cyan]Attached @{escape(ref)}/ "
        f"({len(attached)} files{suffix})[/cyan]"
    )
    header = (
        f"@{ref}/\n"
        f"[Attached folder: {folder} — {len(attached)} files{suffix}]"
    )
    return header + "\n\n" + "\n\n".join(blocks)


# ── file reference expansion ──────────────────────────────────────────────────

def _expand_file_references(
    user_input: str,
) -> tuple[str, list[str], set[Path]]:
    """
    Replace ``@path`` and ``@path:start-end`` tokens with their file contents.

    Returns:
        expanded   – prompt with file contents inlined
        notes      – rich-markup lines to print before the prompt
        ref_paths  – set of resolved Path objects that were successfully attached
    """
    notes: list[str] = []
    ref_paths: set[Path] = set()

    def _replace_match(match: re.Match[str]) -> str:
        raw_ref = match.group(1)
        ref = raw_ref.rstrip(".,;:!?)]}")
        range_match = _RANGED_FILE_REF_PATTERN.match(ref)
        line_start: int | None = None
        line_end: int | None = None
        if range_match:
            ref = range_match.group("path")
            line_start = int(range_match.group("start"))
            line_end = int(range_match.group("end"))

        candidate = Path(ref)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / ref).resolve()
        if not candidate.exists():
            notes.append(f"[yellow]Could not resolve @{escape(ref)}[/yellow]")
            return match.group(0)

        # ── folder attachment ─────────────────────────────────────────────────
        if candidate.is_dir():
            if line_start is not None:
                notes.append(
                    f"[yellow]Line ranges are not supported on folders (@{escape(raw_ref)})[/yellow]"
                )
            return _attach_folder(candidate, ref, notes, ref_paths)

        if not candidate.is_file():
            notes.append(f"[yellow]Could not resolve @{escape(ref)}[/yellow]")
            return match.group(0)
        try:
            content = read_file(str(candidate))
        except Exception as exc:
            notes.append(
                f"[red]Failed to read @{escape(ref)}: {escape(str(exc))}[/red]"
            )
            return match.group(0)

        line_header = ""
        if line_start is not None and line_end is not None:
            if line_start <= 0 or line_end <= 0 or line_start > line_end:
                notes.append(
                    f"[yellow]Invalid line range in @{escape(raw_ref)}; "
                    f"expected start \u2264 end[/yellow]"
                )
                return match.group(0)
            file_lines = content.splitlines()
            if line_start > len(file_lines):
                notes.append(
                    f"[yellow]Line {line_start} out of bounds in @{escape(raw_ref)}; "
                    f"file has {len(file_lines)} lines[/yellow]"
                )
                return match.group(0)
            clipped_end = min(line_end, len(file_lines))
            content = "\n".join(file_lines[line_start - 1 : clipped_end])
            line_header = f"[Attached line range: {line_start}-{clipped_end}]\n"
            notes.append(f"[cyan]Attached @{escape(raw_ref)}[/cyan]")
        else:
            notes.append(f"[cyan]Attached @{escape(ref)}[/cyan]")

        ref_paths.add(candidate)
        return (
            f"@{raw_ref}\n"
            f"[Attached file: {candidate}]\n"
            f"{line_header}"
            f"```text\n{content}\n```"
        )

    resolved = _FILE_REF_PATTERN.sub(_replace_match, user_input)
    return resolved, notes, ref_paths


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text from HTML while skipping script/style noise."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        merged = " ".join(self._chunks)
        return re.sub(r"\s+", " ", merged).strip()


def _fetch_url_text(url: str) -> str:
    """Fetch URL content and return cleaned text."""
    request = Request(
        url,
        headers={
            "User-Agent": "Euler-Agent/1.0 (+web-context)",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=_WEB_FETCH_TIMEOUT_SEC) as resp:
        content_type = (resp.headers.get("Content-Type", "") or "").lower()
        charset = resp.headers.get_content_charset() or "utf-8"
        raw = resp.read(200_000)
    body = raw.decode(charset, errors="replace")
    if "html" in content_type or "<html" in body.lower():
        parser = _HTMLTextExtractor()
        parser.feed(body)
        parser.close()
        text = parser.text()
    else:
        text = re.sub(r"\s+", " ", body).strip()
    if len(text) > _WEB_CONTENT_CHAR_LIMIT:
        return text[:_WEB_CONTENT_CHAR_LIMIT] + " ... [truncated]"
    return text


def _expand_web_references(user_input: str) -> tuple[str, list[str]]:
    """
    Attach URL context from raw prompt URLs for all modes.

    Returns:
        expanded_prompt  – prompt with web snippets appended
        notes            – rich-markup lines describing attached URLs
    """
    urls = []
    seen: set[str] = set()
    for match in _URL_PATTERN.finditer(user_input):
        raw = match.group(0).rstrip(".,;:!?)]}")
        if raw in seen:
            continue
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        seen.add(raw)
        urls.append(raw)

    if not urls:
        return user_input, []

    notes: list[str] = []
    blocks: list[str] = []
    for url in urls:
        try:
            text = _fetch_url_text(url)
            if not text:
                notes.append(f"[yellow]Fetched URL but no readable content: {escape(url)}[/yellow]")
                continue
            notes.append(f"[cyan]Attached web context: {escape(url)}[/cyan]")
            blocks.append(
                f"### Web Source\n"
                f"[Attached URL: {url}]\n"
                f"```text\n{text}\n```"
            )
        except HTTPError as exc:
            notes.append(
                f"[yellow]Failed to fetch {escape(url)} (HTTP {exc.code})[/yellow]"
            )
        except URLError as exc:
            notes.append(
                f"[yellow]Failed to fetch {escape(url)} ({escape(str(exc.reason))})[/yellow]"
            )
        except Exception as exc:
            notes.append(
                f"[yellow]Failed to fetch {escape(url)} ({escape(str(exc))})[/yellow]"
            )

    if not blocks:
        return user_input, notes
    expanded = user_input + "\n\n" + "\n\n".join(blocks)
    return expanded, notes


# ── safe print ────────────────────────────────────────────────────────────────

def _safe_print(console: Console, text: str) -> None:
    """Print LLM output safely — never let Rich parse brackets as markup."""
    try:
        console.print(escape(str(text)))
    except Exception:
        print(str(text))


# ── error rendering ───────────────────────────────────────────────────────────

def _print_error(console: Console, exc: Exception) -> None:
    """Render an API or runtime error without crashing the REPL."""
    raw_msg = str(exc)
    safe_msg = escape(raw_msg[:600])

    if "ClientError" in raw_msg or "ChatGoogle" in raw_msg or "openai" in raw_msg.lower():
        hint = (
            "[bold yellow]API Error[/bold yellow]\n"
            f"{safe_msg}\n\n"
            "[dim]Tip: Check API key and model name.\n"
            "Gemini keys start with 'AIza'. "
            "Stable models: gemini-2.5-flash  gemini-2.5-pro[/dim]"
        )
    elif "400" in raw_msg or "Bad Request" in raw_msg:
        hint = (
            "[bold red]API 400 Bad Request[/bold red]\n"
            f"{safe_msg}\n\n"
            "[dim]Fix: Euler config set --provider gemini --model gemini-2.5-flash[/dim]"
        )
    elif "401" in raw_msg or "403" in raw_msg or "permission" in raw_msg.lower():
        hint = (
            "[bold red]Auth Error (401/403)[/bold red]\n"
            f"{safe_msg}\n\n"
            "[dim]Re-enter key: Euler config set --provider gemini --model gemini-2.5-flash[/dim]"
        )
    elif "429" in raw_msg or "quota" in raw_msg.lower() or "rate" in raw_msg.lower():
        hint = (
            "[bold yellow]Rate Limit / Quota[/bold yellow]\n"
            f"{safe_msg}\n\n"
            "[dim]Wait a moment then retry, or switch model.[/dim]"
        )
    else:
        hint = f"[bold red]Error:[/bold red] {safe_msg}"

    try:
        console.print(Panel(hint, border_style="red"))
    except Exception:
        print(f"\nError: {raw_msg[:400]}\n")


# ── @ auto-completer ──────────────────────────────────────────────────────────

if _PT_AVAILABLE:
    class _AtCompleter(Completer):
        """Complete slash commands and @path references as the user types.

        - ``/``: show all commands, then filter as user types.
        - ``@``: complete file/folder references with subdirectory drilling.
        """

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            stripped = text.lstrip()

            # Slash-command completion on first token.
            if stripped.startswith("/") and " " not in stripped:
                base_commands = [
                    "/help",
                    "/agent",
                    "/agent modes",
                    "/agent show",
                    "/agent set",
                    "/sql",
                    "/replace",
                    "/convert",
                    "/convert-code",
                    "/auto",
                    "/memory",
                    "/index",
                    "/search",
                    "/graph",
                    "/knowledge-graph",
                    "/pick",
                    "/strict-patch",
                    "/delete",
                    "/exit",
                    "/quit",
                ]
                mode_commands = [f"/agent set {mode}" for mode in _AGENT_MODES]
                all_commands = sorted(set(base_commands + mode_commands))
                token = stripped
                for cmd in all_commands:
                    if cmd.startswith(token):
                        yield Completion(
                            cmd,
                            start_position=-len(token),
                            display=cmd,
                            display_meta="command",
                        )
                return

            # Find the nearest @ that hasn't been closed by a space
            at_pos = text.rfind("@")
            if at_pos == -1:
                return
            after_at = text[at_pos + 1:]
            if " " in after_at:
                return  # @ reference already completed

            after_at = after_at.replace("\\", "/")
            cwd = Path.cwd()

            # Split into directory prefix + filename prefix
            if "/" in after_at:
                dir_part, name_prefix = after_at.rsplit("/", 1)
                base_dir = cwd / dir_part
            else:
                dir_part = ""
                name_prefix = after_at
                base_dir = cwd

            if not base_dir.is_dir():
                return

            try:
                entries = sorted(
                    base_dir.iterdir(),
                    key=lambda p: (p.is_file(), p.name.lower()),
                )
            except PermissionError:
                return

            for entry in entries:
                name = entry.name
                # Skip hidden files and known noisy dirs
                if name.startswith(".") or name in _SKIP_DIRS:
                    continue
                if not name.lower().startswith(name_prefix.lower()):
                    continue

                is_dir = entry.is_dir()
                suffix = "/" if is_dir else ""
                display_meta = "dir" if is_dir else (entry.suffix or "file")

                yield Completion(
                    name + suffix,
                    start_position=-len(name_prefix),
                    display=name + suffix,
                    display_meta=display_meta,
                )

    _PT_STYLE = Style.from_dict({
        "prompt":          "bold ansigreen",
        "completion-menu.completion":          "bg:#1e1e2e fg:#cdd6f4",
        "completion-menu.completion.current":  "bg:#89b4fa fg:#1e1e2e bold",
        "completion-menu.meta.completion":     "fg:#6c7086",
        "completion-menu.meta.completion.current": "fg:#1e1e2e",
    })

    def _make_session() -> "PromptSession":
        history_file = Path.home() / ".euler_agent" / "repl_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        return PromptSession(
            history=FileHistory(str(history_file)),
            completer=_AtCompleter(),
            complete_while_typing=True,
            style=_PT_STYLE,
            mouse_support=False,
        )


# ── REPL entry point ──────────────────────────────────────────────────────────

def run_repl(agent: EulerAgent) -> None:
    console = Console()
    mode_state: dict[str, Any] = {
        "agent_mode": "basic",
        "strict_patch_mode": True,
        "pending_refs": [],
        "runs": 0,
        "cache_hits": 0,
        "last_metrics": {},
        "last_metrics_seq": 0,
    }
    console.print(
        "[bold cyan]Euler REPL[/bold cyan] — type [bold]/help[/bold] for commands, "
        "[bold]@[/bold] to reference files/folders "
        "[dim](agent mode: basic)[/dim]"
    )

    session = _make_session() if _PT_AVAILABLE else None

    while True:
        try:
            console.print(_render_run_metrics_header(mode_state))
            if session is not None:
                user_input = session.prompt("euler> ").strip()
            else:
                user_input = console.input("[bold green]euler> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]bye[/yellow]")
            break

        if not user_input:
            continue

        try:
            pending_refs = mode_state.get("pending_refs", [])
            if pending_refs and not user_input.startswith("/"):
                user_input = " ".join(pending_refs) + " " + user_input
                mode_state["pending_refs"] = []
            _handle_input(console, agent, user_input, mode_state)
            _record_agent_metrics(agent, mode_state)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]bye[/yellow]")
            break
        except Exception as exc:
            _print_error(console, exc)


# ── dispatcher ────────────────────────────────────────────────────────────────

def _handle_input(
    console: Console,
    agent: EulerAgent,
    user_input: str,
    mode_state: dict[str, str],
) -> None:

    if user_input in {"/exit", "/quit"}:
        console.print("[yellow]bye[/yellow]")
        raise KeyboardInterrupt

    if user_input == "/help":
        console.print(
            "[bold]Euler REPL commands[/bold]\n"
            "  /agent                         — show current agent mode\n"
            "  /agent set <mode>              — set mode until you change it\n"
            "  /agent modes                    — list available modes\n"
            "  /agent show <mode>             — detailed mode responsibility/examples\n"
            "  /sql <requirement>               — production SQL generation\n"
            "  /replace <file> <s> <e> <instr>  — rewrite a line range\n"
            "  /convert <file> <lang>           — convert file to another language\n"
            "  /convert-code <src>\u2192<tgt>        — paste & convert inline code\n"
            "  /auto <goal> [| <verify_cmd>]    — run autopilot loop\n"
            "  /memory <query>                  — search past session memory\n"
            "  /index [full]                    — build/update semantic index\n"
            "  /search <query>                  — semantic code search\n"
            "  /graph                           — build cross-language code graph\n"
            "  /knowledge-graph [folder]        — save graph to ./Euler/knowledge_graph*.json\n"
            "  /pick <query>                    — fuzzy-select files and queue @refs\n"
            "  /strict-patch on|off             — enforce unified diff in patch mode\n"
            "  @file.ext                        — attach full file to your prompt\n"
            "  @file.ext:start-end              — attach a line range to your prompt\n"
            "  @folder/                         — attach all code files in a folder\n"
            "  https://example.com/page         — auto-attach webpage context\n"
            "  /delete @file1 @file2 ...        — delete files (with confirmation)\n"
            "  /exit | /quit                    — exit REPL\n\n"
            "[dim]Agent mode examples:\n"
            "  /agent set basic      (default routing)\n"
            "  /agent set swarm      (always multi-agent pipeline)\n"
            "  /agent set assistant  (always single-call assistant)\n"
            "  /agent set coder      (always specialist role)\n\n"
            "[dim]Examples:\n"
            "  fix @ema.py\n"
            "  explain @ema.py:17-22\n"
            "  refactor @euler_agent/repl.py\n"
            "  review all code in @euler_agent/\n"
            "  fix bugs in @euler_agent/ and @ema.py\n"
            "  delete @ema.py @calculator.py\n"
            "  /delete @ema.py @calculator.py[/dim]"
        )
        return

    if user_input == "/agent":
        current = mode_state.get("agent_mode", "basic")
        console.print(f"[cyan]Current agent mode:[/cyan] [bold]{escape(current)}[/bold]")
        return

    if user_input.startswith("/strict-patch "):
        val = user_input.removeprefix("/strict-patch ").strip().lower()
        if val not in {"on", "off"}:
            console.print("[yellow]usage: /strict-patch on|off[/yellow]")
            return
        mode_state["strict_patch_mode"] = (val == "on")
        console.print(
            "[green]Strict patch mode:[/green] "
            + ("[bold]ON[/bold]" if mode_state["strict_patch_mode"] else "[bold]OFF[/bold]")
        )
        return

    if user_input.startswith("/pick "):
        query = user_input.removeprefix("/pick ").strip()
        _fuzzy_file_pick(console, Path.cwd(), query, mode_state)
        return

    if user_input in {"/agent modes", "/agent list"}:
        _print_modes_overview(console)
        return

    if user_input.startswith("/agent show "):
        mode_name = user_input.removeprefix("/agent show ").strip().lower()
        _print_mode_details(console, mode_name)
        return

    if user_input.startswith("/agent set "):
        requested = user_input.removeprefix("/agent set ").strip().lower()
        if requested not in _AGENT_MODES:
            console.print(
                "[red]Invalid mode.[/red] Use [bold]/agent modes[/bold] "
                "to list supported modes."
            )
            return
        mode_state["agent_mode"] = requested
        console.print(
            f"[green]Agent mode set to[/green] [bold]{escape(requested)}[/bold] "
            "[dim](stays active until changed)[/dim]"
        )
        return

    # ── /sql ──────────────────────────────────────────────────────────────────
    if user_input.startswith("/sql "):
        requirement = user_input.removeprefix("/sql ").strip()
        if not requirement:
            return
        try:
            with console.status("[bold cyan]Generating SQL...[/bold cyan]", spinner="dots"):
                result = agent.generate_sql(requirement)
            _safe_print(console, result)
        except Exception as exc:
            _print_error(console, exc)
        return

    # ── /replace ──────────────────────────────────────────────────────────────
    if user_input.startswith("/replace "):
        parts = user_input.split(" ", 4)
        if len(parts) < 5:
            console.print("[red]usage: /replace <file> <start> <end> <instruction>[/red]")
            return
        _, file_path, start, end, instruction = parts
        target = Path(file_path)
        if not target.exists():
            console.print(f"[red]file not found: {escape(file_path)}[/red]")
            return
        selected_lines = read_file(str(target)).splitlines()
        try:
            start_line, end_line = int(start), int(end)
        except ValueError:
            console.print("[red]start and end must be integers[/red]")
            return
        try:
            selected_text = "\n".join(selected_lines[start_line - 1 : end_line])
            with console.status("[bold cyan]Rewriting selection...[/bold cyan]", spinner="dots"):
                replacement = agent.rewrite_selection(str(target), selected_text, instruction)
            message = replace_range(str(target), start_line, end_line, replacement)
            console.print(f"[bold green]DONE:[/bold green] {escape(message)}")
        except Exception as exc:
            _print_error(console, exc)
        return

    # ── /delete ───────────────────────────────────────────────────────────────
    if user_input.startswith("/delete "):
        raw_paths = user_input.removeprefix("/delete ").split()
        _handle_delete_command(console, raw_paths)
        return

    # ── /auto ─────────────────────────────────────────────────────────────────
    if user_input.startswith("/auto "):
        payload = user_input.removeprefix("/auto ").strip()
        goal, sep, verify = payload.partition(" | ")
        try:
            with console.status("[bold cyan]Autopilot running...[/bold cyan]", spinner="dots"):
                output = run_autopilot(
                    agent=agent,
                    goal=goal,
                    workdir=str(Path.cwd()),
                    max_rounds=4,
                    verify_command=verify if sep else None,
                )
            _safe_print(console, output)
        except Exception as exc:
            _print_error(console, exc)
        return

    # ── /memory ───────────────────────────────────────────────────────────────
    if user_input.startswith("/memory "):
        query = user_input.removeprefix("/memory ").strip()
        rows = search_memory(project=str(Path.cwd()), query=query, limit=3)
        if not rows:
            console.print("[yellow]No matching memory found.[/yellow]")
        else:
            for row in rows:
                console.print(
                    f"[bold]{escape(str(row.timestamp))}[/bold] {escape(row.goal)}"
                )
                _safe_print(console, row.result[:500])
                console.print("\u2500" * 40)
        return

    # ── /index ────────────────────────────────────────────────────────────────
    if user_input.startswith("/index"):
        full = user_input.strip().lower() == "/index full"
        with console.status("[bold cyan]Indexing...[/bold cyan]", spinner="dots"):
            result = index_path(str(Path.cwd()), incremental=not full)
        console.print(f"[green]{escape(result)}[/green]")
        return

    # ── /search ───────────────────────────────────────────────────────────────
    if user_input.startswith("/search "):
        query = user_input.removeprefix("/search ").strip()
        hits = search_index(str(Path.cwd()), query, limit=5)
        if not hits:
            console.print("[yellow]No semantic hits found. Run /index first.[/yellow]")
        else:
            for hit in hits:
                console.print(
                    f"[bold]{escape(hit['path'])}[/bold] "
                    f"[dim]lines {hit['start_line']}-{hit['end_line']}[/dim]"
                )
                _safe_print(console, hit["content"][:450])
                console.print("\u2500" * 40)
        return

    # ── /graph ────────────────────────────────────────────────────────────────
    if user_input == "/graph":
        with console.status("[bold cyan]Building graph...[/bold cyan]", spinner="dots"):
            result = build_code_graph(str(Path.cwd()))
        _safe_print(console, result)
        return

    if user_input.startswith("/knowledge-graph"):
        raw_target = user_input.removeprefix("/knowledge-graph").strip()
        if raw_target:
            raw_target = raw_target.strip().strip("\"'")
            raw_target = raw_target.lstrip("@").rstrip(".,;:!?)]}")
        cwd = Path.cwd().resolve()
        target_root = (cwd / raw_target).resolve() if raw_target else cwd
        if not target_root.exists() or not target_root.is_dir():
            console.print(
                f"[red]Invalid folder:[/red] {escape(str(target_root))}. "
                "Provide a valid directory path."
            )
            return
        euler_dir = cwd / "Euler"
        euler_dir.mkdir(parents=True, exist_ok=True)
        file_name = _safe_graph_filename_for_target(cwd, target_root)
        output_file = euler_dir / file_name
        with console.status("[bold cyan]Building knowledge graph...[/bold cyan]", spinner="dots"):
            result = build_code_graph(str(target_root), output_path=str(output_file))
        _safe_print(
            console,
            (
                f"Knowledge graph target: {target_root}\n"
                f"Saved at: {output_file}\n"
                f"{result}"
            ),
        )
        return

    # ── /convert <file> <lang> ────────────────────────────────────────────────
    if user_input.startswith("/convert "):
        parts = user_input.split(" ", 2)
        if len(parts) < 3:
            console.print("[red]usage: /convert <file> <target_lang>[/red]")
            return
        _, file_path, target_lang = parts
        console.print(
            f"[cyan]Converting {escape(file_path)} \u2192 {escape(target_lang)}...[/cyan]"
        )
        try:
            with console.status("[bold cyan]Converting...[/bold cyan]", spinner="dots"):
                result = agent.convert_file(file_path.strip(), target_lang.strip())
            _safe_print(console, result)
        except Exception as exc:
            _print_error(console, exc)
        return

    # ── /convert-code <src>→<tgt> ─────────────────────────────────────────────
    if user_input.startswith("/convert-code "):
        spec = user_input.removeprefix("/convert-code ").strip()
        arrow = "\u2192" if "\u2192" in spec else "->" if "->" in spec else None
        if not arrow:
            console.print("[red]usage: /convert-code <src_lang>\u2192<tgt_lang>[/red]")
            return
        src_lang, tgt_lang = [p.strip() for p in spec.split(arrow, 1)]
        console.print("[cyan]Paste source code, then a line with just '---' to convert:[/cyan]")
        code_lines: list[str] = []
        while True:
            line = console.input("")
            if line.strip() == "---":
                break
            code_lines.append(line)
        source_code = "\n".join(code_lines)
        if not source_code.strip():
            console.print("[yellow]No code provided.[/yellow]")
            return
        console.print(
            f"[cyan]Converting {escape(src_lang)} \u2192 {escape(tgt_lang)}...[/cyan]"
        )
        try:
            with console.status("[bold cyan]Converting...[/bold cyan]", spinner="dots"):
                result = agent.convert_language(source_code, src_lang, tgt_lang)
            _safe_print(console, result)
        except Exception as exc:
            _print_error(console, exc)
        return

    # ── free-form prompt ──────────────────────────────────────────────────────
    _handle_freeform(console, agent, user_input, mode_state.get("agent_mode", "basic"), mode_state)


def _handle_delete_command(console: Console, raw_paths: list[str]) -> None:
    """Resolve, confirm, and delete files listed explicitly via /delete."""
    resolved: list[Path] = []
    for rp in raw_paths:
        ref = rp.lstrip("@").rstrip(".,;:!?)]}")
        candidate = Path(ref)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / ref).resolve()
        if not candidate.exists():
            console.print(f"[yellow]Not found: {escape(str(candidate))}[/yellow]")
        elif not candidate.is_file():
            console.print(f"[yellow]Not a file: {escape(str(candidate))}[/yellow]")
        else:
            resolved.append(candidate)

    if not resolved:
        return

    console.print("[bold]Files to delete:[/bold]")
    for p in resolved:
        console.print(f"  [red]{escape(str(p))}[/red]")
    try:
        confirm = console.input("Confirm delete? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    if confirm not in {"y", "yes"}:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    for p in resolved:
        try:
            p.unlink()
            console.print(f"[bold green]DELETED:[/bold green] {escape(str(p.name))}")
        except Exception as exc:
            console.print(f"[red]Failed to delete {escape(str(p))}: {escape(str(exc))}[/red]")


def _detect_delete_intent(user_input: str, ref_paths: set[Path]) -> bool:
    """
    Return True only for explicit delete-only intent.

    Mixed instructions (e.g. "convert X and delete Y") should not short-circuit
    into immediate deletion before creation/update actions.
    """
    if not ref_paths:
        return False
    text = user_input.strip().lower()
    if not any(word in text for word in _DELETE_WORDS):
        return False
    if any(v in text for v in _NON_DELETE_ACTION_VERBS):
        return False
    starts_with_delete = (
        text.startswith("delete ")
        or text.startswith("remove ")
        or text.startswith("rm ")
    )
    return starts_with_delete or text in {"delete", "remove", "rm"}


def _mode_prefixed_prompt(agent_mode: str, prompt: str) -> str:
    spec = MODE_BY_NAME.get(agent_mode)
    if spec is None or not spec.prompt_preamble.strip():
        return prompt
    return f"{spec.prompt_preamble}\n\n{prompt}"


def _print_mode_details(console: Console, mode_name: str) -> None:
    spec = MODE_BY_NAME.get(mode_name)
    if spec is None:
        console.print(
            f"[red]Unknown mode:[/red] {escape(mode_name)}. "
            "Use [bold]/agent modes[/bold]."
        )
        return

    examples = "\n".join(f"  - {escape(example)}" for example in spec.examples) or "  - (none)"
    role = spec.specialist_role or "N/A"
    console.print(
        Panel(
            f"[bold]{escape(spec.name)}[/bold]\n"
            f"[cyan]Summary:[/cyan] {escape(spec.summary)}\n"
            f"[cyan]Responsibility:[/cyan] {escape(spec.responsibility)}\n"
            f"[cyan]Execution strategy:[/cyan] {escape(spec.strategy)}\n"
            f"[cyan]Specialist role:[/cyan] {escape(role)}\n"
            f"[cyan]Prompt preamble:[/cyan]\n{escape(spec.prompt_preamble)}\n\n"
            f"[cyan]Examples:[/cyan]\n{examples}",
            title="Agent Mode Details",
            border_style="cyan",
        )
    )


def _print_modes_overview(console: Console) -> None:
    console.print("[bold]Available agent modes[/bold]")
    for spec in MODE_SPECS:
        console.print(
            f"  - [bold]{escape(spec.name)}[/bold]: {escape(spec.summary)} "
            f"[dim](strategy: {escape(spec.strategy)})[/dim]"
        )
    console.print(
        "\n[dim]Tip: /agent show <mode> to see responsibilities, prompt, and examples.[/dim]"
    )


def _render_run_metrics_header(ui_state: dict[str, Any]) -> str:
    runs = int(ui_state.get("runs", 0))
    cache_hits = int(ui_state.get("cache_hits", 0))
    last = ui_state.get("last_metrics", {}) or {}
    cache_hit_pct = (cache_hits * 100.0 / runs) if runs else 0.0
    stage_tokens = last.get("stage_tokens", {}) or {}
    total_tokens = sum(int(v) for v in stage_tokens.values())
    specialists_used = int(last.get("specialists_used", 0))
    specialists_total = int(last.get("specialists_total", 8))
    util_pct = (specialists_used * 100.0 / specialists_total) if specialists_total else 0.0
    if stage_tokens:
        top = sorted(stage_tokens.items(), key=lambda kv: kv[1], reverse=True)[:3]
        stage_summary = ", ".join(f"{k}:{int(v)}t" for k, v in top)
    else:
        stage_summary = "n/a"
    return (
        f"[dim]RunMetrics | tokens:{total_tokens} | cache-hit:{cache_hit_pct:.1f}% "
        f"| specialists:{specialists_used}/{specialists_total} ({util_pct:.0f}%) "
        f"| stage-cost:{stage_summary}[/dim]"
    )


def _record_agent_metrics(agent: EulerAgent, ui_state: dict[str, Any]) -> None:
    metrics = agent.get_last_run_stats()
    if not metrics:
        return
    seq = int(metrics.get("seq", 0))
    if seq <= int(ui_state.get("last_metrics_seq", 0)):
        return
    ui_state["runs"] = int(ui_state.get("runs", 0)) + 1
    if metrics.get("cache_hit"):
        ui_state["cache_hits"] = int(ui_state.get("cache_hits", 0)) + 1
    ui_state["last_metrics"] = metrics
    ui_state["last_metrics_seq"] = seq


def _safe_graph_filename_for_target(cwd: Path, target_root: Path) -> str:
    if target_root.resolve() == cwd.resolve():
        return "knowledge_graph.json"
    safe = "_".join(part for part in target_root.parts if part not in {"/", "\\", ":"})
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe).strip("_")
    if not safe:
        safe = "selected_folder"
    return f"knowledge_graph_{safe}.json"


def _handle_freeform(
    console: Console,
    agent: EulerAgent,
    user_input: str,
    agent_mode: str = "basic",
    ui_state: dict[str, Any] | None = None,
) -> None:
    """Route free-form prompts, inject file content, call agent, apply patches."""
    expanded_input, notes, ref_paths = _expand_file_references(user_input)
    expanded_input, web_notes = _expand_web_references(expanded_input)
    notes.extend(web_notes)
    for note in notes:
        console.print(note)

    # ── delete intent: user asked to remove/delete/rm the @files ─────────────
    if _detect_delete_intent(user_input, ref_paths):
        _handle_delete_command(console, [str(p) for p in ref_paths])
        return

    mode = (agent_mode or "basic").strip().lower()
    if mode not in MODE_BY_NAME:
        mode = "basic"
    is_quick = _should_use_quick_ask(user_input)
    has_file_refs = bool(ref_paths)
    workdir = Path.cwd()
    mode_prompt = _mode_prefixed_prompt(mode, expanded_input)
    strict_patch_mode = bool((ui_state or {}).get("strict_patch_mode", False))

    # ── forced mode: assistant single-call ────────────────────────────────────
    if mode == "assistant":
        try:
            with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                output = agent.ask(mode_prompt, role="assistant")
        except Exception as exc:
            _print_error(console, exc)
            return
        _print_output(console, output)
        return

    # ── forced mode: specialist role single-call ──────────────────────────────
    if mode in _SPECIALIST_MODES:
        specialist_prompt = mode_prompt
        if has_file_refs:
            specialist_prompt += _build_patch_hint(ref_paths)
        specialist_role = MODE_BY_NAME[mode].specialist_role or mode
        try:
            with console.status(
                f"[bold cyan]Running specialist ({escape(mode)})...[/bold cyan]",
                spinner="dots",
            ):
                output = agent.ask(specialist_prompt, role=specialist_role)
        except Exception as exc:
            _print_error(console, exc)
            return
        _print_output(console, output)
        _apply_and_report(console, output, workdir, ref_paths if has_file_refs else set())
        return

    # ── forced mode: patch-oriented single-call ────────────────────────────────
    if mode == "patch":
        prompt = mode_prompt + (_build_patch_hint(ref_paths) if has_file_refs else "")
        if strict_patch_mode:
            prompt += (
                "\n\nSTRICT PATCH MODE:\n"
                "You MUST output exactly one ```json fenced patch payload using the required edits schema.\n"
                "Reject free-form prose."
            )
        try:
            with console.status(
                "[bold cyan]Thinking and writing patches...[/bold cyan]",
                spinner="dots",
            ):
                output = agent.ask(prompt, role="senior engineer fixing code")
        except Exception as exc:
            _print_error(console, exc)
            return
        if strict_patch_mode:
            has_json_patch = bool(_JSON_BLOCK_RE.search(output))
            if not has_json_patch:
                console.print(
                    "[red]Strict patch mode rejected response:[/red] "
                    "missing JSON patch payload block."
                )
                return
        _print_output(console, output)
        _apply_and_report(console, output, workdir, ref_paths if has_file_refs else set())
        return

    # ── forced mode: full multi-agent run ─────────────────────────────────────
    if mode == "swarm":
        try:
            with console.status(
                "[bold cyan]Running agents (may take ~30s)...[/bold cyan]", spinner="dots"
            ):
                output = agent.run(mode_prompt)
        except Exception as exc:
            _print_error(console, exc)
            return
        _print_output(console, output)
        _apply_and_report(console, output, workdir, set())
        return

    # ── default mode: existing heuristic behavior ("basic") ───────────────────
    # ── path A: Q&A — fast single call, no file writes ────────────────────────
    if is_quick:
        try:
            with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                output = agent.ask(mode_prompt, role="assistant")
        except Exception as exc:
            _print_error(console, exc)
            return
        _print_output(console, output)
        return

    # ── path B: action + @file refs — fast call with explicit patch format ────
    if has_file_refs:
        patch_hint = _build_patch_hint(ref_paths)
        prompted = mode_prompt + patch_hint
        try:
            with console.status(
                "[bold cyan]Thinking and writing patches...[/bold cyan]", spinner="dots"
            ):
                output = agent.ask(prompted, role="senior engineer fixing code")
        except Exception as exc:
            _print_error(console, exc)
            return
        _print_output(console, output)
        _apply_and_report(console, output, workdir, ref_paths)
        return

    # ── path C: action without file refs — full multi-agent pipeline ──────────
    try:
        with console.status(
            "[bold cyan]Running agents (may take ~30s)...[/bold cyan]", spinner="dots"
        ):
            output = agent.run(mode_prompt)
    except Exception as exc:
        _print_error(console, exc)
        return
    _print_output(console, output)
    _apply_and_report(console, output, workdir, set())


def _build_patch_hint(ref_paths: set[Path]) -> str:
    """Return strict patch protocol instructions."""
    names = ", ".join(p.name for p in ref_paths)
    return (
        f"\n\n---\nIMPORTANT — OUTPUT FORMAT:\n"
        f"For every file you modify ({names}), output exactly one JSON fenced block "
        f"with this schema:\n"
        f"```json\n"
        f'{{"edits":[{{"path":"relative/path.ext","operation":"write","content":"<complete updated file>"}},'
        f'{{"path":"relative/path.ext","operation":"delete"}}]}}\n'
        f"```\n"
        f"Rules:\n"
        f"- operation must be write or delete.\n"
        f"- write requires content with COMPLETE updated file, not partial snippets.\n"
        f"- delete must omit content or set content to null.\n"
        f"- path must point only to attached files when files are attached.\n"
        f"- for mixed create+delete requests, include both edits in one payload.\n"
        f"- no prose outside the JSON block."
    )


def _iter_candidate_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _SKIP_DIRS or part.startswith(".") for part in rel_parts):
            continue
        if path.suffix.lower() in _CODE_EXTENSIONS:
            out.append(path)
    return out


def _fuzzy_file_pick(console: Console, root: Path, query: str, ui_state: dict[str, Any]) -> None:
    q = query.strip().lower()
    if not q:
        console.print("[yellow]usage: /pick <file-or-fragment>[/yellow]")
        return
    files = _iter_candidate_files(root)
    scored: list[tuple[int, Path]] = []
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        name = path.name.lower()
        rel_lower = rel.lower()
        score = 0
        if rel_lower.startswith(q):
            score += 50
        if q in name:
            score += 30
        if q in rel_lower:
            score += 10
        if score > 0:
            scored.append((score, path))
    if not scored:
        console.print("[yellow]No matching files found.[/yellow]")
        return
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    top = scored[:20]
    console.print("[bold]Pick files[/bold] (comma list, e.g. 1,3,5; or 'a' for all):")
    for idx, (_, path) in enumerate(top, 1):
        rel = str(path.relative_to(root)).replace("\\", "/")
        console.print(f"  {idx:>2}. {escape(rel)}")
    choice = console.input("Select: ").strip().lower()
    chosen: list[Path] = []
    if choice == "a":
        chosen = [p for _, p in top]
    else:
        indexes: list[int] = []
        for part in choice.split(","):
            part = part.strip()
            if not part:
                continue
            if part.isdigit():
                indexes.append(int(part))
        for i in indexes:
            if 1 <= i <= len(top):
                chosen.append(top[i - 1][1])
    if not chosen:
        console.print("[yellow]No files selected.[/yellow]")
        return
    refs = [f"@{str(p.relative_to(root)).replace('\\', '/')}" for p in chosen]
    ui_state["pending_refs"] = refs
    console.print(
        "[green]Selected refs queued for next prompt:[/green] "
        + " ".join(escape(r) for r in refs)
    )


def _print_output(console: Console, output: str) -> None:
    if not output or not str(output).strip():
        console.print(
            Panel(
                "[bold yellow]No response generated.[/bold yellow]\n\n"
                "Common causes:\n"
                "  [red]\u2022[/red] Invalid or expired API key\n"
                "  [red]\u2022[/red] Wrong model slug\n"
                "  [red]\u2022[/red] Network / proxy issue\n\n"
                "Fix:\n"
                "  Euler config show\n"
                "  Euler config set --provider gemini --model gemini-2.5-flash",
                border_style="yellow",
            )
        )
        return
    _safe_print(console, output)


def _apply_and_report(
    console: Console,
    output: str,
    workdir: Path,
    ref_paths: set[Path],
) -> None:
    """
    Extract patches from the LLM response, show a diff for each, ask for
    approval, and only then write the approved files.
    """
    patches = _extract_json_protocol_patches(output, console, workdir, ref_paths)
    if patches is None:
        return
    if not patches:
        patches = _extract_patches(output, console, workdir, ref_paths)
    if not patches:
        if ref_paths:
            console.print(
                "[dim]No file patch detected in response. "
                "Use /replace for targeted line edits.[/dim]"
            )
        return
    _review_and_apply(patches, console, workdir)
