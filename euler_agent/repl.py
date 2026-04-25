"""Interactive REPL shell for Euler agent."""

from __future__ import annotations

import difflib
import re
from ast import parse as ast_parse
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

from euler_agent.core.agent import EulerAgent
from euler_agent.core.autopilot import run_autopilot
from euler_agent.analysis.code_graph import build_code_graph
from euler_agent.memory.store import search_memory
from euler_agent.analysis.semantic_index import index_path, search_index
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

# Matches ```lang\n<body>\n``` (captures body)
_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

# Detects a file-path comment as the first line of a code block
_FILE_PATH_COMMENT_RE = re.compile(
    r"^(?:#|//|/\*)\s*(?:file:\s*)?(?P<path>[^\s*]+\.\w+)"
)

# Verbs that imply an action → full pipeline or patch path
_ACTION_VERBS = frozenset({
    "fix", "refactor", "implement", "build", "write", "create",
    "update", "change", "add", "remove", "delete", "rename",
    "convert", "generate", "deploy", "patch", "rewrite", "optimize",
    "migrate", "scaffold", "test", "improve", "correct", "repair",
    "edit", "modify", "clean", "format", "lint", "upgrade", "extend",
    "complete", "finish", "solve", "debug",
})

# First words that mark a question → fast agent.ask()
_QUESTION_FIRST_WORDS = frozenset({
    "explain", "what", "why", "how", "describe", "summarize",
    "tell", "show", "is", "are", "does", "can", "could",
    "should", "would", "hi", "hello", "hey",
})


# ── patch extraction & approval ────────────────────────────────────────────────

def _extract_patches(
    response: str,
    console: Console,
    workdir: Path,
    allowed_files: set[Path],
) -> list[tuple[Path, str]]:
    """
    Scan LLM response for fenced code blocks and return (path, new_content) pairs.
    Nothing is written to disk here — writing happens only after user approval.
    """
    patches: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for block_match in _CODE_BLOCK_RE.finditer(response):
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
                candidate = parsed if parsed.is_absolute() else (workdir / parsed).resolve()

        # Safety: must stay inside workdir
        try:
            candidate.relative_to(workdir)
        except ValueError:
            console.print(
                f"[yellow]Skipped {escape(str(candidate))} — outside workdir[/yellow]"
            )
            continue

        if candidate in seen:
            continue
        seen.add(candidate)

        code = "\n".join(code_lines).strip() + "\n"
        if not code.strip():
            continue

        # Python AST validation (reject before even showing diff)
        if candidate.suffix == ".py":
            try:
                ast_parse(code)
            except SyntaxError as exc:
                console.print(
                    f"[yellow]Skipped {escape(candidate.name)} — "
                    f"syntax error: {escape(str(exc))}[/yellow]"
                )
                continue

        patches.append((candidate, code))

    return patches


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
    patches: list[tuple[Path, str]],
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

    written: list[Path] = []
    apply_all = False

    for path, new_content in patches:
        old_content = path.read_text(encoding="utf-8") if path.exists() else ""
        is_new = not path.exists()

        if old_content == new_content:
            console.print(f"[dim]{path.name}: no changes[/dim]")
            continue

        # ── show header + diff ────────────────────────────────────────────────
        console.print()
        if is_new:
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
        _show_diff(console, path, old_content, new_content)

        if apply_all:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_content, encoding="utf-8")
            written.append(path)
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
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_content, encoding="utf-8")
            written.append(path)
            console.print(f"[bold green]APPLIED[/bold green] {escape(path.name)}")
        elif choice in {"q", "quit"}:
            console.print("[yellow]Stopped — remaining changes discarded.[/yellow]")
            break
        else:
            console.print(f"[yellow]SKIPPED[/yellow] {escape(path.name)}")

    return written


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
        """Complete @path references as the user types them.

        Triggers whenever the cursor is inside a ``@…`` token.
        Shows matching files and folders from cwd, with subdirectory
        drilling (e.g. ``@euler_agent/c`` shows files starting with "c"
        inside the euler_agent/ folder).
        """

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor

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
    console.print(
        "[bold cyan]Euler REPL[/bold cyan] — type [bold]/help[/bold] for commands, "
        "[bold]@[/bold] to reference files/folders"
    )

    session = _make_session() if _PT_AVAILABLE else None

    while True:
        try:
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
            _handle_input(console, agent, user_input)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]bye[/yellow]")
            break
        except Exception as exc:
            _print_error(console, exc)


# ── dispatcher ────────────────────────────────────────────────────────────────

def _handle_input(console: Console, agent: EulerAgent, user_input: str) -> None:

    if user_input in {"/exit", "/quit"}:
        console.print("[yellow]bye[/yellow]")
        raise KeyboardInterrupt

    if user_input == "/help":
        console.print(
            "[bold]Euler REPL commands[/bold]\n"
            "  /sql <requirement>               — production SQL generation\n"
            "  /replace <file> <s> <e> <instr>  — rewrite a line range\n"
            "  /convert <file> <lang>           — convert file to another language\n"
            "  /convert-code <src>\u2192<tgt>        — paste & convert inline code\n"
            "  /auto <goal> [| <verify_cmd>]    — run autopilot loop\n"
            "  /memory <query>                  — search past session memory\n"
            "  /index [full]                    — build/update semantic index\n"
            "  /search <query>                  — semantic code search\n"
            "  /graph                           — build cross-language code graph\n"
            "  @file.ext                        — attach full file to your prompt\n"
            "  @file.ext:start-end              — attach a line range to your prompt\n"
            "  @folder/                         — attach all code files in a folder\n"
            "  /delete @file1 @file2 ...        — delete files (with confirmation)\n"
            "  /exit | /quit                    — exit REPL\n\n"
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
    _handle_freeform(console, agent, user_input)


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
    """Return True if the user wants to DELETE the referenced files."""
    if not ref_paths:
        return False
    text = user_input.strip().lower()
    return any(word in text for word in _DELETE_WORDS)


def _handle_freeform(console: Console, agent: EulerAgent, user_input: str) -> None:
    """Route free-form prompts, inject file content, call agent, apply patches."""
    expanded_input, notes, ref_paths = _expand_file_references(user_input)
    for note in notes:
        console.print(note)

    # ── delete intent: user asked to remove/delete/rm the @files ─────────────
    if _detect_delete_intent(user_input, ref_paths):
        _handle_delete_command(console, [str(p) for p in ref_paths])
        return

    is_quick = _should_use_quick_ask(user_input)
    has_file_refs = bool(ref_paths)
    workdir = Path.cwd()

    # ── path A: Q&A — fast single call, no file writes ────────────────────────
    if is_quick:
        try:
            with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                output = agent.ask(expanded_input, role="assistant")
        except Exception as exc:
            _print_error(console, exc)
            return
        _print_output(console, output)
        return

    # ── path B: action + @file refs — fast call with explicit patch format ────
    if has_file_refs:
        patch_hint = _build_patch_hint(ref_paths)
        prompted = expanded_input + patch_hint
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
            output = agent.run(expanded_input)
    except Exception as exc:
        _print_error(console, exc)
        return
    _print_output(console, output)
    _apply_and_report(console, output, workdir, set())


def _build_patch_hint(ref_paths: set[Path]) -> str:
    """Return an appended instruction that tells the LLM to emit patchable code blocks."""
    names = ", ".join(p.name for p in ref_paths)
    return (
        f"\n\n---\nIMPORTANT — OUTPUT FORMAT:\n"
        f"For every file you modify ({names}), output the COMPLETE updated file content "
        f"inside a fenced code block where the VERY FIRST LINE of the block is a comment "
        f"with the filename, like this:\n"
        f"```python\n"
        f"# ema.py\n"
        f"<complete new file content here>\n"
        f"```\n"
        f"Do not truncate. Output the entire file, not just the changed section."
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
    patches = _extract_patches(output, console, workdir, ref_paths)
    if not patches:
        if ref_paths:
            console.print(
                "[dim]No file patch detected in response. "
                "Use /replace for targeted line edits.[/dim]"
            )
        return
    _review_and_apply(patches, console, workdir)
