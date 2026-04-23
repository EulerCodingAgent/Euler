"""Interactive REPL shell for Euler agent."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from euler_agent.agent import EulerAgent
from euler_agent.autopilot import run_autopilot
from euler_agent.code_graph import build_code_graph
from euler_agent.memory import search_memory
from euler_agent.semantic_index import index_path, search_index
from euler_agent.tools import read_file, replace_range


def _print_error(console: Console, exc: Exception) -> None:
    """Render an API or runtime error without crashing the REPL."""
    msg = str(exc)
    # Extract the most useful part of long tracebacks from langchain wrappers.
    if "ClientError" in msg or "ChatGoogle" in msg or "openai" in msg.lower():
        hint = (
            "[bold yellow]API Error[/bold yellow]\n"
            f"{msg[:600]}\n\n"
            "[dim]Tip: Check API key first, then model name.\n"
            "Gemini API keys from AI Studio usually start with 'AIza'.\n"
            "Common stable Gemini models: gemini-2.5-flash  gemini-2.5-pro  gemini-2.5-flash-lite  gemini-2.0-flash[/dim]"
        )
    else:
        hint = f"[bold red]Error:[/bold red] {msg[:600]}"
    console.print(Panel(hint, border_style="red"))


def run_repl(agent: EulerAgent) -> None:
    console = Console()
    console.print("[bold cyan]Euler REPL[/bold cyan] — type [bold]/help[/bold] for commands")

    while True:
        try:
            user_input = console.input("[bold green]euler> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]bye[/yellow]")
            break

        if not user_input:
            continue

        if user_input in {"/exit", "/quit"}:
            console.print("[yellow]bye[/yellow]")
            break

        if user_input == "/help":
            console.print(
                "[bold]Euler REPL commands[/bold]\n"
                "  /sql <requirement>               — production-grade SQL generation\n"
                "  /replace <file> <s> <e> <instr>  — rewrite selected line range\n"
                "  /convert <file> <target_lang>    — convert file to another language\n"
                "  /convert-code <src>→<tgt>        — paste & convert inline code\n"
                "  /auto <goal> [| <verify_cmd>]    — run autopilot loop\n"
                "  /memory <query>                  — search past session memory\n"
                "  /index [full]                    — build/update semantic index\n"
                "  /search <query>                  — semantic code search\n"
                "  /graph                           — build cross-language code graph\n"
                "  /exit | /quit                    — exit REPL"
            )
            continue

        # ── /sql ─────────────────────────────────────────────────────────────
        if user_input.startswith("/sql "):
            requirement = user_input.removeprefix("/sql ").strip()
            if not requirement:
                continue
            try:
                console.print(agent.generate_sql(requirement))
            except Exception as exc:
                _print_error(console, exc)
            continue

        # ── /replace ─────────────────────────────────────────────────────────
        if user_input.startswith("/replace "):
            parts = user_input.split(" ", 4)
            if len(parts) < 5:
                console.print("[red]usage: /replace <file> <start> <end> <instruction>[/red]")
                continue
            _, file_path, start, end, instruction = parts
            target = Path(file_path)
            if not target.exists():
                console.print(f"[red]file not found: {target}[/red]")
                continue
            selected_lines = read_file(str(target)).splitlines()
            try:
                start_line, end_line = int(start), int(end)
            except ValueError:
                console.print("[red]start and end must be integers[/red]")
                continue
            try:
                selected_text = "\n".join(selected_lines[start_line - 1 : end_line])
                replacement = agent.rewrite_selection(str(target), selected_text, instruction)
                message = replace_range(str(target), start_line, end_line, replacement)
                console.print(f"[green]{message}[/green]")
            except Exception as exc:
                _print_error(console, exc)
            continue

        # ── /auto ─────────────────────────────────────────────────────────────
        if user_input.startswith("/auto "):
            payload = user_input.removeprefix("/auto ").strip()
            goal, sep, verify = payload.partition(" | ")
            try:
                output = run_autopilot(
                    agent=agent,
                    goal=goal,
                    workdir=str(Path.cwd()),
                    max_rounds=4,
                    verify_command=verify if sep else None,
                )
                console.print(output)
            except Exception as exc:
                _print_error(console, exc)
            continue

        # ── /memory ───────────────────────────────────────────────────────────
        if user_input.startswith("/memory "):
            query = user_input.removeprefix("/memory ").strip()
            rows = search_memory(project=str(Path.cwd()), query=query, limit=3)
            if not rows:
                console.print("[yellow]No matching memory found.[/yellow]")
            else:
                for row in rows:
                    console.print(f"[bold]{row.timestamp}[/bold] {row.goal}")
                    console.print(row.result[:500])
                    console.print("─" * 40)
            continue

        # ── /index ────────────────────────────────────────────────────────────
        if user_input.startswith("/index"):
            full = user_input.strip().lower() == "/index full"
            console.print(index_path(str(Path.cwd()), incremental=not full))
            continue

        # ── /search ───────────────────────────────────────────────────────────
        if user_input.startswith("/search "):
            query = user_input.removeprefix("/search ").strip()
            hits = search_index(str(Path.cwd()), query, limit=5)
            if not hits:
                console.print("[yellow]No semantic hits found. Run /index first.[/yellow]")
            else:
                for hit in hits:
                    console.print(
                        f"[bold]{hit['path']}[/bold] "
                        f"[dim]lines {hit['start_line']}-{hit['end_line']}[/dim]"
                    )
                    console.print(hit["content"][:450])
                    console.print("─" * 40)
            continue

        # ── /graph ────────────────────────────────────────────────────────────
        if user_input == "/graph":
            console.print(build_code_graph(str(Path.cwd())))
            continue

        # ── /convert <file> <lang> ────────────────────────────────────────────
        if user_input.startswith("/convert "):
            parts = user_input.split(" ", 2)
            if len(parts) < 3:
                console.print("[red]usage: /convert <file> <target_lang>[/red]")
                continue
            _, file_path, target_lang = parts
            console.print(f"[cyan]Converting {file_path} → {target_lang}...[/cyan]")
            try:
                result = agent.convert_file(file_path.strip(), target_lang.strip())
                console.print(result)
            except Exception as exc:
                _print_error(console, exc)
            continue

        # ── /convert-code <src>→<tgt> ─────────────────────────────────────────
        if user_input.startswith("/convert-code "):
            spec = user_input.removeprefix("/convert-code ").strip()
            arrow = "→" if "→" in spec else "->" if "->" in spec else None
            if not arrow:
                console.print("[red]usage: /convert-code <src_lang>→<tgt_lang>[/red]")
                continue
            src_lang, tgt_lang = [p.strip() for p in spec.split(arrow, 1)]
            console.print("[cyan]Paste source code, then a line with just '---' to convert:[/cyan]")
            lines: list[str] = []
            while True:
                line = console.input("")
                if line.strip() == "---":
                    break
                lines.append(line)
            source_code = "\n".join(lines)
            if not source_code.strip():
                console.print("[yellow]No code provided.[/yellow]")
                continue
            console.print(f"[cyan]Converting {src_lang} → {tgt_lang}...[/cyan]")
            try:
                result = agent.convert_language(source_code, src_lang, tgt_lang)
                console.print(result)
            except Exception as exc:
                _print_error(console, exc)
            continue

        # ── free-form prompt → full agent run ─────────────────────────────────
        try:
            output = agent.run(user_input)
            console.print(output)
        except Exception as exc:
            _print_error(console, exc)
