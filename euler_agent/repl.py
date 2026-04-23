"""Interactive shell for Euler agent."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from euler_agent.agent import EulerAgent
from euler_agent.autopilot import run_autopilot
from euler_agent.code_graph import build_code_graph
from euler_agent.memory import search_memory
from euler_agent.semantic_index import index_path, search_index
from euler_agent.tools import read_file, replace_range


def run_repl(agent: EulerAgent) -> None:
    console = Console()
    console.print("[bold cyan]Euler REPL[/bold cyan] - type /help for commands")

    while True:
        user_input = console.input("[bold green]euler> [/bold green]").strip()
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
                "  /convert-code <src_lang>→<tgt_lang> — paste code, convert inline\n"
                "  /auto <goal> [| <verify_cmd>]    — run autopilot loop\n"
                "  /memory <query>                  — search past session memory\n"
                "  /index [full]                    — build/update semantic index\n"
                "  /search <query>                  — semantic code search\n"
                "  /graph                           — build cross-language code graph\n"
                "  /exit | /quit                    — exit REPL"
            )
            continue

        if user_input.startswith("/sql "):
            requirement = user_input.removeprefix("/sql ").strip()
            if requirement:
                console.print(agent.generate_sql(requirement))
            continue

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
                start_line = int(start)
                end_line = int(end)
            except ValueError:
                console.print("[red]start and end must be integers[/red]")
                continue
            selected_text = "\n".join(selected_lines[start_line - 1 : end_line])
            replacement = agent.rewrite_selection(str(target), selected_text, instruction)
            message = replace_range(str(target), start_line, end_line, replacement)
            console.print(f"[green]{message}[/green]")
            continue

        if user_input.startswith("/auto "):
            payload = user_input.removeprefix("/auto ").strip()
            goal, sep, verify = payload.partition(" | ")
            output = run_autopilot(
                agent=agent,
                goal=goal,
                workdir=str(Path.cwd()),
                max_rounds=4,
                verify_command=verify if sep else None,
            )
            console.print(output)
            continue

        if user_input.startswith("/memory "):
            query = user_input.removeprefix("/memory ").strip()
            rows = search_memory(project=str(Path.cwd()), query=query, limit=3)
            if not rows:
                console.print("[yellow]No matching memory found.[/yellow]")
                continue
            for row in rows:
                console.print(f"[bold]{row.timestamp}[/bold] {row.goal}")
                console.print(row.result[:500])
                console.print("-" * 40)
            continue

        if user_input.startswith("/index"):
            full = user_input.strip().lower() == "/index full"
            console.print(index_path(str(Path.cwd()), incremental=not full))
            continue

        if user_input.startswith("/search "):
            query = user_input.removeprefix("/search ").strip()
            hits = search_index(str(Path.cwd()), query, limit=5)
            if not hits:
                console.print("[yellow]No semantic hits found. Run /index first.[/yellow]")
                continue
            for hit in hits:
                console.print(
                    f"[bold]{hit['path']}[/bold] "
                    f"[dim]lines {hit['start_line']}-{hit['end_line']}[/dim]"
                )
                console.print(hit["content"][:450])
                console.print("-" * 40)
            continue

        if user_input == "/graph":
            console.print(build_code_graph(str(Path.cwd())))
            continue

        if user_input.startswith("/convert "):
            parts = user_input.split(" ", 2)
            if len(parts) < 3:
                console.print("[red]usage: /convert <file> <target_lang>[/red]")
                continue
            _, file_path, target_lang = parts
            console.print(f"[cyan]Converting {file_path} → {target_lang}...[/cyan]")
            result = agent.convert_file(file_path.strip(), target_lang.strip())
            console.print(result)
            continue

        if user_input.startswith("/convert-code "):
            spec = user_input.removeprefix("/convert-code ").strip()
            if "→" not in spec and "->" not in spec:
                console.print("[red]usage: /convert-code <src_lang>→<tgt_lang>[/red]")
                continue
            arrow = "→" if "→" in spec else "->"
            src_lang, tgt_lang = [p.strip() for p in spec.split(arrow, 1)]
            console.print(f"[cyan]Paste source code, then a line with just '---' to convert:[/cyan]")
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
            result = agent.convert_language(source_code, src_lang, tgt_lang)
            console.print(result)
            continue

        output = agent.run(user_input)
        console.print(output)
