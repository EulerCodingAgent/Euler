"""Interactive shell for Euler agent."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from euler_agent.agent import EulerAgent
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
                "/exit | /quit\n"
                "/sql <requirement>\n"
                "/replace <file> <start_line> <end_line> <instruction>\n"
                "or write a normal prompt for full-task execution."
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

        output = agent.run(user_input)
        console.print(output)
