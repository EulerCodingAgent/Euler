"""Typer CLI entrypoint for Euler."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from euler_agent.core.agent import EulerAgent
from euler_agent.core.autopilot import run_autopilot
from euler_agent.ui.avatar import print_activation_banner
from euler_agent.analysis.code_graph import build_code_graph
from euler_agent.config.settings import AgentConfig, Provider, load_config, save_config
from euler_agent.memory.store import search_memory
from euler_agent.analysis.semantic_index import index_path, search_index
from euler_agent.repl import run_repl

app = typer.Typer(help="Euler coding agent CLI")
console = Console()


def _build_agent(cfg: AgentConfig) -> EulerAgent:
    clean_key = cfg.api_key.strip()
    if not clean_key:
        raise typer.BadParameter(
            "API key is missing. Run: Euler config set --provider <provider> --model <model>"
        )
    return EulerAgent(provider=cfg.provider, model_name=cfg.model, api_key=clean_key)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """
    Start interactive Euler REPL if no command is given.
    """
    if ctx.invoked_subcommand is None:
        cfg = load_config()
        print_activation_banner(cfg.provider, cfg.model)
        agent = _build_agent(cfg)
        run_repl(agent)


config_app = typer.Typer(help="Manage provider/model/API key configuration")
app.add_typer(config_app, name="config")


@config_app.command("show")
def show_config() -> None:
    cfg = load_config()
    hidden = cfg.api_key[:4] + "..." if cfg.api_key else "(not set)"
    console.print(f"provider={cfg.provider}")
    console.print(f"model={cfg.model}")
    console.print(f"api_key={hidden}")


@config_app.command("set")
def set_config(
    provider: Provider = typer.Option(..., help="openai | anthropic | gemini"),
    model: str = typer.Option(..., help="Model slug for provider"),
    api_key: str = typer.Option(..., prompt=True, hide_input=False),
) -> None:
    clean_key = api_key.strip()
    if not clean_key:
        raise typer.BadParameter("API key cannot be empty.")
    if provider == "gemini" and not clean_key.startswith("AIza"):
        raise typer.BadParameter(
            "Gemini API keys usually start with 'AIza'. "
            "Please paste a valid Google AI Studio API key."
        )
    cfg = AgentConfig(provider=provider, model=model, api_key=clean_key)
    save_config(cfg)
    console.print("[green]Config saved.[/green]")


@app.command("run")
def run_once(
    prompt: str = typer.Argument(..., help="Task prompt"),
    workdir: Optional[Path] = typer.Option(None, help="Project directory"),
) -> None:
    cfg = load_config()
    print_activation_banner(cfg.provider, cfg.model)
    agent = _build_agent(cfg)
    output = agent.run(prompt, workdir=str(workdir) if workdir else None)
    console.print(output)


@app.command("autopilot")
def autopilot(
    goal: str = typer.Argument(..., help="Autonomous coding goal"),
    workdir: Optional[Path] = typer.Option(None, help="Project directory"),
    max_rounds: int = typer.Option(4, min=1, max=12, help="Max execution rounds"),
    policy_profile: str = typer.Option(
        "normal",
        help="Guardrail profile: safe | normal | aggressive",
    ),
    require_approval_for_risky: bool = typer.Option(
        True,
        "--require-approval-for-risky/--no-require-approval-for-risky",
        help="Require approval gate for risky commands/actions",
    ),
    auto_approve_risky: bool = typer.Option(
        False,
        "--auto-approve-risky",
        help="Auto approve risky operations (use carefully)",
    ),
    verify_command: Optional[str] = typer.Option(
        None,
        help="Optional validation command executed each round (e.g. pytest -q)",
    ),
    max_file_mutations: int = typer.Option(
        25,
        min=1,
        max=500,
        help="Maximum file mutation actions allowed in this run",
    ),
) -> None:
    cfg = load_config()
    print_activation_banner(cfg.provider, cfg.model)
    agent = _build_agent(cfg)
    wd = str((workdir or Path.cwd()).resolve())
    output = run_autopilot(
        agent=agent,
        goal=goal,
        workdir=wd,
        max_rounds=max_rounds,
        policy_profile=policy_profile,
        require_approval_for_risky=require_approval_for_risky,
        auto_approve_risky=auto_approve_risky,
        verify_command=verify_command,
        max_file_mutations=max_file_mutations,
    )
    console.print(output)


@app.command("memory")
def memory(
    query: str = typer.Argument(..., help="Search phrase for prior project runs"),
    workdir: Optional[Path] = typer.Option(None, help="Project directory"),
    limit: int = typer.Option(3, min=1, max=20, help="Max memories to show"),
) -> None:
    wd = str((workdir or Path.cwd()).resolve())
    rows = search_memory(project=wd, query=query, limit=limit)
    if not rows:
        console.print("[yellow]No matching memory found.[/yellow]")
        return
    for row in rows:
        console.print(f"[bold]{row.timestamp}[/bold] {row.goal}")
        console.print(row.result[:500])
        console.print("-" * 40)


@app.command("init")
def init_workspace(path: Optional[Path] = typer.Option(None, help="Target project root")) -> None:
    """
    Create Euler instruction folder used as project guidance memory.
    """
    root = (path or Path.cwd()).resolve()
    instruction_dir = root / "Euler"
    instruction_dir.mkdir(parents=True, exist_ok=True)
    default_md = instruction_dir / "project.md"
    if not default_md.exists():
        default_md.write_text(
            "# Euler Project Instructions\n"
            "- Add coding standards and architecture notes here.\n"
            "- Euler will include these notes in every run.\n",
            encoding="utf-8",
        )
    console.print(f"[green]Initialized instructions at {instruction_dir}[/green]")


@app.command("index")
def build_index(
    workdir: Optional[Path] = typer.Option(None, help="Project directory"),
    full: bool = typer.Option(
        False,
        "--full",
        help="Force full reindex (default is incremental)",
    ),
) -> None:
    wd = str((workdir or Path.cwd()).resolve())
    message = index_path(wd, incremental=not full)
    console.print(f"[green]{message}[/green]")


@app.command("search-code")
def semantic_search(
    query: str = typer.Argument(..., help="Natural-language code search query"),
    workdir: Optional[Path] = typer.Option(None, help="Project directory"),
    limit: int = typer.Option(5, min=1, max=20, help="Result count"),
) -> None:
    wd = str((workdir or Path.cwd()).resolve())
    hits = search_index(wd, query, limit=limit)
    if not hits:
        console.print("[yellow]No semantic hits found. Run `Euler index` first.[/yellow]")
        return
    for hit in hits:
        console.print(
            f"[bold]{hit['path']}[/bold] "
            f"[dim]lines {hit['start_line']}-{hit['end_line']}[/dim]"
        )
        console.print(hit["content"][:450])
        console.print("-" * 40)


@app.command("graph")
def graph(
    workdir: Optional[Path] = typer.Option(None, help="Project directory"),
) -> None:
    wd = str((workdir or Path.cwd()).resolve())
    message = build_code_graph(wd)
    console.print(f"[green]{message}[/green]")


@app.command("convert")
def convert_file(
    file: Path = typer.Argument(..., help="Source file to convert"),
    target_lang: str = typer.Argument(
        ...,
        help="Target language (python|typescript|javascript|go|rust|java|kotlin|sql|...)",
    ),
    output: Optional[Path] = typer.Option(None, help="Write converted code to this file"),
) -> None:
    cfg = load_config()
    print_activation_banner(cfg.provider, cfg.model)
    agent = _build_agent(cfg)
    console.print(f"[cyan]Converting {file} → {target_lang}...[/cyan]")
    result = agent.convert_file(str(file), target_lang)
    if output:
        output.write_text(result, encoding="utf-8")
        console.print(f"[green]Written to {output}[/green]")
    else:
        console.print(result)


if __name__ == "__main__":
    app()
