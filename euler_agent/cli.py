"""Typer CLI entrypoint for Euler."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from euler_agent.agent import EulerAgent
from euler_agent.avatar import print_activation_banner
from euler_agent.config import AgentConfig, Provider, load_config, save_config
from euler_agent.repl import run_repl

app = typer.Typer(help="Euler coding agent CLI")
console = Console()


def _build_agent(cfg: AgentConfig) -> EulerAgent:
    if not cfg.api_key:
        raise typer.BadParameter(
            "API key is missing. Run: Euler config set --provider <provider> --model <model>"
        )
    return EulerAgent(provider=cfg.provider, model_name=cfg.model, api_key=cfg.api_key)


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
    api_key: str = typer.Option(..., prompt=True, hide_input=True),
) -> None:
    cfg = AgentConfig(provider=provider, model=model, api_key=api_key)
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


if __name__ == "__main__":
    app()
