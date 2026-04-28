"""Typer CLI entrypoint for Euler."""

from __future__ import annotations

import time
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


def _safe_graph_filename_for_target(cwd: Path, target_root: Path) -> str:
    if target_root.resolve() == cwd.resolve():
        return "knowledge_graph.json"
    safe_parts = [p for p in target_root.parts if p not in {"/", "\\", ":"}]
    safe = "_".join(safe_parts).strip("_")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in safe).strip("_")
    if not safe:
        safe = "selected_folder"
    return f"knowledge_graph_{safe}.json"


def _build_agent(cfg: AgentConfig) -> EulerAgent:
    clean_key = cfg.api_key.strip()
    if cfg.provider in {"openai", "anthropic", "gemini"} and not clean_key:
        raise typer.BadParameter(
            "API key is missing. Run: Euler config set --provider <provider> --model <model>"
        )
    if cfg.provider == "local" and not cfg.base_url.strip():
        raise typer.BadParameter(
            "Local provider requires base_url. Run: "
            "Euler config set --provider local --model <model> --base-url <url>"
        )
    return EulerAgent(
        provider=cfg.provider,
        model_name=cfg.model,
        api_key=clean_key,
        base_url=cfg.base_url.strip(),
    )


def _probe_connection(agent: EulerAgent) -> None:
    """
    Send a tiny probe prompt to verify model connectivity.
    """
    probe = agent.ask("Hi", role="assistant")
    preview = (probe or "").strip().replace("\n", " ")
    if not preview:
        preview = "(empty response)"
    console.print(f"[bold green]Connection OK[/bold green] [dim]AI:[/dim] {preview[:180]}")


def _startup_animation() -> None:
    """
    Lightweight terminal animation before startup menu.
    """
    frames = ["[cyan]•[/cyan]", "[cyan]••[/cyan]", "[cyan]•••[/cyan]"]
    for frame in frames:
        console.print(f"[dim]Preparing startup menu {frame}[/dim]")
        time.sleep(0.08)
        print("\x1b[1A\x1b[2K", end="", flush=True)
    console.print("[dim]Preparing startup menu [green]done[/green][/dim]\n")


def _arrow_select(
    title: str,
    text: str,
    values: list[tuple[str, str]],
    default: str,
) -> str:
    """
    In-terminal selector. On Windows uses arrow keys; otherwise numeric fallback.
    """
    try:
        import msvcrt  # type: ignore

        default_idx = 0
        for i, (value, _) in enumerate(values):
            if value == default:
                default_idx = i
                break
        idx = default_idx
        console.print("")
        console.print(f"[bold cyan]{title}[/bold cyan]")
        console.print(f"[white]{text}[/white]")
        console.print("[dim]Use ↑/↓ and Enter (or j/k).[/dim]\n")
        total_lines = len(values) + 1

        def _draw() -> None:
            print("\r", end="", flush=True)
            for i, (_, label) in enumerate(values):
                prefix = "[bold bright_cyan]→[/bold bright_cyan]" if i == idx else " "
                color = "[bold white]" if i == idx else "[dim]"
                end = "[/bold white]" if i == idx else "[/dim]"
                console.print(f"{prefix} {color}{i + 1}) {label}{end}")
            print("")

        _draw()
        while True:
            ch = msvcrt.getwch()
            if ch in {"\r", "\n"}:
                print("\x1b[1A", end="", flush=True)
                console.print(
                    f"[green]✓ Selected:[/green] [bold]{values[idx][1]}[/bold]\n"
                )
                return values[idx][0]
            moved = False
            if ch in {"j", "J"}:
                idx = (idx + 1) % len(values)
                moved = True
            if ch in {"k", "K"}:
                idx = (idx - 1) % len(values)
                moved = True
            elif ch in {"\x00", "\xe0"}:
                key = msvcrt.getwch()
                if key == "H":  # up
                    idx = (idx - 1) % len(values)
                    moved = True
                elif key == "P":  # down
                    idx = (idx + 1) % len(values)
                    moved = True
            if moved:
                print("\x1b[1A" * total_lines, end="", flush=True)
                _draw()
    except Exception:
        pass

    labels = "\n".join(f"  {idx + 1}) {label}" for idx, (_, label) in enumerate(values))
    console.print(f"\n[bold cyan]{title}[/bold cyan]\n{text}\n{labels}")
    raw = typer.prompt("Choose option", default="1").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(values):
            return values[idx][0]
    return default


def _interactive_startup_config(cfg: AgentConfig) -> AgentConfig:
    """
    Startup wizard shown before entering REPL.

    Keeps existing config support and allows quick cloud/local setup.
    """
    choice = _arrow_select(
        title="Euler Startup",
        text="Use arrow keys and Enter to choose startup flow.",
        values=[
            ("cloud", "Cloud/API model (OpenAI/Anthropic/Gemini)"),
            ("local", "Local LLM (Ollama/OpenAI-compatible local server)"),
            ("saved", "Continue with saved config"),
        ],
        default="saved",
    )
    if choice == "saved":
        return cfg

    if choice == "cloud":
        provider_raw = _arrow_select(
            title="Cloud Provider",
            text="Select cloud provider.",
            values=[
                ("openai", "openai"),
                ("anthropic", "anthropic"),
                ("gemini", "gemini"),
            ],
            default=cfg.provider if cfg.provider in {"openai", "anthropic", "gemini"} else "gemini",
        )
        if provider_raw not in {"openai", "anthropic", "gemini"}:
            raise typer.BadParameter("Provider must be one of: openai, anthropic, gemini.")
        model = typer.prompt("Model", default=cfg.model).strip()
        api_key = typer.prompt("API key", hide_input=False).strip()
        if not api_key:
            raise typer.BadParameter("API key cannot be empty for cloud providers.")
        if provider_raw == "gemini" and not api_key.startswith("AIza"):
            raise typer.BadParameter("Gemini API keys usually start with 'AIza'.")
        updated = AgentConfig(
            provider=provider_raw,  # type: ignore[arg-type]
            model=model,
            api_key=api_key,
            base_url="",
        )
        save_config(updated)
        console.print("[green]Cloud config saved.[/green]")
        return updated

    local_provider = _arrow_select(
        title="Local Provider",
        text="Select local provider/runtime.",
        values=[
            ("ollama", "ollama"),
            ("local", "local (OpenAI-compatible endpoint)"),
        ],
        default=cfg.provider if cfg.provider in {"ollama", "local"} else "ollama",
    )
    if local_provider not in {"ollama", "local"}:
        raise typer.BadParameter("Local provider must be 'ollama' or 'local'.")
    model = typer.prompt("Local model", default=cfg.model if cfg.provider in {"ollama", "local"} else "qwen2.5-coder:7b").strip()
    default_url = "http://localhost:11434/v1" if local_provider == "ollama" else (cfg.base_url or "http://localhost:1234/v1")
    base_url = typer.prompt("Base URL", default=default_url).strip()
    api_key = typer.prompt(
        "API key (optional for local, press Enter to skip)",
        default="",
        show_default=False,
        hide_input=False,
    ).strip()
    updated = AgentConfig(
        provider=local_provider,  # type: ignore[arg-type]
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    save_config(updated)
    console.print("[green]Local config saved.[/green]")
    return updated


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """
    Start interactive Euler REPL if no command is given.
    """
    if ctx.invoked_subcommand is None:
        cfg = load_config()
        print_activation_banner(cfg.provider, cfg.model)
        _startup_animation()
        cfg = _interactive_startup_config(cfg)
        console.print(
            f"[bold cyan]Using[/bold cyan] [white]provider={cfg.provider}[/white] "
            f"[white]model={cfg.model}[/white]\n"
        )
        agent = _build_agent(cfg)
        _probe_connection(agent)
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
    console.print(f"base_url={cfg.base_url or '(not set)'}")


@config_app.command("set")
def set_config(
    provider: Provider = typer.Option(..., help="openai | anthropic | gemini | ollama | local"),
    model: str = typer.Option(..., help="Model slug for provider"),
    api_key: Optional[str] = typer.Option(
        None,
        help="API key (required for cloud providers; optional for local providers)",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        help="Endpoint base URL for local/ollama providers",
    ),
) -> None:
    clean_key = (api_key or "").strip()
    if provider in {"openai", "anthropic", "gemini"} and not clean_key:
        raise typer.BadParameter("API key cannot be empty.")
    if provider == "gemini" and not clean_key.startswith("AIza"):
        raise typer.BadParameter(
            "Gemini API keys usually start with 'AIza'. "
            "Please paste a valid Google AI Studio API key."
        )
    resolved_base_url = (base_url or "").strip()
    if provider == "ollama" and not resolved_base_url:
        resolved_base_url = "http://localhost:11434/v1"
    if provider == "local" and not resolved_base_url:
        raise typer.BadParameter(
            "Local provider requires --base-url (e.g. http://localhost:1234/v1)."
        )
    cfg = AgentConfig(
        provider=provider,
        model=model,
        api_key=clean_key,
        base_url=resolved_base_url,
    )
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


@app.command("knowledge-graph")
def knowledge_graph(
    folder: Optional[Path] = typer.Argument(None, help="Optional sub-folder to graph"),
    workdir: Optional[Path] = typer.Option(None, help="Project root (defaults to cwd)"),
) -> None:
    """
    Build graph for root or selected folder and save into ./Euler.
    """
    cwd = (workdir or Path.cwd()).resolve()
    target = (cwd / folder).resolve() if folder and not folder.is_absolute() else (folder.resolve() if folder else cwd)
    if not target.exists() or not target.is_dir():
        raise typer.BadParameter(f"Invalid folder: {target}")
    euler_dir = cwd / "Euler"
    euler_dir.mkdir(parents=True, exist_ok=True)
    out_file = euler_dir / _safe_graph_filename_for_target(cwd, target)
    message = build_code_graph(str(target), output_path=str(out_file))
    console.print(f"[green]{message}[/green]")
    console.print(f"[cyan]Saved knowledge graph:[/cyan] {out_file}")


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
