"""Terminal avatar and activation banner."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def print_activation_banner(provider: str, model: str) -> None:
    """Print a recognizable CLI identity banner."""
    console = Console()
    title = Text(" E U L E R ", style="bold cyan")
    subtitle = Text(f"provider={provider}  model={model}", style="bright_black")
    claw = r"""
      /\_/\
   __( o.o )__
  / _/\_V_/\_ \
 /_/  EULER  \_\
"""
    body = Text(claw, style="bright_magenta")
    panel = Panel.fit(
        Text.assemble(title, "\n", body, "\n", subtitle),
        border_style="cyan",
        title="Activated",
    )
    console.print(panel)
