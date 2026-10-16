"""What every ``replay`` command shares: the console, exit codes and error exits."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2

console = Console(highlight=False)

DB_OPTION = typer.Option(help="Run store URL; defaults to $REPLAY_DB_URL or .replay/replay.db.")


def fail(message: str) -> typer.Exit:
    """Print ``message`` as an error; raise the result to exit with :data:`EXIT_USAGE`."""
    console.print(f"[bold red]error:[/] {escape(message)}")
    return typer.Exit(EXIT_USAGE)
