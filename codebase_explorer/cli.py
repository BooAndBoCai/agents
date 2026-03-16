"""Command-line interface for the codebase explorer."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from codebase_explorer.agent import ExploreAgent
from codebase_explorer.analyzer import CodebaseAnalyzer

console = Console()


@click.group()
def main() -> None:
    """Quick codebase analysis – ask questions without touching your main context."""


@main.command("ask")
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
@click.argument("question", required=False)
@click.option("--model", default="gpt-4o-mini", show_default=True, help="OpenAI model to use.")
@click.option(
    "--max-context-kb",
    default=80,
    show_default=True,
    type=int,
    help="Maximum kilobytes of source code to include in the context.",
)
@click.option("--api-key", envvar="OPENAI_API_KEY", default=None, help="OpenAI API key.")
def ask_command(
    path: str,
    question: str | None,
    model: str,
    max_context_kb: int,
    api_key: str | None,
) -> None:
    """Ask a single QUESTION about the codebase at PATH.

    If QUESTION is omitted, an interactive REPL is started instead.

    \b
    Examples
    --------
      explore ask . "What does this codebase do?"
      explore ask /path/to/repo "Where is authentication handled?"
      explore ask /path/to/repo        # starts interactive REPL
    """
    agent = ExploreAgent(
        path=path,
        api_key=api_key,
        model=model,
        max_context_bytes=max_context_kb * 1024,
    )

    if question:
        _run_single_question(agent, question)
    else:
        _run_repl(agent)


@main.command("summary")
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
def summary_command(path: str) -> None:
    """Print a summary of the files found in PATH."""
    analyzer = CodebaseAnalyzer(path)
    index = analyzer.scan()
    console.print(Panel(index.summary(), title="Codebase summary", border_style="cyan"))


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _run_single_question(agent: ExploreAgent, question: str) -> None:
    with console.status("Analysing codebase…"):
        answer = agent.ask(question)
    console.print(Markdown(answer))


def _run_repl(agent: ExploreAgent) -> None:
    console.print(
        Panel(
            "[bold cyan]Codebase Explorer[/bold cyan]\n"
            "Ask questions about your codebase.  "
            "Type [bold]exit[/bold] or press [bold]Ctrl+D[/bold] to quit.",
            border_style="cyan",
        )
    )

    # Show summary first so the user knows what was indexed.
    with console.status("Scanning codebase…"):
        summary_text = agent.summary()
    console.print(Panel(summary_text, title="Codebase summary", border_style="dim"))

    while True:
        try:
            question = Prompt.ask("[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if question.strip().lower() in {"exit", "quit", "q"}:
            console.print("[dim]Bye![/dim]")
            break

        if not question.strip():
            continue

        with console.status("Thinking…"):
            try:
                answer = agent.ask(question)
            except (OSError, ValueError, RuntimeError) as exc:
                console.print(f"[red]Error:[/red] {exc}")
                continue

        console.print(Panel(Markdown(answer), title="[bold blue]Answer[/bold blue]", border_style="blue"))
