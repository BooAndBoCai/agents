"""Command-line interface for the codebase analysis agent.

Examples
--------
Scan a project and ask a question (uses the built-in stub LLM by default)::

    codebase-analyze --path /my/project --query "Where is the login logic?"

Scan with a real OpenAI model::

    OPENAI_API_KEY=sk-... codebase-analyze \\
        --path /my/project \\
        --query "How are errors handled?" \\
        --model gpt-4o

Show the codebase summary only::

    codebase-analyze --path /my/project --summary
"""

from __future__ import annotations

import argparse
import os
import sys

from agents.codebase_analyzer import CodebaseAnalyzer
from agents.explore_agent import ExploreAgent, StubLLMClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codebase-analyze",
        description=(
            "Quick codebase analysis — ask questions about your code "
            "without adding to your main context."
        ),
    )
    parser.add_argument(
        "--path",
        "-p",
        default=".",
        metavar="DIR",
        help="Root directory (or file) to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "--query",
        "-q",
        metavar="QUESTION",
        help="Question to ask about the codebase.",
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Print a summary of the indexed codebase and exit.",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        nargs="+",
        metavar="EXT",
        help="Restrict search to these extensions (e.g. .py .js).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of code snippets to retrieve (default: 10).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help=(
            "OpenAI model name (e.g. gpt-4o). "
            "Requires the 'openai' package and OPENAI_API_KEY env var."
        ),
    )
    return parser


def _make_llm_client(model: str | None):
    """Return an LLM client based on CLI options."""
    if model is None:
        return StubLLMClient()

    try:
        from openai import OpenAI  # type: ignore[import]

        from agents.explore_agent import OpenAILLMClient

        return OpenAILLMClient(OpenAI(), model=model)
    except ImportError:
        print(
            "Error: the 'openai' package is required to use --model. "
            "Install it with: pip install 'agents[openai]'",
            file=sys.stderr,
        )
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``codebase-analyze`` command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    path = os.path.abspath(args.path)
    if not os.path.exists(path):
        print(f"Error: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {path} …", file=sys.stderr)
    analyzer = CodebaseAnalyzer()
    analyzer.scan(path)
    print(f"Indexed {analyzer.file_count} file(s).", file=sys.stderr)

    if args.summary:
        print(analyzer.summary())
        return

    if not args.query:
        parser.print_help()
        sys.exit(0)

    llm_client = _make_llm_client(args.model)
    agent = ExploreAgent(
        analyzer,
        llm_client=llm_client,
        max_search_results=args.max_results,
    )

    result = agent.ask(args.query, extensions=args.extensions)
    print(result)


if __name__ == "__main__":
    main()
