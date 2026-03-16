"""ExploreAgent – answers questions about a codebase in an isolated context.

The agent's conversation history is kept *entirely separate* from the
caller's main context.  Callers can therefore ask as many follow-up
questions as they like without any of the codebase content ever leaking
into their primary conversation.
"""

from __future__ import annotations

import os
import textwrap
from typing import Iterator

from codebase_explorer.analyzer import CodebaseAnalyzer, CodebaseIndex, SourceFile
from codebase_explorer.context import AnalysisContext

# Maximum number of bytes of source code included in the initial prompt.
_DEFAULT_CONTEXT_BYTES = 80_000

_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are an expert code-analysis assistant.  The user will ask you
    questions about the codebase described below.  Answer concisely and
    accurately, referencing file names and line numbers where helpful.

    ## Codebase summary
    {summary}

    ## Source files (most important files shown first)
    {source_files}
    """
)


class ExploreAgent:
    """An agent that performs quick codebase analysis in a separate context.

    The agent is intentionally stateless from the *caller's* perspective:
    every :class:`ExploreAgent` instance owns its own :class:`AnalysisContext`
    that is never shared with or visible from the outside.

    Parameters
    ----------
    path:
        Root directory of the codebase to analyse.
    api_key:
        OpenAI API key.  Defaults to the ``OPENAI_API_KEY`` environment
        variable when omitted.
    model:
        OpenAI chat model to use (default ``gpt-4o-mini``).
    max_context_bytes:
        Upper bound on the number of source-code bytes injected into the
        initial system prompt.
    analyzer:
        Optional pre-configured :class:`CodebaseAnalyzer`.  Useful in
        tests to inject a mock analyser.
    """

    def __init__(
        self,
        path: str,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_context_bytes: int = _DEFAULT_CONTEXT_BYTES,
        analyzer: CodebaseAnalyzer | None = None,
    ) -> None:
        self.path = path
        self.model = model
        self.max_context_bytes = max_context_bytes
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._analyzer = analyzer or CodebaseAnalyzer(path)
        self._index: CodebaseIndex | None = None
        self._ctx: AnalysisContext | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def ask(self, question: str) -> str:
        """Ask *question* about the codebase and return the answer.

        The first call triggers a full scan of the codebase; subsequent
        calls reuse the cached index and context.

        The answer is returned as a plain string and nothing is appended
        to any external conversation history.
        """
        ctx = self._get_or_build_context()
        ctx.add_user(question)
        answer = self._complete(ctx)
        ctx.add_assistant(answer)
        return answer

    def reset(self) -> None:
        """Clear the analysis context (but keep the index cache)."""
        if self._ctx is not None:
            self._ctx.clear()

    def summary(self) -> str:
        """Return a human-readable summary of the scanned codebase."""
        return self._get_or_build_index().summary()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_or_build_index(self) -> CodebaseIndex:
        if self._index is None:
            self._index = self._analyzer.scan()
        return self._index

    def _get_or_build_context(self) -> AnalysisContext:
        if self._ctx is None:
            index = self._get_or_build_index()
            system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                summary=index.summary(),
                source_files=self._format_source_files(index),
            )
            self._ctx = AnalysisContext(system_prompt=system_prompt)
        return self._ctx

    def _format_source_files(self, index: CodebaseIndex) -> str:
        """Build the source-file block injected into the system prompt."""
        budget = self.max_context_bytes
        parts: list[str] = []
        for sf in self._prioritise(index.files):
            if budget <= 0:
                parts.append(
                    f"### {sf.relative_path}\n[omitted – context budget exhausted]\n"
                )
                continue
            content = sf.read(max_bytes=min(budget, sf.size_bytes or 1))
            budget -= len(content.encode("utf-8", errors="replace"))
            parts.append(f"### {sf.relative_path}\n```\n{content}\n```\n")
        return "\n".join(parts)

    @staticmethod
    def _prioritise(files: list[SourceFile]) -> list[SourceFile]:
        """Return files sorted so the most likely 'entry points' come first."""
        priority_names = {
            "readme.md", "readme.txt", "readme.rst",
            "main.py", "app.py", "index.py",
            "main.ts", "index.ts", "app.ts",
            "main.js", "index.js", "app.js",
            "main.go", "main.rs", "main.java",
        }

        def key(sf: SourceFile) -> tuple[int, int, str]:
            name = sf.path.name.lower()
            depth = sf.relative_path.count(os.sep)
            is_priority = 0 if name in priority_names else 1
            return (is_priority, depth, sf.relative_path)

        return sorted(files, key=key)

    def _complete(self, ctx: AnalysisContext) -> str:
        """Call the OpenAI chat-completion API and return the reply text."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required.  Install it with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=ctx.to_api_messages(),  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""
