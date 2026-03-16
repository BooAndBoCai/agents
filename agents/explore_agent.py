"""ExploreAgent: ask questions about a codebase in a separate context.

The agent retrieves relevant code snippets from a :class:`CodebaseAnalyzer`
and forwards them—together with the user's question—to an LLM.  Only the
*retrieved snippets* end up in the LLM call; the entire codebase is **not**
added to the caller's main context.

Minimal usage (no external dependencies – uses the built-in stub LLM)::

    from agents import CodebaseAnalyzer, ExploreAgent

    analyzer = CodebaseAnalyzer().scan("/path/to/project")
    agent = ExploreAgent(analyzer)
    response = agent.ask("Where is the authentication logic?")
    print(response.answer)

Usage with the OpenAI client::

    from openai import OpenAI
    from agents import CodebaseAnalyzer, ExploreAgent
    from agents.explore_agent import OpenAILLMClient

    analyzer = CodebaseAnalyzer().scan("/path/to/project")
    llm = OpenAILLMClient(OpenAI(), model="gpt-4o")
    agent = ExploreAgent(analyzer, llm_client=llm)
    response = agent.ask("How are database migrations handled?")
    print(response.answer)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

from agents.codebase_analyzer import CodebaseAnalyzer, SearchResult

# ---------------------------------------------------------------------------
# LLM client protocol
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Minimal protocol for an LLM backend.

    Any callable object (or class instance) that accepts a list of message
    dicts and returns a string satisfies this protocol.
    """

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send *messages* to the model and return the assistant reply."""
        ...


# ---------------------------------------------------------------------------
# Built-in stub – useful for testing and offline use
# ---------------------------------------------------------------------------


class StubLLMClient:
    """A no-dependency stub that echoes context without calling any API.

    Useful for smoke-testing without API keys.
    """

    def complete(self, messages: list[dict[str, str]]) -> str:
        # Return a short, deterministic reply so tests can assert on it.
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        first_line = user_msg.splitlines()[0] if user_msg else "(empty)"
        return f"[StubLLMClient] Received question: {first_line!r}"


# ---------------------------------------------------------------------------
# OpenAI adapter (optional – only used when openai is installed)
# ---------------------------------------------------------------------------


class OpenAILLMClient:
    """Thin wrapper around the OpenAI Chat Completions API.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` (or ``openai.AsyncOpenAI``) instance.
    model:
        Model name, e.g. ``"gpt-4o"``.
    temperature:
        Sampling temperature passed to the API.
    """

    def __init__(self, client: Any, model: str = "gpt-4o", temperature: float = 0.2):
        self._client = client
        self._model = model
        self._temperature = temperature

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=messages,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ExploreResult:
    """The response from :meth:`ExploreAgent.ask`."""

    question: str
    """The original question posed to the agent."""

    answer: str
    """The LLM's answer."""

    sources: list[SearchResult]
    """The code snippets used to answer the question."""

    def __str__(self) -> str:
        lines = [
            f"Q: {self.question}",
            "",
            f"A: {self.answer}",
        ]
        if self.sources:
            lines += ["", "Sources:"]
            seen_paths: set[str] = set()
            for src in self.sources:
                if src.path not in seen_paths:
                    lines.append(f"  • {src.path}")
                    seen_paths.add(src.path)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ExploreAgent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a code analysis assistant. You will be given a question about a
    software codebase and a set of relevant code snippets retrieved from that
    codebase.  Answer the question concisely and accurately based solely on the
    provided snippets.  If the snippets do not contain enough information to
    answer the question, say so clearly.  Do not invent code that is not shown
    in the snippets.
    """
)


@dataclass
class ExploreAgent:
    """Ask questions about a codebase in a *separate* context.

    The agent uses :class:`CodebaseAnalyzer` to retrieve relevant snippets and
    then asks the configured LLM to answer the question.  The full codebase is
    **never** placed in context—only the retrieved snippets are forwarded to
    the LLM, keeping token usage low and the caller's context clean.

    Parameters
    ----------
    analyzer:
        A :class:`CodebaseAnalyzer` that has already been scanned (or will be
        scanned lazily when :meth:`ask` is first called).
    llm_client:
        An object implementing the :class:`LLMClient` protocol.  Defaults to
        :class:`StubLLMClient` (no API key required).
    max_search_results:
        How many code snippets to include in the LLM prompt.
    system_prompt:
        Override the default system prompt.
    """

    analyzer: CodebaseAnalyzer
    llm_client: Any = field(default_factory=StubLLMClient)
    max_search_results: int = 10
    system_prompt: str = _SYSTEM_PROMPT

    def ask(
        self,
        question: str,
        *,
        extensions: Optional[Sequence[str]] = None,
    ) -> ExploreResult:
        """Ask *question* about the codebase.

        Parameters
        ----------
        question:
            A natural-language question about the codebase.
        extensions:
            Optional list of file extensions to restrict the search to.

        Returns
        -------
        ExploreResult
            Contains the answer and the source snippets used.
        """
        sources = self.analyzer.search(
            question,
            max_results=self.max_search_results,
            extensions=list(extensions) if extensions else None,
        )

        context_block = self._build_context(sources)
        messages = self._build_messages(question, context_block)
        answer = self.llm_client.complete(messages)

        return ExploreResult(question=question, answer=answer, sources=sources)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(self, sources: list[SearchResult]) -> str:
        if not sources:
            return "(No relevant snippets found in the codebase.)"
        parts: list[str] = []
        for i, src in enumerate(sources, start=1):
            parts.append(
                f"--- Snippet {i} | {src.path}:{src.line_number} ---\n{src.snippet}"
            )
        return "\n\n".join(parts)

    def _build_messages(
        self, question: str, context_block: str
    ) -> list[dict[str, str]]:
        user_content = (
            f"Codebase snippets:\n\n{context_block}\n\n"
            f"Question: {question}"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
