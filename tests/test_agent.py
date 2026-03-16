"""Tests for codebase_explorer.agent (ExploreAgent)."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from codebase_explorer.agent import ExploreAgent
from codebase_explorer.analyzer import CodebaseAnalyzer


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture()
def sample_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "main.py").write_text("def main():\n    print('hello')\n")
    (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "README.md").write_text("# Sample project\n")
    return tmp_path


def _make_agent(path: str, mock_answer: str = "Mock answer") -> ExploreAgent:
    """Create an ExploreAgent whose LLM call is mocked."""
    agent = ExploreAgent(path=path, api_key="test-key")
    agent._complete = MagicMock(return_value=mock_answer)  # type: ignore[method-assign]
    return agent


# ------------------------------------------------------------------ #
# ExploreAgent.ask                                                     #
# ------------------------------------------------------------------ #


class TestExploreAgentAsk:
    def test_ask_returns_string(self, sample_repo: pathlib.Path) -> None:
        agent = _make_agent(str(sample_repo))
        result = agent.ask("What does this code do?")
        assert isinstance(result, str)
        assert result == "Mock answer"

    def test_ask_calls_complete_once(self, sample_repo: pathlib.Path) -> None:
        agent = _make_agent(str(sample_repo))
        agent.ask("Hello?")
        assert agent._complete.call_count == 1  # type: ignore[attr-defined]

    def test_ask_accumulates_history(self, sample_repo: pathlib.Path) -> None:
        agent = _make_agent(str(sample_repo), mock_answer="A")
        agent.ask("Q1")
        agent.ask("Q2")
        # Context should have 4 messages: user, assistant, user, assistant
        assert len(agent._ctx) == 4  # type: ignore[arg-type]

    def test_ask_does_not_share_context(self, sample_repo: pathlib.Path) -> None:
        agent_a = _make_agent(str(sample_repo))
        agent_b = _make_agent(str(sample_repo))
        agent_a.ask("Q from A")
        # Agent B must have a completely independent (untouched) context.
        assert agent_b._ctx is None, (
            "agent_b should not have been initialised by agent_a's ask()"
        )


# ------------------------------------------------------------------ #
# ExploreAgent.reset                                                   #
# ------------------------------------------------------------------ #


class TestExploreAgentReset:
    def test_reset_clears_context(self, sample_repo: pathlib.Path) -> None:
        agent = _make_agent(str(sample_repo))
        agent.ask("Q1")
        assert agent._ctx is not None
        agent.reset()
        assert len(agent._ctx) == 0  # type: ignore[arg-type]

    def test_reset_before_first_ask_is_safe(self, sample_repo: pathlib.Path) -> None:
        agent = _make_agent(str(sample_repo))
        agent.reset()  # Should not raise


# ------------------------------------------------------------------ #
# ExploreAgent.summary                                                 #
# ------------------------------------------------------------------ #


class TestExploreAgentSummary:
    def test_summary_returns_string(self, sample_repo: pathlib.Path) -> None:
        agent = _make_agent(str(sample_repo))
        s = agent.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_mentions_root(self, sample_repo: pathlib.Path) -> None:
        agent = _make_agent(str(sample_repo))
        s = agent.summary()
        assert str(sample_repo) in s


# ------------------------------------------------------------------ #
# ExploreAgent index caching                                           #
# ------------------------------------------------------------------ #


class TestExploreAgentCaching:
    def test_index_is_built_once(self, sample_repo: pathlib.Path) -> None:
        agent = _make_agent(str(sample_repo))
        agent.ask("Q1")
        agent.ask("Q2")
        # The index should be built only once across multiple questions.
        assert agent._index is not None
        initial_id = id(agent._index)
        agent.ask("Q3")
        assert id(agent._index) == initial_id

    def test_context_is_built_once(self, sample_repo: pathlib.Path) -> None:
        agent = _make_agent(str(sample_repo))
        agent.ask("Q1")
        ctx_id = id(agent._ctx)
        agent.ask("Q2")
        assert id(agent._ctx) == ctx_id


# ------------------------------------------------------------------ #
# ExploreAgent context isolation (main context untouched)             #
# ------------------------------------------------------------------ #


class TestExploreAgentContextIsolation:
    def test_agent_context_separate_from_caller(self, sample_repo: pathlib.Path) -> None:
        """Simulate a caller that has its own message list – agent must not touch it."""
        caller_messages: list[str] = ["existing message"]
        agent = _make_agent(str(sample_repo))
        agent.ask("Tell me about the code.")
        # Caller's own list is untouched.
        assert caller_messages == ["existing message"]

    def test_multiple_agents_independent(self, sample_repo: pathlib.Path) -> None:
        agents = [_make_agent(str(sample_repo), mock_answer=f"Answer {i}") for i in range(3)]
        for i, ag in enumerate(agents):
            ag.ask(f"Question {i}")
        # Each agent has its own context of length 2 (1 user + 1 assistant).
        for ag in agents:
            assert len(ag._ctx) == 2  # type: ignore[arg-type]
