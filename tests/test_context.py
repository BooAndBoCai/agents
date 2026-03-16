"""Tests for codebase_explorer.context."""

from __future__ import annotations

import pytest

from codebase_explorer.context import AnalysisContext, Message


class TestMessage:
    def test_to_dict(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.to_dict() == {"role": "user", "content": "hello"}


class TestAnalysisContext:
    def test_initial_length_is_zero(self) -> None:
        ctx = AnalysisContext(system_prompt="You are helpful.")
        assert len(ctx) == 0

    def test_add_user_increments_length(self) -> None:
        ctx = AnalysisContext(system_prompt="sys")
        ctx.add_user("Hi")
        assert len(ctx) == 1

    def test_add_assistant_increments_length(self) -> None:
        ctx = AnalysisContext(system_prompt="sys")
        ctx.add_assistant("Hello")
        assert len(ctx) == 1

    def test_messages_are_ordered(self) -> None:
        ctx = AnalysisContext(system_prompt="sys")
        ctx.add_user("Q1")
        ctx.add_assistant("A1")
        ctx.add_user("Q2")
        roles = [m.role for m in ctx.messages]
        assert roles == ["user", "assistant", "user"]

    def test_to_api_messages_includes_system(self) -> None:
        ctx = AnalysisContext(system_prompt="BE HELPFUL")
        ctx.add_user("Hi")
        api_msgs = ctx.to_api_messages()
        assert api_msgs[0] == {"role": "system", "content": "BE HELPFUL"}

    def test_to_api_messages_length(self) -> None:
        ctx = AnalysisContext(system_prompt="sys")
        ctx.add_user("Q")
        ctx.add_assistant("A")
        assert len(ctx.to_api_messages()) == 3  # system + user + assistant

    def test_messages_property_returns_copy(self) -> None:
        ctx = AnalysisContext(system_prompt="sys")
        ctx.add_user("Q")
        msgs = ctx.messages
        msgs.clear()
        # Original context should be unaffected.
        assert len(ctx) == 1

    def test_snapshot_and_restore(self) -> None:
        ctx = AnalysisContext(system_prompt="sys")
        ctx.add_user("Q1")
        snap = ctx.snapshot()
        ctx.add_user("Q2")
        assert len(ctx) == 2
        ctx.restore(snap)
        assert len(ctx) == 1
        assert ctx.messages[0].content == "Q1"

    def test_snapshot_is_independent(self) -> None:
        ctx = AnalysisContext(system_prompt="sys")
        ctx.add_user("Q1")
        snap = ctx.snapshot()
        # Mutate the snapshot; context must be unaffected.
        snap[0].content = "MUTATED"
        assert ctx.messages[0].content == "Q1"

    def test_clear(self) -> None:
        ctx = AnalysisContext(system_prompt="sys")
        ctx.add_user("Q")
        ctx.clear()
        assert len(ctx) == 0

    def test_isolated_from_caller(self) -> None:
        """Each AnalysisContext instance is fully independent."""
        ctx_a = AnalysisContext(system_prompt="sys")
        ctx_b = AnalysisContext(system_prompt="sys")
        ctx_a.add_user("only in A")
        assert len(ctx_b) == 0
