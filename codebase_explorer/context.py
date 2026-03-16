"""Context management helpers.

The core idea:  analysis happens in an *isolated* context that is separate
from the caller's ongoing conversation.  The :class:`AnalysisContext` class
owns its own message list and never mutates any external state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single chat-style message."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class AnalysisContext:
    """An isolated context used exclusively for codebase-analysis queries.

    Creating an :class:`AnalysisContext` never touches the caller's state.
    When the analysis is finished the context can simply be discarded – it
    will not leak into any external conversation history.
    """

    system_prompt: str
    _messages: list[Message] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Message management                                                   #
    # ------------------------------------------------------------------ #

    def add_user(self, content: str) -> None:
        self._messages.append(Message("user", content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(Message("assistant", content))

    @property
    def messages(self) -> list[Message]:
        """All messages in the context (read-only copy)."""
        return list(self._messages)

    def to_api_messages(self) -> list[dict[str, str]]:
        """Serialise to the OpenAI chat-completion format."""
        system = [{"role": "system", "content": self.system_prompt}]
        rest = [m.to_dict() for m in self._messages]
        return system + rest

    # ------------------------------------------------------------------ #
    # Snapshot / restore                                                   #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> list[Message]:
        """Return a deep copy of the current message list."""
        return deepcopy(self._messages)

    def restore(self, snapshot: list[Message]) -> None:
        """Replace the message list with *snapshot*."""
        self._messages = deepcopy(snapshot)

    def clear(self) -> None:
        self._messages.clear()

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"AnalysisContext(messages={len(self._messages)})"
