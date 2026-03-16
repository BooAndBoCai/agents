"""Tests for ExploreAgent."""

from __future__ import annotations

from agents.codebase_analyzer import CodebaseAnalyzer
from agents.explore_agent import (
    ExploreAgent,
    ExploreResult,
    OpenAILLMClient,
    StubLLMClient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_file(tmp_path, rel_path: str, content: str) -> str:
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return str(full)


class CapturingLLMClient:
    """Records the messages it receives and returns a fixed reply."""

    def __init__(self, reply: str = "test answer"):
        self.calls: list[list[dict]] = []
        self.reply = reply

    def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply


# ---------------------------------------------------------------------------
# StubLLMClient
# ---------------------------------------------------------------------------


class TestStubLLMClient:
    def test_complete_returns_string(self):
        client = StubLLMClient()
        result = client.complete([{"role": "user", "content": "hello"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_complete_echoes_question(self):
        client = StubLLMClient()
        result = client.complete([{"role": "user", "content": "What is foo?"}])
        assert "What is foo?" in result


# ---------------------------------------------------------------------------
# ExploreAgent
# ---------------------------------------------------------------------------


class TestExploreAgent:
    def test_ask_returns_explore_result(self, tmp_path):
        write_file(tmp_path, "auth.py", "def login(user, pw):\n    pass\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        agent = ExploreAgent(analyzer)
        result = agent.ask("How does login work?")
        assert isinstance(result, ExploreResult)

    def test_ask_sets_question(self, tmp_path):
        write_file(tmp_path, "a.py", "x = 1")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        agent = ExploreAgent(analyzer)
        result = agent.ask("What is x?")
        assert result.question == "What is x?"

    def test_ask_passes_context_to_llm(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper():\n    return 42\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        llm = CapturingLLMClient(reply="It returns 42.")
        agent = ExploreAgent(analyzer, llm_client=llm)
        result = agent.ask("helper")
        assert len(llm.calls) == 1
        messages = llm.calls[0]
        # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # The code snippet should be in the user message
        assert "helper" in messages[1]["content"]
        assert result.answer == "It returns 42."

    def test_ask_no_results_message(self, tmp_path):
        write_file(tmp_path, "a.py", "x = 1")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        llm = CapturingLLMClient()
        agent = ExploreAgent(analyzer, llm_client=llm)
        agent.ask("ZZZ_UNLIKELY_TOKEN_XYZ")
        user_content = llm.calls[0][1]["content"]
        assert "No relevant snippets" in user_content

    def test_ask_sources_populated(self, tmp_path):
        write_file(tmp_path, "db.py", "def connect():\n    pass\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        agent = ExploreAgent(analyzer)
        result = agent.ask("connect")
        assert len(result.sources) >= 1
        assert any("db.py" in s.path for s in result.sources)

    def test_ask_respects_max_results(self, tmp_path):
        content = "\n".join(f"# TODO item {i}" for i in range(30))
        write_file(tmp_path, "todos.py", content)
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        llm = CapturingLLMClient()
        agent = ExploreAgent(analyzer, llm_client=llm, max_search_results=3)
        result = agent.ask("TODO")
        assert len(result.sources) == 3

    def test_ask_extension_filter(self, tmp_path):
        write_file(tmp_path, "code.py", "def foo(): pass")
        write_file(tmp_path, "notes.md", "Call foo")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        llm = CapturingLLMClient()
        agent = ExploreAgent(analyzer, llm_client=llm)
        result = agent.ask("foo", extensions=[".py"])
        assert all(s.path.endswith(".py") for s in result.sources)

    def test_explore_result_str(self, tmp_path):
        write_file(tmp_path, "main.py", "def run():\n    pass\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        agent = ExploreAgent(analyzer, llm_client=CapturingLLMClient("It runs."))
        result = agent.ask("run")
        text = str(result)
        assert "Q:" in text
        assert "A:" in text
        assert "run" in text

    def test_explore_result_str_no_sources(self):
        result = ExploreResult(question="q", answer="a", sources=[])
        text = str(result)
        assert "Sources:" not in text

    def test_default_llm_is_stub(self, tmp_path):
        write_file(tmp_path, "a.py", "x = 1")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        agent = ExploreAgent(analyzer)
        assert isinstance(agent.llm_client, StubLLMClient)

    def test_system_prompt_customizable(self, tmp_path):
        write_file(tmp_path, "a.py", "x = 1")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        llm = CapturingLLMClient()
        custom_prompt = "You are a custom assistant."
        agent = ExploreAgent(analyzer, llm_client=llm, system_prompt=custom_prompt)
        agent.ask("x")
        assert llm.calls[0][0]["content"] == custom_prompt


# ---------------------------------------------------------------------------
# OpenAILLMClient (offline – verifies it wraps the client correctly)
# ---------------------------------------------------------------------------


class TestOpenAILLMClient:
    def test_calls_create_with_correct_args(self):
        """Verify OpenAILLMClient forwards messages to the openai client."""

        class FakeChoice:
            message = type("M", (), {"content": "mocked"})()

        class FakeCompletion:
            choices = [FakeChoice()]

        class FakeChat:
            def create(self, **kwargs):
                self.last_kwargs = kwargs
                return FakeCompletion()

        class FakeOpenAI:
            chat = property(lambda self: self._chat)

            def __init__(self):
                self._chat = type("C", (), {"completions": FakeChat()})()

        client = OpenAILLMClient(FakeOpenAI(), model="gpt-4o", temperature=0.0)
        messages = [{"role": "user", "content": "hi"}]
        result = client.complete(messages)
        assert result == "mocked"
