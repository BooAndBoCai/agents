# agents

**Quick codebase analysis** — ask questions about your code without adding the entire codebase to your main LLM context.

## Overview

`agents` provides a lightweight Python toolkit that:

1. **Scans** a directory tree and indexes all source files.
2. **Retrieves** relevant code snippets for a given query (no embeddings needed — fast substring search with context lines).
3. **Answers** your questions via an LLM using *only* the retrieved snippets, keeping token usage low and your main context clean.

The core design principle: the full source code is **never** forwarded to the LLM — only the small set of relevant snippets found by the search step.

## Installation

```bash
# Basic install (no external dependencies)
pip install .

# With OpenAI support
pip install ".[openai]"

# Development (includes pytest and ruff)
pip install -e ".[dev]"
```

## Quick start

### Python API

```python
from agents import CodebaseAnalyzer, ExploreAgent

# 1. Scan a codebase
analyzer = CodebaseAnalyzer().scan("/path/to/my/project")
print(analyzer.summary())

# 2. Ask a question (uses the built-in stub LLM — no API key required)
agent = ExploreAgent(analyzer)
result = agent.ask("Where is the authentication logic?")
print(result)
```

**With the OpenAI API:**

```python
from openai import OpenAI
from agents import CodebaseAnalyzer, ExploreAgent
from agents.explore_agent import OpenAILLMClient

analyzer = CodebaseAnalyzer().scan("/path/to/my/project")
llm = OpenAILLMClient(OpenAI(), model="gpt-4o")
agent = ExploreAgent(analyzer, llm_client=llm)

result = agent.ask("How are database migrations handled?")
print(result.answer)
print("Sources:", [s.path for s in result.sources])
```

**Bring your own LLM:**

Any callable that accepts `list[dict]` messages and returns a `str` will work:

```python
class MyLLM:
    def complete(self, messages: list[dict]) -> str:
        # call your own backend
        ...

agent = ExploreAgent(analyzer, llm_client=MyLLM())
```

### CLI

```bash
# Ask a question about the current directory
codebase-analyze --query "Where is error handling?"

# Specify a project path
codebase-analyze --path /my/project --query "How does login work?"

# Print a codebase summary
codebase-analyze --path /my/project --summary

# Use OpenAI (requires OPENAI_API_KEY env var)
codebase-analyze --path /my/project --query "What does the CI pipeline do?" --model gpt-4o

# Restrict search to specific file types
codebase-analyze --path /my/project --query "TODO" --extensions .py .js

# Limit retrieved snippets
codebase-analyze --path /my/project --query "authenticate" --max-results 5
```

## Architecture

```
agents/
├── codebase_analyzer.py   # Scans directories, indexes files, text search
├── explore_agent.py       # Retrieves snippets, calls LLM, returns answers
└── cli.py                 # Command-line interface
```

| Class | Responsibility |
|---|---|
| `CodebaseAnalyzer` | Walk a directory tree, skip excluded paths, read files, search by keyword |
| `ExploreAgent` | Build a context-limited prompt from search results, call the LLM |
| `ExploreResult` | Holds the question, answer, and source snippets |
| `StubLLMClient` | Built-in no-dependency stub (useful for tests / offline use) |
| `OpenAILLMClient` | Thin wrapper around the OpenAI Chat Completions API |

## Running tests

```bash
pytest
```

## License

MIT
