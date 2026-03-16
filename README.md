# agents

## Codebase Explorer

> **Quick codebase analysis** – ask questions about your code without adding to your main context.

`codebase-explorer` is a lightweight Python tool that scans a directory tree and lets you ask natural-language questions about the code it contains.  All analysis runs in an **isolated context** that is completely separate from any ongoing conversation, so your primary context stays clean.

---

### Features

| Feature | Details |
|---|---|
| **Quick scan** | Walks a directory tree and indexes source files by extension |
| **Isolated context** | Each `ExploreAgent` owns its own `AnalysisContext`; results never leak into the caller's conversation |
| **Interactive REPL** | Multi-turn Q&A session directly in the terminal |
| **Single question** | One-shot CLI for scripting and automation |
| **Summary** | Human-readable overview of what was indexed |
| **Configurable** | Custom extensions, ignore-dirs, context budget, and model |

---

### Installation

```bash
pip install -e .
```

An [OpenAI](https://platform.openai.com/) API key is required for the LLM backend:

```bash
export OPENAI_API_KEY="sk-..."
```

---

### Usage

#### Command-line interface

```bash
# Ask a single question about the current directory:
explore ask . "What does this codebase do?"

# Ask about a specific repository:
explore ask /path/to/repo "Where is the authentication logic?"

# Start an interactive REPL (omit the question):
explore ask /path/to/repo

# Print a file-system summary (no LLM call needed):
explore summary /path/to/repo
```

#### Python API

```python
from codebase_explorer import ExploreAgent

agent = ExploreAgent("/path/to/repo")

# Ask questions – the analysis runs in a separate context
answer = agent.ask("What does this codebase do?")
print(answer)

# Follow-up questions build on the conversation history
# while still remaining isolated from your own context
answer2 = agent.ask("Which file should I look at first?")
print(answer2)

# Reset the analysis context when you're done
agent.reset()
```

---

### Architecture

```
codebase_explorer/
├── analyzer.py   – CodebaseAnalyzer: scans the file system → CodebaseIndex
├── context.py    – AnalysisContext: isolated, self-contained message history
├── agent.py      – ExploreAgent: orchestrates scanning + LLM completions
└── cli.py        – Click-based CLI with rich terminal output
```

The key design principle is **context isolation**: the `AnalysisContext` class owns its own message list and never mutates any external state.  Callers can therefore run as many analysis sessions as they like in parallel without any interference.

---

### Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run the test suite
pytest
```
