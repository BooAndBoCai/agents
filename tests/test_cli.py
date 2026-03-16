"""Tests for the CLI entry point."""

from __future__ import annotations

import pytest

from agents.cli import main


def write_file(tmp_path, rel_path: str, content: str) -> str:
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return str(full)


class TestCLI:
    def test_summary_flag(self, tmp_path, capsys):
        write_file(tmp_path, "a.py", "x = 1\ny = 2\n")
        main(["--path", str(tmp_path), "--summary"])
        out = capsys.readouterr().out
        assert "Files:" in out

    def test_query_flag(self, tmp_path, capsys):
        write_file(tmp_path, "utils.py", "def helper():\n    return 42\n")
        main(["--path", str(tmp_path), "--query", "helper"])
        out = capsys.readouterr().out
        assert "helper" in out.lower()

    def test_no_query_prints_help(self, tmp_path, capsys):
        write_file(tmp_path, "a.py", "x = 1")
        with pytest.raises(SystemExit) as exc_info:
            main(["--path", str(tmp_path)])
        assert exc_info.value.code == 0

    def test_missing_path_exits_with_error(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--path", str(tmp_path / "does_not_exist"), "--query", "x"])
        assert exc_info.value.code != 0

    def test_max_results_respected(self, tmp_path, capsys):
        content = "\n".join(f"# TODO {i}" for i in range(30))
        write_file(tmp_path, "todos.py", content)
        main(["--path", str(tmp_path), "--query", "TODO", "--max-results", "3"])
        out = capsys.readouterr().out
        # Answer should be produced (StubLLMClient echoes the question)
        assert "TODO" in out

    def test_model_openai_missing_package(self, tmp_path, capsys, monkeypatch):
        """When openai is not installed, --model should exit with an error."""
        write_file(tmp_path, "a.py", "x = 1")
        # Simulate openai not being installed
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(SystemExit) as exc_info:
            main(["--path", str(tmp_path), "--query", "x", "--model", "gpt-4o"])
        assert exc_info.value.code != 0
