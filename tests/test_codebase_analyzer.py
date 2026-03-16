"""Tests for CodebaseAnalyzer."""

from __future__ import annotations

from agents.codebase_analyzer import CodebaseAnalyzer, IndexedFile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_file(tmp_path, rel_path: str, content: str) -> str:
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return str(full)


# ---------------------------------------------------------------------------
# IndexedFile
# ---------------------------------------------------------------------------


class TestIndexedFile:
    def test_lines(self):
        f = IndexedFile(path="foo.py", content="a\nb\nc")
        assert f.lines == ["a", "b", "c"]

    def test_repr(self):
        f = IndexedFile(path="foo.py", content="hello world")
        assert "foo.py" in repr(f)

    def test_empty_content(self):
        f = IndexedFile(path="empty.py", content="")
        assert f.lines == []


# ---------------------------------------------------------------------------
# CodebaseAnalyzer.scan
# ---------------------------------------------------------------------------


class TestScan:
    def test_scan_single_file(self, tmp_path):
        write_file(tmp_path, "hello.py", "print('hello')")
        analyzer = CodebaseAnalyzer()
        analyzer.scan(str(tmp_path / "hello.py"))
        assert analyzer.file_count == 1
        assert analyzer.files[0].path.endswith("hello.py")
        assert "print" in analyzer.files[0].content

    def test_scan_directory(self, tmp_path):
        write_file(tmp_path, "a.py", "x = 1")
        write_file(tmp_path, "sub/b.py", "y = 2")
        analyzer = CodebaseAnalyzer()
        analyzer.scan(str(tmp_path))
        paths = [f.path for f in analyzer.files]
        assert any(p.endswith("a.py") for p in paths)
        assert any(p.endswith("b.py") for p in paths)

    def test_scan_excludes_unknown_extensions(self, tmp_path):
        write_file(tmp_path, "binary.exe", "\x00\x01\x02")
        analyzer = CodebaseAnalyzer()
        analyzer.scan(str(tmp_path))
        assert analyzer.file_count == 0

    def test_scan_excludes_node_modules(self, tmp_path):
        write_file(tmp_path, "src/index.js", "const x = 1;")
        write_file(tmp_path, "node_modules/lib/index.js", "// lib")
        analyzer = CodebaseAnalyzer()
        analyzer.scan(str(tmp_path))
        paths = [f.path for f in analyzer.files]
        assert any("src" in p for p in paths)
        assert not any("node_modules" in p for p in paths)

    def test_scan_returns_self(self, tmp_path):
        write_file(tmp_path, "a.py", "x = 1")
        analyzer = CodebaseAnalyzer()
        result = analyzer.scan(str(tmp_path))
        assert result is analyzer

    def test_scan_chaining(self, tmp_path):
        write_file(tmp_path, "a.py", "def foo(): pass")
        results = CodebaseAnalyzer().scan(str(tmp_path)).search("foo")
        assert len(results) >= 1

    def test_scan_missing_path_skips_gracefully(self, tmp_path):
        analyzer = CodebaseAnalyzer()
        # Non-existent path – should index 0 files and not raise
        analyzer.scan(str(tmp_path / "does_not_exist"))
        assert analyzer.file_count == 0

    def test_scan_root_property(self, tmp_path):
        write_file(tmp_path, "x.py", "pass")
        analyzer = CodebaseAnalyzer()
        analyzer.scan(str(tmp_path))
        assert analyzer.root == str(tmp_path)

    def test_scan_skips_large_files(self, tmp_path):
        big_file = tmp_path / "big.py"
        big_file.write_bytes(b"x" * 2_000_000)
        analyzer = CodebaseAnalyzer(max_file_size_bytes=1_000_000)
        analyzer.scan(str(tmp_path))
        assert analyzer.file_count == 0


# ---------------------------------------------------------------------------
# CodebaseAnalyzer.search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_basic_search(self, tmp_path):
        write_file(tmp_path, "auth.py", "def authenticate(user, password):\n    pass\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        results = analyzer.search("authenticate")
        assert len(results) == 1
        assert "authenticate" in results[0].snippet

    def test_case_insensitive_by_default(self, tmp_path):
        write_file(tmp_path, "a.py", "class MyClass:\n    pass\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        assert len(analyzer.search("myclass")) == 1
        assert len(analyzer.search("MYCLASS")) == 1

    def test_case_sensitive(self, tmp_path):
        write_file(tmp_path, "a.py", "class MyClass:\n    pass\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        assert len(analyzer.search("MyClass", case_sensitive=True)) == 1
        assert len(analyzer.search("myclass", case_sensitive=True)) == 0

    def test_no_results(self, tmp_path):
        write_file(tmp_path, "a.py", "x = 1\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        assert analyzer.search("nonexistent_token_xyz") == []

    def test_max_results(self, tmp_path):
        # Create a file with many occurrences of the same token
        content = "\n".join(f"x = {i}  # TODO" for i in range(50))
        write_file(tmp_path, "todos.py", content)
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        results = analyzer.search("TODO", max_results=5)
        assert len(results) == 5

    def test_search_extension_filter(self, tmp_path):
        write_file(tmp_path, "code.py", "def foo(): pass")
        write_file(tmp_path, "notes.md", "Call foo here")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        # Should only return the .py result when filtering to .py
        results = analyzer.search("foo", extensions=[".py"])
        assert len(results) >= 1
        assert all(r.path.endswith(".py") for r in results)
        # .md files also contain "foo", so filtering to .md yields .md only
        results_md = analyzer.search("foo", extensions=[".md"])
        assert all(r.path.endswith(".md") for r in results_md)
        # No .ts files → empty result
        results_ts = analyzer.search("foo", extensions=[".ts"])
        assert len(results_ts) == 0

    def test_search_on_empty_index(self):
        analyzer = CodebaseAnalyzer()
        assert analyzer.search("anything") == []

    def test_search_result_line_number(self, tmp_path):
        write_file(tmp_path, "a.py", "line1\nline2\nTARGET\nline4\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        results = analyzer.search("TARGET")
        assert len(results) == 1
        assert results[0].line_number == 3

    def test_snippet_contains_context(self, tmp_path):
        write_file(tmp_path, "a.py", "line1\nline2\nTARGET\nline4\nline5\n")
        analyzer = CodebaseAnalyzer(context_lines=1).scan(str(tmp_path))
        results = analyzer.search("TARGET")
        assert "line2" in results[0].snippet
        assert "line4" in results[0].snippet


# ---------------------------------------------------------------------------
# CodebaseAnalyzer helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_get_file_exact(self, tmp_path):
        write_file(tmp_path, "foo.py", "x = 1")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        f = analyzer.get_file("foo.py")
        assert f is not None
        assert "x = 1" in f.content

    def test_get_file_not_found(self, tmp_path):
        write_file(tmp_path, "foo.py", "x = 1")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        assert analyzer.get_file("bar.py") is None

    def test_summary(self, tmp_path):
        write_file(tmp_path, "a.py", "x = 1\ny = 2\n")
        analyzer = CodebaseAnalyzer().scan(str(tmp_path))
        s = analyzer.summary()
        assert "Files:" in s
        assert "Lines:" in s
        assert ".py" in s

    def test_summary_empty(self):
        assert CodebaseAnalyzer().summary() == "No files indexed."
