"""Tests for codebase_explorer.analyzer."""

from __future__ import annotations

import pathlib

import pytest

from codebase_explorer.analyzer import CodebaseAnalyzer, CodebaseIndex, SourceFile


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture()
def simple_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a minimal fake repository for scanning tests."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass\n")
    (tmp_path / "src" / "utils.py").write_text("def helper(): return 42\n")
    (tmp_path / "README.md").write_text("# My Project\n")
    # Files that should be ignored:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("// ignored\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "output.py").write_text("# ignored\n")
    # A file with an unsupported extension that should be skipped:
    (tmp_path / "binary.exe").write_text("MZ")
    return tmp_path


# ------------------------------------------------------------------ #
# CodebaseAnalyzer                                                     #
# ------------------------------------------------------------------ #


class TestCodebaseAnalyzer:
    def test_scan_returns_index(self, simple_repo: pathlib.Path) -> None:
        analyzer = CodebaseAnalyzer(simple_repo)
        index = analyzer.scan()
        assert isinstance(index, CodebaseIndex)

    def test_scan_finds_source_files(self, simple_repo: pathlib.Path) -> None:
        analyzer = CodebaseAnalyzer(simple_repo)
        index = analyzer.scan()
        paths = {f.relative_path for f in index.files}
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)
        assert any("README.md" in p for p in paths)

    def test_scan_ignores_node_modules(self, simple_repo: pathlib.Path) -> None:
        analyzer = CodebaseAnalyzer(simple_repo)
        index = analyzer.scan()
        paths = {f.relative_path for f in index.files}
        assert not any("node_modules" in p for p in paths)

    def test_scan_ignores_build_dir(self, simple_repo: pathlib.Path) -> None:
        analyzer = CodebaseAnalyzer(simple_repo)
        index = analyzer.scan()
        paths = {f.relative_path for f in index.files}
        assert not any("build" in p for p in paths)

    def test_scan_skips_unsupported_extensions(self, simple_repo: pathlib.Path) -> None:
        analyzer = CodebaseAnalyzer(simple_repo)
        index = analyzer.scan()
        extensions = {f.extension for f in index.files}
        assert ".exe" not in extensions

    def test_custom_extensions(self, simple_repo: pathlib.Path) -> None:
        analyzer = CodebaseAnalyzer(simple_repo, extensions=[".md"])
        index = analyzer.scan()
        assert all(f.extension == ".md" for f in index.files)

    def test_custom_ignore_dirs(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "mydir").mkdir()
        (tmp_path / "mydir" / "file.py").write_text("x = 1\n")
        (tmp_path / "other.py").write_text("y = 2\n")
        analyzer = CodebaseAnalyzer(tmp_path, ignore_dirs=["mydir"])
        index = analyzer.scan()
        paths = {f.relative_path for f in index.files}
        assert not any("mydir" in p for p in paths)
        assert any("other.py" in p for p in paths)


# ------------------------------------------------------------------ #
# CodebaseIndex                                                        #
# ------------------------------------------------------------------ #


class TestCodebaseIndex:
    def test_total_files(self, simple_repo: pathlib.Path) -> None:
        index = CodebaseAnalyzer(simple_repo).scan()
        assert index.total_files == len(index.files)

    def test_total_bytes(self, simple_repo: pathlib.Path) -> None:
        index = CodebaseAnalyzer(simple_repo).scan()
        assert index.total_bytes == sum(f.size_bytes for f in index.files)

    def test_by_extension(self, simple_repo: pathlib.Path) -> None:
        index = CodebaseAnalyzer(simple_repo).scan()
        by_ext = index.by_extension()
        assert ".py" in by_ext
        assert ".md" in by_ext

    def test_search_finds_file(self, simple_repo: pathlib.Path) -> None:
        index = CodebaseAnalyzer(simple_repo).scan()
        results = index.search("utils")
        assert len(results) == 1
        assert "utils.py" in results[0].relative_path

    def test_search_case_insensitive(self, simple_repo: pathlib.Path) -> None:
        index = CodebaseAnalyzer(simple_repo).scan()
        results_lower = index.search("main")
        results_upper = index.search("MAIN")
        assert len(results_lower) == len(results_upper)

    def test_summary_contains_root(self, simple_repo: pathlib.Path) -> None:
        index = CodebaseAnalyzer(simple_repo).scan()
        assert str(simple_repo) in index.summary()

    def test_summary_contains_extension_info(self, simple_repo: pathlib.Path) -> None:
        index = CodebaseAnalyzer(simple_repo).scan()
        summary = index.summary()
        assert ".py" in summary
        assert ".md" in summary


# ------------------------------------------------------------------ #
# SourceFile                                                           #
# ------------------------------------------------------------------ #


class TestSourceFile:
    def test_read_returns_content(self, tmp_path: pathlib.Path) -> None:
        fp = tmp_path / "hello.py"
        fp.write_text("print('hello')\n")
        sf = SourceFile(
            path=fp,
            relative_path="hello.py",
            extension=".py",
            size_bytes=fp.stat().st_size,
        )
        assert "print('hello')" in sf.read()

    def test_read_truncates_large_file(self, tmp_path: pathlib.Path) -> None:
        fp = tmp_path / "big.py"
        fp.write_text("x" * 200)
        sf = SourceFile(
            path=fp,
            relative_path="big.py",
            extension=".py",
            size_bytes=200,
        )
        content = sf.read(max_bytes=10)
        assert "truncated" in content

    def test_read_handles_missing_file(self, tmp_path: pathlib.Path) -> None:
        fp = tmp_path / "ghost.py"
        sf = SourceFile(
            path=fp,
            relative_path="ghost.py",
            extension=".py",
            size_bytes=0,
        )
        content = sf.read()
        assert "Could not read file" in content
