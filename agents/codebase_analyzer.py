"""Codebase scanner and indexer.

Scans a directory tree, reads source files, and provides a simple
keyword/substring search so the ExploreAgent can retrieve relevant
snippets without loading the entire codebase into context.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

# File extensions treated as source code by default.
DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rb",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".swift",
    ".kt",
    ".sh",
    ".bash",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
)

# Directories that are almost never interesting source code.
DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    ".idea",
    ".vscode",
)


@dataclass
class IndexedFile:
    """A single file that has been read from disk."""

    path: str
    """Absolute (or relative-to-root) path of the file."""

    content: str
    """Full text content of the file."""

    @property
    def lines(self) -> list[str]:
        return self.content.splitlines()

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", "\\n")
        return f"IndexedFile(path={self.path!r}, chars={len(self.content)}, preview={preview!r})"


@dataclass
class SearchResult:
    """A snippet found within an indexed file."""

    file: IndexedFile
    """The file this result came from."""

    line_number: int
    """1-based line number of the first matching line."""

    snippet: str
    """A short excerpt around the match (a few lines of context)."""

    @property
    def path(self) -> str:
        return self.file.path


@dataclass
class CodebaseAnalyzer:
    """Scans a directory and indexes source files for quick text search.

    Usage::

        analyzer = CodebaseAnalyzer()
        analyzer.scan("/path/to/project")
        results = analyzer.search("def authenticate")
    """

    extensions: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EXTENSIONS)
    exclude_dirs: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EXCLUDE_DIRS)
    max_file_size_bytes: int = 1_000_000  # 1 MB – skip very large files
    context_lines: int = 3  # lines of context around each match

    # Populated after scan()
    _files: list[IndexedFile] = field(default_factory=list, init=False, repr=False)
    _root: Optional[str] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, root: str) -> "CodebaseAnalyzer":
        """Walk *root* recursively and index all matching source files.

        Returns *self* so calls can be chained::

            results = CodebaseAnalyzer().scan("/my/project").search("TODO")

        Parameters
        ----------
        root:
            The directory (or file) to scan.
        """
        root = os.path.abspath(root)
        self._root = root
        self._files = []

        if os.path.isfile(root):
            indexed = self._read_file(root)
            if indexed is not None:
                self._files.append(indexed)
        else:
            for dirpath, dirnames, filenames in os.walk(root):
                # Prune excluded directories in-place so os.walk skips them.
                dirnames[:] = [
                    d for d in dirnames if not self._is_excluded_dir(d)
                ]
                for filename in filenames:
                    if self._has_source_extension(filename):
                        full_path = os.path.join(dirpath, filename)
                        indexed = self._read_file(full_path)
                        if indexed is not None:
                            self._files.append(indexed)

        return self

    def search(
        self,
        query: str,
        *,
        case_sensitive: bool = False,
        max_results: int = 20,
        extensions: Optional[Sequence[str]] = None,
    ) -> list[SearchResult]:
        """Search indexed files for *query* and return matching snippets.

        Parameters
        ----------
        query:
            Plain-text substring to look for.
        case_sensitive:
            When *False* (default) the search is case-insensitive.
        max_results:
            Maximum number of :class:`SearchResult` objects to return.
        extensions:
            Optional list of extensions to restrict the search to
            (e.g. ``[".py", ".js"]``).
        """
        if not self._files:
            return []

        needle = query if case_sensitive else query.lower()
        results: list[SearchResult] = []

        for indexed_file in self._files:
            if extensions and not any(
                indexed_file.path.endswith(ext) for ext in extensions
            ):
                continue

            lines = indexed_file.lines
            for i, line in enumerate(lines):
                haystack = line if case_sensitive else line.lower()
                if needle in haystack:
                    snippet = self._build_snippet(lines, i)
                    results.append(
                        SearchResult(
                            file=indexed_file,
                            line_number=i + 1,
                            snippet=snippet,
                        )
                    )
                    if len(results) >= max_results:
                        return results

        return results

    def get_file(self, path: str) -> Optional[IndexedFile]:
        """Return the :class:`IndexedFile` for *path* (exact or suffix match)."""
        for f in self._files:
            if f.path == path or f.path.endswith(path):
                return f
        return None

    @property
    def files(self) -> list[IndexedFile]:
        """All indexed files (read-only list)."""
        return list(self._files)

    @property
    def file_count(self) -> int:
        """Number of indexed files."""
        return len(self._files)

    @property
    def root(self) -> Optional[str]:
        """The root directory that was scanned, or *None* if not yet scanned."""
        return self._root

    def summary(self) -> str:
        """Return a human-readable summary of the indexed codebase."""
        if not self._files:
            return "No files indexed."
        total_lines = sum(len(f.lines) for f in self._files)
        total_chars = sum(len(f.content) for f in self._files)
        ext_counts: dict[str, int] = {}
        for f in self._files:
            _, ext = os.path.splitext(f.path)
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        top_exts = sorted(ext_counts.items(), key=lambda x: -x[1])[:5]
        ext_str = ", ".join(f"{ext or 'no-ext'}({n})" for ext, n in top_exts)
        return (
            f"Root: {self._root}\n"
            f"Files: {self.file_count}\n"
            f"Lines: {total_lines:,}\n"
            f"Characters: {total_chars:,}\n"
            f"Top extensions: {ext_str}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_excluded_dir(self, name: str) -> bool:
        return name in self.exclude_dirs

    def _has_source_extension(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() in {e.lower() for e in self.extensions}

    def _read_file(self, path: str) -> Optional[IndexedFile]:
        """Read *path* and return an :class:`IndexedFile`, or *None* on error."""
        try:
            if os.path.getsize(path) > self.max_file_size_bytes:
                return None
            with open(path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            if self._root and os.path.isdir(self._root):
                rel_path = os.path.relpath(path, self._root)
            else:
                rel_path = os.path.basename(path)
            return IndexedFile(path=rel_path, content=content)
        except OSError:
            return None

    def _build_snippet(self, lines: list[str], match_index: int) -> str:
        start = max(0, match_index - self.context_lines)
        end = min(len(lines), match_index + self.context_lines + 1)
        numbered = [
            f"{i + 1:>4}: {lines[i]}" for i in range(start, end)
        ]
        return "\n".join(numbered)
