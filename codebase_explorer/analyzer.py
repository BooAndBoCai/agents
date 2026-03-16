"""Scans a directory tree and builds a lightweight index of source files."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Generator, Sequence

# File extensions considered "source code" by default.
_DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
        ".java", ".kt", ".c", ".cpp", ".h", ".hpp", ".cs",
        ".rb", ".php", ".swift", ".scala", ".sh", ".bash",
        ".yaml", ".yml", ".toml", ".json", ".md", ".txt",
        ".html", ".css", ".scss", ".sql",
    }
)

# Directories that are virtually never useful to index.
_DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn",
        "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
        ".tox", ".venv", "venv", "env", ".env",
        "dist", "build", "target", "out", "bin", "obj",
        ".idea", ".vscode",
    }
)


@dataclass
class SourceFile:
    """A single source file discovered during a scan."""

    path: pathlib.Path
    relative_path: str
    extension: str
    size_bytes: int

    def read(self, max_bytes: int = 100_000) -> str:
        """Return the file contents, truncated to *max_bytes* if necessary."""
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(max_bytes)
            if self.size_bytes > max_bytes:
                content += f"\n\n[... truncated – {self.size_bytes} bytes total ...]"
            return content
        except OSError as exc:
            return f"[Could not read file: {exc}]"


@dataclass
class CodebaseIndex:
    """The result of scanning a directory: a collection of :class:`SourceFile` objects."""

    root: pathlib.Path
    files: list[SourceFile] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    def by_extension(self) -> dict[str, list[SourceFile]]:
        """Group files by their extension."""
        result: dict[str, list[SourceFile]] = {}
        for sf in self.files:
            result.setdefault(sf.extension, []).append(sf)
        return result

    def search(self, pattern: str, *, case_sensitive: bool = False) -> list[SourceFile]:
        """Return files whose *relative_path* contains *pattern*."""
        cmp = pattern if case_sensitive else pattern.lower()
        return [
            sf for sf in self.files
            if cmp in (sf.relative_path if case_sensitive else sf.relative_path.lower())
        ]

    def summary(self) -> str:
        """Return a short human-readable summary of the index."""
        by_ext = self.by_extension()
        lines = [
            f"Root : {self.root}",
            f"Files: {self.total_files}",
            f"Size : {self.total_bytes / 1024:.1f} KB",
            "",
            "Extensions:",
        ]
        for ext, files in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"  {ext if ext else '(no ext)':12s}  {len(files):4d} file(s)")
        return "\n".join(lines)


class CodebaseAnalyzer:
    """Walks a directory tree and produces a :class:`CodebaseIndex`.

    Parameters
    ----------
    root:
        The directory to scan.
    extensions:
        A set of file extensions (with leading dot) to include.  Pass
        ``None`` to use the built-in defaults.
    ignore_dirs:
        Directory names to skip entirely.  Pass ``None`` for defaults.
    max_file_bytes:
        Files larger than this are still indexed but truncated when read.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        extensions: Sequence[str] | None = None,
        ignore_dirs: Sequence[str] | None = None,
        max_file_bytes: int = 100_000,
    ) -> None:
        self.root = pathlib.Path(root).resolve()
        self.extensions: frozenset[str] = (
            frozenset(extensions) if extensions is not None else _DEFAULT_EXTENSIONS
        )
        self.ignore_dirs: frozenset[str] = (
            frozenset(ignore_dirs) if ignore_dirs is not None else _DEFAULT_IGNORE_DIRS
        )
        self.max_file_bytes = max_file_bytes

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def scan(self) -> CodebaseIndex:
        """Scan :attr:`root` and return a :class:`CodebaseIndex`."""
        index = CodebaseIndex(root=self.root)
        for sf in self._walk():
            index.files.append(sf)
        return index

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _walk(self) -> Generator[SourceFile, None, None]:
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune ignored directories in-place so os.walk won't descend.
            dirnames[:] = [
                d for d in dirnames if d not in self.ignore_dirs
            ]
            dp = pathlib.Path(dirpath)
            for filename in filenames:
                fp = dp / filename
                ext = fp.suffix.lower()
                if ext not in self.extensions:
                    continue
                try:
                    size = fp.stat().st_size
                except OSError:
                    continue
                yield SourceFile(
                    path=fp,
                    relative_path=str(fp.relative_to(self.root)),
                    extension=ext,
                    size_bytes=size,
                )
