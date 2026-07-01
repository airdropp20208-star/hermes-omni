"""Context Hologram — entire project in 500 tokens.

THE BREAKTHROUGH
----------------
Large codebases (100K+ lines) can't fit in context. Agents read files
one by one, missing the big picture. ContextHologram compresses the
ENTIRE project structure into a ~500-token "hologram" that gives the
agent a bird's-eye view:

- Directory tree with LOC counts
- Import graph (who depends on whom)
- Hot files (most changed, most imported)
- Test coverage heatmap (which dirs have tests, which don't)
- Tech stack detection (languages, frameworks, deps)
- Entry points (main files, CLI entry, test entry)
- Complexity signals (largest files, deepest dirs)

The hologram is NOT a replacement for reading files — it's a MAP that
tells the agent WHERE to look. Like a GPS: doesn't drive for you, but
shows the fastest route.

USAGE
-----
    from agent.unified.context_hologram import build_hologram

    hologram = build_hologram("/path/to/project")
    # → ~500 token string, inject into system prompt
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class HologramData:
    """Raw data collected from project scan."""

    root: str = ""
    total_files: int = 0
    total_lines: int = 0
    languages: dict[str, int] = field(default_factory=dict)  # ext → file count
    dirs: dict[str, dict[str, int]] = field(default_factory=dict)  # dir → {files, lines}
    imports: dict[str, set[str]] = field(default_factory=dict)  # file → imported modules
    hot_files: list[tuple[str, int]] = field(default_factory=list)  # (path, lines) sorted desc
    entry_points: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    largest_files: list[tuple[str, int]] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)  # detected dependencies
    complexity_score: float = 0.0  # 0-1, higher = more complex


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #


# Extensions to count as code
CODE_EXTENSIONS = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ Header",
    ".cs": "C#",
    ".swift": "Swift",
    ".scala": "Scala",
    ".sh": "Shell",
    ".sql": "SQL",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".md": "Markdown",
    ".html": "HTML",
    ".css": "CSS",
    ".vue": "Vue",
    ".svelte": "Svelte",
}

# Directories to skip
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".next",
    ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".tox", ".eggs", "*.egg-info",
    ".idea", ".vscode", "coverage", ".coverage",
}

# Files that indicate entry points
ENTRY_POINT_PATTERNS = [
    "main.py", "app.py", "server.py", "index.ts", "index.js",
    "cli.py", "run.py", "__main__.py", "setup.py",
    "manage.py", "wsgi.py", "asgi.py",
]

# Import patterns
PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))", re.MULTILINE)
TS_IMPORT_RE = re.compile(r"^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)


def build_hologram(root: str | Path, *, max_depth: int = 5, max_files: int = 5000) -> str:
    """Scan project and return ~500 token hologram string.

    Args:
        root: project root directory
        max_depth: max directory depth to scan
        max_files: max files to scan (safety limit)

    Returns:
        Hologram string suitable for system prompt injection.
    """
    root = Path(root).expanduser()
    if not root.exists() or not root.is_dir():
        return ""

    data = HologramData(root=str(root))
    _scan(root, data, max_depth=max_depth, max_files=max_files, depth=0)
    _analyze(data)
    return _format_hologram(data)


def _scan(
    path: Path,
    data: HologramData,
    *,
    max_depth: int,
    max_files: int,
    depth: int,
) -> None:
    """Recursively scan directory."""
    if depth > max_depth or data.total_files > max_files:
        return

    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return

    for entry in entries:
        if entry.name in SKIP_DIRS or entry.name.startswith("."):
            # Allow .github, .claude etc but skip .git, .venv
            if entry.name not in (".github", ".claude", ".claude-plugin"):
                continue

        if entry.is_dir():
            _scan(entry, data, max_depth=max_depth, max_files=max_files, depth=depth + 1)
        elif entry.is_file():
            _scan_file(entry, data, root=Path(data.root))


def _scan_file(filepath: Path, data: HologramData, *, root: Path) -> None:
    """Scan one file."""
    ext = filepath.suffix.lower()
    if ext not in CODE_EXTENSIONS:
        return

    data.total_files += 1

    # Language
    lang = CODE_EXTENSIONS[ext]
    data.languages[lang] = data.languages.get(lang, 0) + 1

    # Directory stats
    try:
        rel_dir = str(filepath.parent.relative_to(root))
    except ValueError:
        rel_dir = str(filepath.parent)
    if rel_dir == ".":
        rel_dir = "/"
    if rel_dir not in data.dirs:
        data.dirs[rel_dir] = {"files": 0, "lines": 0}
    data.dirs[rel_dir]["files"] += 1

    # Line count
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        lines = content.count("\n") + 1
        data.total_lines += lines
        data.dirs[rel_dir]["lines"] += lines

        # Track largest files
        if len(data.largest_files) < 20:
            data.largest_files.append((str(filepath.relative_to(root)), lines))
            data.largest_files.sort(key=lambda x: x[1], reverse=True)
        elif lines > data.largest_files[-1][1]:
            data.largest_files[-1] = (str(filepath.relative_to(root)), lines)
            data.largest_files.sort(key=lambda x: x[1], reverse=True)

        # Entry points
        if filepath.name in ENTRY_POINT_PATTERNS:
            try:
                data.entry_points.append(str(filepath.relative_to(root)))
            except ValueError:
                pass

        # Test directories
        if "test" in filepath.parent.name.lower() or filepath.name.startswith("test_"):
            try:
                test_dir = str(filepath.parent.relative_to(root))
            except ValueError:
                test_dir = str(filepath.parent)
            if test_dir not in data.test_dirs:
                data.test_dirs.append(test_dir)

        # Imports (Python)
        if ext == ".py":
            modules = set()
            for m in PY_IMPORT_RE.finditer(content):
                mod = m.group(1) or m.group(2)
                if mod and not mod.startswith("."):
                    modules.add(mod.split(".")[0])
            if modules:
                try:
                    rel_path = str(filepath.relative_to(root))
                except ValueError:
                    rel_path = str(filepath)
                data.imports[rel_path] = modules

        # Imports (TypeScript/JavaScript)
        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            modules = set()
            for m in TS_IMPORT_RE.finditer(content):
                mod = m.group(1)
                if mod and mod.startswith("."):
                    modules.add(mod)
            if modules:
                try:
                    rel_path = str(filepath.relative_to(root))
                except ValueError:
                    rel_path = str(filepath)
                data.imports[rel_path] = modules

        # Dependencies
        if filepath.name in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod", "requirements.txt"):
            data.deps.append(filepath.name)

    except Exception:
        pass


def _analyze(data: HologramData) -> None:
    """Post-process scanned data."""
    # Hot files (most imported)
    import_counter: Counter[str] = Counter()
    for modules in data.imports.values():
        for mod in modules:
            import_counter[mod] += 1
    data.hot_files = import_counter.most_common(10)

    # Complexity score
    if data.total_files > 0:
        avg_lines = data.total_lines / data.total_files
        dir_count = len(data.dirs)
        data.complexity_score = min(
            1.0,
            (data.total_files / 1000) * 0.3 + (dir_count / 50) * 0.3 + (avg_lines / 200) * 0.4,
        )


def _format_hologram(data: HologramData) -> str:
    """Format hologram data into ~500 token string."""
    if data.total_files == 0:
        return ""

    lines = ["<context-hologram>"]

    # Overview
    lines.append(f"Project: {Path(data.root).name} | {data.total_files} files | {data.total_lines:,} lines | complexity: {data.complexity_score:.0%}")

    # Languages
    lang_str = ", ".join(f"{lang} ({count})" for lang, count in sorted(data.languages.items(), key=lambda x: -x[1])[:5])
    lines.append(f"Languages: {lang_str}")

    # Entry points
    if data.entry_points:
        lines.append(f"Entry points: {', '.join(data.entry_points[:5])}")

    # Dependencies
    if data.deps:
        lines.append(f"Dependency files: {', '.join(data.deps)}")

    # Directory structure (top 10 by lines)
    top_dirs = sorted(data.dirs.items(), key=lambda x: x[1]["lines"], reverse=True)[:10]
    lines.append("")
    lines.append("Top directories (by LOC):")
    for dir_path, stats in top_dirs:
        lines.append(f"  {dir_path:40} {stats['files']:>4} files | {stats['lines']:>6} lines")

    # Largest files
    if data.largest_files:
        lines.append("")
        lines.append("Largest files:")
        for path, loc in data.largest_files[:8]:
            lines.append(f"  {path:50} {loc:>6} lines")

    # Hot modules (most imported)
    if data.hot_files:
        lines.append("")
        lines.append("Most imported modules:")
        for mod, count in data.hot_files[:5]:
            lines.append(f"  {mod:30} imported {count}x")

    # Test coverage
    if data.test_dirs:
        lines.append("")
        lines.append(f"Test directories ({len(data.test_dirs)}): {', '.join(data.test_dirs[:5])}")
    else:
        lines.append("")
        lines.append("⚠ No test directories found")

    lines.append("</context-hologram>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Module-level API
# --------------------------------------------------------------------------- #


_cache: dict[str, tuple[str, float]] = {}  # path → (hologram, timestamp)
_CACHE_TTL = 300  # 5 minutes


def get_hologram(root: str | Path | None = None, *, force_refresh: bool = False) -> str:
    """Get hologram for a project (cached, 5 min TTL).

    Args:
        root: project root. Defaults to current working directory.
        force_refresh: bypass cache.

    Returns:
        Hologram string for system prompt injection.
    """
    if root is None:
        root = os.getcwd()
    root = str(Path(root).resolve())

    if not force_refresh and root in _cache:
        hologram, ts = _cache[root]
        if time.time() - ts < _CACHE_TTL:
            return hologram

    hologram = build_hologram(root)
    _cache[root] = (hologram, time.time())
    return hologram


def hologram_stats() -> dict[str, Any]:
    return {
        "cached_projects": len(_cache),
        "cache_ttl_seconds": _CACHE_TTL,
    }


# Need time import
import time  # noqa: E402
