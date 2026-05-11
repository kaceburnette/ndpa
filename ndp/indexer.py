"""
Project file indexer.

Walks the project tree and creates/updates NDP objects in the store.
Estimates token counts. Parses summaries from file headers.
"""

import os
from pathlib import Path

from .schema import NDPObject
from .store import NDPStore

INCLUDE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".py", ".go", ".rs", ".java", ".kt", ".swift",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    ".sql", ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".json", ".toml", ".env.example",
    ".md", ".mdx", ".html", ".css", ".scss", ".sass",
    ".graphql", ".proto",
}

EXCLUDE_DIRS = {
    "node_modules", ".git", "__pycache__", ".next", "dist", "build",
    ".venv", "venv", ".env", "vendor", "target", ".cache", ".turbo",
    "coverage", ".nyc_output", "out", ".svelte-kit",
}

MAX_FILE_BYTES = 150_000


def scan_project(root: Path, store: NDPStore) -> int:
    count = 0
    for file_path in _walk(root):
        try:
            stat = file_path.stat()
        except OSError:
            continue
        if stat.st_size > MAX_FILE_BYTES:
            continue

        existing = store.get_by_path(str(file_path))
        if existing and existing.freshness >= stat.st_mtime:
            count += 1
            continue  # unchanged, skip re-read

        try:
            content = file_path.read_text(errors="replace")
        except Exception:
            continue

        if existing:
            existing.freshness = stat.st_mtime
            existing.token_estimate = _token_est(content)
            existing.summary = _summary(file_path, content)
            store.upsert(existing)
        else:
            store.upsert(NDPObject(
                source_path=str(file_path),
                type="file",
                source_type="file",
                summary=_summary(file_path, content),
                tags=_tags(file_path),
                freshness=stat.st_mtime,
                token_estimate=_token_est(content),
                tier="cold",
            ))
        count += 1
    return count


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix in INCLUDE_EXTENSIONS:
                yield p


def _token_est(content: str) -> int:
    return max(1, len(content) // 4)


def _summary(path: Path, content: str) -> str:
    for line in content.splitlines()[:15]:
        s = line.strip().lstrip("#/*\"' \t")
        if len(s) > 8 and not s.startswith("!"):
            return s[:100]
    return path.name


def _tags(path: Path) -> list:
    tags = [path.suffix.lstrip(".")]
    parts = list(path.parts)
    if len(parts) >= 3:
        tags.append(parts[-2])
    return [t for t in tags if t and t != "."]
