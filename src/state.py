from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models import Article, now_vn_iso


class StateStore:
    """Persists seen article URLs and initialized source IDs in a compact JSON file."""

    def __init__(self, path: str | Path, max_entries: int = 5000) -> None:
        self.path = Path(path)
        self.max_entries = max_entries
        self.initialized_sources: set[str] = set()
        self.seen: dict[str, dict[str, Any]] = {}
        self.dirty: bool = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.initialized_sources = set(data.get("initialized_sources", []))
        self.seen = dict(data.get("seen", {}))

    def is_source_initialized(self, source_id: str) -> bool:
        return source_id in self.initialized_sources

    def mark_source_initialized(self, source_id: str) -> None:
        if source_id not in self.initialized_sources:
            self.initialized_sources.add(source_id)
            self.dirty = True

    def is_seen(self, article: Article) -> bool:
        return article.canonical_url in self.seen

    def mark_seen(self, article: Article) -> None:
        canon = article.canonical_url
        if canon not in self.seen:
            self.seen[canon] = {
                "source_id": article.source_id,
                "title": article.title,
                "first_seen_at": now_vn_iso(),
            }
            self.dirty = True

    def _prune(self) -> None:
        if len(self.seen) <= self.max_entries:
            return
        # Keep the newest entries (Python dicts preserve insertion order)
        excess = len(self.seen) - self.max_entries
        keys_to_drop = list(self.seen.keys())[:excess]
        for key in keys_to_drop:
            del self.seen[key]

    def save(self) -> bool:
        """Write state to disk only if modified. Returns True if written."""
        if not self.dirty:
            return False
        self._prune()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "initialized_sources": sorted(self.initialized_sources),
            "seen": self.seen,
        }
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        tmp_path.replace(self.path)
        self.dirty = False
        return True
