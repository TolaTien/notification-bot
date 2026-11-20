from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

VN_TZ = timezone(timedelta(hours=7))

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
}


def now_vn_iso() -> str:
    return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S GMT+7")


def normalize_url(url: str) -> str:
    """Normalize a URL for reliable deduplication across scrapes."""
    cleaned = url.strip()
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    filtered_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in TRACKING_PARAMS
    ]
    query = urlencode(sorted(filtered_query))
    return urlunparse((scheme, netloc, path, "", query, ""))


@dataclass(slots=True)
class AISummary:
    tldr: str
    key_points: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    action_required: str = ""
    importance: str = "Trung bình"


@dataclass(slots=True)
class Article:
    source_id: str
    source_name: str
    title: str
    url: str
    published_at: str = ""
    summary: str = ""
    source_color: str = "#2563EB"
    detected_at: str = field(default_factory=now_vn_iso)
    full_content: str = ""
    attachments: list[tuple[str, str]] = field(default_factory=list)
    ai_summary: AISummary | None = None

    @property
    def canonical_url(self) -> str:
        return normalize_url(self.url)


@dataclass(slots=True)
class SourceConfig:
    id: str
    name: str
    url: str
    type: str = "auto"
    enabled: bool = True
    limit: int = 15
    color: str = "#2563EB"
    # wp_api options
    categories: list[int] = field(default_factory=list)
    # html options
    item_selector: str = ""
    link_selector: str = ""
    title_selector: str = ""
    date_selector: str = ""
    summary_selector: str = ""
    content_selector: str = ""
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    # shared filtering options
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceConfig":
        return cls(
            id=str(raw["id"]).strip(),
            name=str(raw.get("name") or raw["id"]).strip(),
            url=str(raw["url"]).strip(),
            type=str(raw.get("type", "auto")).strip().lower(),
            enabled=bool(raw.get("enabled", True)),
            limit=int(raw.get("limit", 15)),
            color=str(raw.get("color", "#2563EB")).strip(),
            categories=[int(c) for c in raw.get("categories", [])],
            item_selector=str(raw.get("item_selector", "")).strip(),
            link_selector=str(raw.get("link_selector", "")).strip(),
            title_selector=str(raw.get("title_selector", "")).strip(),
            date_selector=str(raw.get("date_selector", "")).strip(),
            summary_selector=str(raw.get("summary_selector", "")).strip(),
            content_selector=str(raw.get("content_selector", "")).strip(),
            include_patterns=[str(p) for p in raw.get("include_patterns", [])],
            exclude_patterns=[str(p) for p in raw.get("exclude_patterns", [])],
            keywords=[str(k).lower() for k in raw.get("keywords", [])],
            exclude_keywords=[str(k).lower() for k in raw.get("exclude_keywords", [])],
        )
