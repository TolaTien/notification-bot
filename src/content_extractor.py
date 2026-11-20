from __future__ import annotations

import logging
import re
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.fetchers import REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

ATTACHMENT_EXTENSIONS = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar")

DEFAULT_CONTENT_SELECTORS = [
    ".post-content",
    ".entry-content",
    ".article-content",
    ".single-content",
    ".fck_detail",
    ".detail-content",
    ".content-detail",
    ".cd-detail-body",
    ".cd-detail",
    ".prose",
    "article",
    "main",
]

NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    ".related-posts",
    ".cd-related",
    ".share",
    ".social-share",
    ".breadcrumb",
    ".breadcrumbs",
    ".comments",
]


def _extract_from_html(
    html_text: str,
    base_url: str,
    content_selector: str = "",
    max_chars: int = 12000,
) -> tuple[str, list[tuple[str, str]]]:
    soup = BeautifulSoup(html_text, "html.parser")

    container = None
    selectors = [content_selector] if content_selector else []
    selectors.extend(DEFAULT_CONTENT_SELECTORS)
    for sel in selectors:
        if not sel:
            continue
        found = soup.select_one(sel)
        if found is not None and len(found.get_text(strip=True)) >= 80:
            container = found
            break

    if container is None:
        container = soup.body or soup

    # Collect attachments (PDF/Word/Excel) before stripping anything
    attachments: list[tuple[str, str]] = []
    seen_att_urls: set[str] = set()
    for a_tag in container.select("a[href]"):
        href = (a_tag.get("href") or "").strip()
        if not href:
            continue
        full_url = urljoin(base_url, href)
        path_lower = urlparse(full_url).path.lower()
        if any(path_lower.endswith(ext) for ext in ATTACHMENT_EXTENSIONS):
            if full_url in seen_att_urls:
                continue
            seen_att_urls.add(full_url)
            label = " ".join(a_tag.get_text(" ", strip=True).split())
            if not label:
                label = unquote(urlparse(full_url).path.rsplit("/", 1)[-1]) or "Tài liệu đính kèm"
            attachments.append((label[:120], full_url))

    # Remove non-content elements from a copy/in-place
    for noise_sel in NOISE_SELECTORS:
        for el in container.select(noise_sel):
            el.decompose()

    # Extract clean paragraph-separated text
    lines: list[str] = []
    for raw_line in container.get_text("\n", strip=True).splitlines():
        cleaned = re.sub(r"\s+", " ", raw_line).strip()
        if cleaned:
            lines.append(cleaned)

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n…[đã rút gọn]"

    return text, attachments[:8]


def fetch_article_content(
    url: str,
    session: requests.Session,
    content_selector: str = "",
    max_chars: int = 12000,
) -> tuple[str, list[tuple[str, str]]]:
    """Fetch full article page and extract clean readable text + document attachments."""
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    return _extract_from_html(
        resp.text,
        base_url=url,
        content_selector=content_selector,
        max_chars=max_chars,
    )
