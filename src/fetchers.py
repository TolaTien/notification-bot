from __future__ import annotations

import html as html_lib
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.models import Article, SourceConfig, VN_TZ, normalize_url

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}

REQUEST_TIMEOUT = 20


def build_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    retry = Retry(
        total=2,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _clean_text(text: str, max_len: int = 0) -> str:
    decoded = html_lib.unescape(text or "")
    # Strip any leftover HTML tags
    no_tags = re.sub(r"<[^>]+>", " ", decoded)
    compact = re.sub(r"\s+", " ", no_tags).strip()
    if max_len and len(compact) > max_len:
        return compact[: max_len - 1].rstrip() + "…"
    return compact


def _format_iso_date(raw_date: str) -> str:
    raw = (raw_date or "").strip()
    if not raw:
        return ""
    # 1. ISO 8601
    try:
        iso = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is not None:
            dt = dt.astimezone(VN_TZ)
        return dt.strftime("%d/%m/%Y %H:%M")
    except ValueError:
        pass
    # 2. RFC 2822 (RSS pubDate like "Fri, 20 Nov 2026 16:16:06 +0000")
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is not None:
            dt = dt.astimezone(VN_TZ)
        return dt.strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError):
        pass
    # 3. US format (e.g. "11/20/2026 6:16:00 PM" from chinhphu.vn RSS)
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%d/%m/%Y %H:%M")
        except ValueError:
            continue
    return raw


def _keyword_matches(kw: str, haystack: str) -> bool:
    if len(kw) <= 4 and kw.isascii() and kw.isalnum():
        return re.search(rf"\b{re.escape(kw)}\b", haystack) is not None
    return kw in haystack


def _passes_filters(article: Article, source: SourceConfig) -> bool:
    url_lower = article.url.lower()
    if source.include_patterns and not any(
        p.lower() in url_lower for p in source.include_patterns
    ):
        return False
    if source.exclude_patterns and any(
        p.lower() in url_lower for p in source.exclude_patterns
    ):
        return False

    haystack = f"{article.title} {article.summary}".lower()
    if source.keywords and not any(_keyword_matches(kw, haystack) for kw in source.keywords):
        return False
    if source.exclude_keywords and any(_keyword_matches(kw, haystack) for kw in source.exclude_keywords):
        return False
    return True


def fetch_wp_api(source: SourceConfig, session: requests.Session) -> list[Article]:
    base = source.url.rstrip("/")
    api_url = (
        base
        if "/wp-json/" in base
        else f"{base}/wp-json/wp/v2/posts"
    )
    params: dict[str, str | int] = {
        "per_page": max(1, min(source.limit, 100)),
        "_fields": "id,date,date_gmt,link,title,excerpt",
    }
    if source.categories:
        params["categories"] = ",".join(str(c) for c in source.categories)

    resp = session.get(api_url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    posts = resp.json()

    articles: list[Article] = []
    for post in posts:
        link = str(post.get("link") or "").strip()
        title_obj = post.get("title") or {}
        title = _clean_text(
            title_obj.get("rendered", "") if isinstance(title_obj, dict) else str(title_obj)
        )
        excerpt_obj = post.get("excerpt") or {}
        summary = _clean_text(
            excerpt_obj.get("rendered", "") if isinstance(excerpt_obj, dict) else str(excerpt_obj),
            max_len=240,
        )
        raw_date = str(post.get("date") or post.get("date_gmt") or "")
        if not link or not title:
            continue
        article = Article(
            source_id=source.id,
            source_name=source.name,
            title=title,
            url=link,
            published_at=_format_iso_date(raw_date),
            summary=summary,
            source_color=source.color,
        )
        if _passes_filters(article, source):
            articles.append(article)
    return articles[: source.limit]


def _parse_rss_xml(xml_text: str, source: SourceConfig) -> list[Article]:
    root = ET.fromstring(xml_text)
    articles: list[Article] = []

    def _local_name(tag: str) -> str:
        return tag.split("}", 1)[-1].lower()

    # RSS 2.0 <item> or Atom <entry>
    entries = [el for el in root.iter() if _local_name(el.tag) in ("item", "entry")]
    for entry in entries:
        title = ""
        link = ""
        pub_date = ""
        summary = ""
        for child in entry:
            name = _local_name(child.tag)
            if name == "title" and not title:
                title = _clean_text(child.text or "")
            elif name == "link" and not link:
                href = child.attrib.get("href")
                link = (href or child.text or "").strip()
            elif name in ("pubdate", "published", "updated", "date") and not pub_date:
                pub_date = _format_iso_date((child.text or "").strip())
            elif name in ("description", "summary", "content") and not summary:
                summary = _clean_text(child.text or "", max_len=240)

        if not link or not title:
            continue
        full_url = urljoin(source.url, link)
        article = Article(
            source_id=source.id,
            source_name=source.name,
            title=title,
            url=full_url,
            published_at=pub_date,
            summary=summary,
            source_color=source.color,
        )
        if _passes_filters(article, source):
            articles.append(article)
        if len(articles) >= source.limit:
            break
    return articles


def fetch_rss(source: SourceConfig, session: requests.Session) -> list[Article]:
    resp = session.get(source.url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    # Many Vietnamese XML feeds serve Content-Type: text/xml without charset, causing requests to default to ISO-8859-1
    if resp.encoding is None or resp.encoding.lower() in ("iso-8859-1", "latin-1", "ascii"):
        resp.encoding = "utf-8"
    return _parse_rss_xml(resp.text, source)


def _extract_html_articles(html_text: str, source: SourceConfig) -> list[Article]:
    soup = BeautifulSoup(html_text, "html.parser")
    articles: list[Article] = []
    seen_urls: set[str] = set()

    def _add_candidate(link_el, container=None) -> None:
        href = (link_el.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            return
        full_url = urljoin(source.url, href)
        canon = normalize_url(full_url)
        if canon in seen_urls:
            return

        title = ""
        if container is not None and source.title_selector:
            t_el = container.select_one(source.title_selector)
            if t_el:
                title = _clean_text(t_el.get_text(" ", strip=True))
        if not title:
            title = _clean_text(link_el.get_text(" ", strip=True))
        if not title:
            title = _clean_text(link_el.get("title") or link_el.get("aria-label") or "")
        if len(title) < 6:
            return

        published_at = ""
        if container is not None and source.date_selector:
            d_el = container.select_one(source.date_selector)
            if d_el:
                raw_dt = d_el.get("datetime") or d_el.get_text(" ", strip=True)
                published_at = _format_iso_date(str(raw_dt))

        summary = ""
        if container is not None and source.summary_selector:
            s_el = container.select_one(source.summary_selector)
            if s_el:
                summary = _clean_text(s_el.get_text(" ", strip=True), max_len=240)

        article = Article(
            source_id=source.id,
            source_name=source.name,
            title=title,
            url=full_url,
            published_at=published_at,
            summary=summary,
            source_color=source.color,
        )
        if _passes_filters(article, source):
            seen_urls.add(canon)
            articles.append(article)

    if source.item_selector:
        containers = soup.select(source.item_selector)
        link_sel = source.link_selector or "a[href]"
        for container in containers:
            link_el = container.select_one(link_sel)
            if link_el:
                _add_candidate(link_el, container=container)
            if len(articles) >= source.limit:
                break
    else:
        link_sel = source.link_selector or "article a[href], h2 a[href], h3 a[href]"
        for link_el in soup.select(link_sel):
            _add_candidate(link_el, container=link_el.parent)
            if len(articles) >= source.limit:
                break

    return articles[: source.limit]


def fetch_html(source: SourceConfig, session: requests.Session) -> list[Article]:
    resp = session.get(source.url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    return _extract_html_articles(resp.text, source)


def fetch_auto(source: SourceConfig, session: requests.Session) -> list[Article]:
    """Auto-detect best extraction method: RSS -> WordPress API -> Semantic HTML."""
    resp = session.get(source.url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    content_type = (resp.headers.get("Content-Type") or "").lower()

    # 1. Direct RSS/Atom feed
    if "xml" in content_type or resp.text.lstrip().startswith(("<?xml", "<rss", "<feed")):
        try:
            items = _parse_rss_xml(resp.text, source)
            if items:
                return items
        except ET.ParseError:
            pass

    if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")

    # 2. Check for advertised WordPress API header or link
    link_header = resp.headers.get("Link") or ""
    wp_link = soup.find("link", rel="https://api.w.org/")
    if "api.w.org" in link_header or wp_link is not None:
        parsed = urlparse(source.url)
        root_url = f"{parsed.scheme}://{parsed.netloc}"
        wp_source = SourceConfig(
            id=source.id,
            name=source.name,
            url=root_url,
            type="wp_api",
            limit=source.limit,
            color=source.color,
            categories=source.categories,
            include_patterns=source.include_patterns,
            exclude_patterns=source.exclude_patterns,
            keywords=source.keywords,
            exclude_keywords=source.exclude_keywords,
        )
        try:
            items = fetch_wp_api(wp_source, session)
            if items:
                logger.info("[%s] Auto-detected WordPress API", source.id)
                return items
        except Exception as exc:
            logger.debug("[%s] WordPress API auto-detect failed: %s", source.id, exc)

    # 3. Check for advertised RSS/Atom <link>
    rss_link = soup.find(
        "link",
        attrs={"rel": "alternate", "type": re.compile(r"(rss|atom)\+xml", re.I)},
    )
    if rss_link and rss_link.get("href"):
        feed_url = urljoin(source.url, str(rss_link["href"]))
        rss_source = SourceConfig(
            id=source.id,
            name=source.name,
            url=feed_url,
            type="rss",
            limit=source.limit,
            color=source.color,
            include_patterns=source.include_patterns,
            exclude_patterns=source.exclude_patterns,
            keywords=source.keywords,
            exclude_keywords=source.exclude_keywords,
        )
        try:
            items = fetch_rss(rss_source, session)
            if items:
                logger.info("[%s] Auto-detected RSS feed: %s", source.id, feed_url)
                return items
        except Exception as exc:
            logger.debug("[%s] RSS auto-detect failed: %s", source.id, exc)

    # 4. Fallback to semantic HTML extraction
    logger.info("[%s] Auto-detected HTML scraping", source.id)
    return _extract_html_articles(resp.text, source)


def fetch_source(source: SourceConfig, session: requests.Session | None = None) -> list[Article]:
    sess = session or build_http_session()
    fetcher_map = {
        "wp_api": fetch_wp_api,
        "rss": fetch_rss,
        "html": fetch_html,
        "auto": fetch_auto,
    }
    fetcher = fetcher_map.get(source.type)
    if fetcher is None:
        raise ValueError(f"Unsupported source type '{source.type}' for source '{source.id}'")
    return fetcher(source, sess)
