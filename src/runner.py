from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from src.fetchers import build_http_session, fetch_source
from src.models import (
    Article,
    SourceConfig,
    VN_TZ,
    meets_min_importance,
    parse_published_datetime,
)
from src.notifier import send_articles_email
from src.state import StateStore
from src.summarizer import enrich_articles_with_ai

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunSummary:
    sources_checked: int = 0
    sources_failed: int = 0
    sources_initialized: int = 0
    total_fetched: int = 0
    ai_summarized: int = 0
    filtered_out_by_ai: int = 0
    new_articles: list[Article] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def load_sources(config_path: str | Path) -> list[SourceConfig]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw_sources = data.get("sources", [])
    sources = [SourceConfig.from_dict(item) for item in raw_sources]
    return [s for s in sources if s.enabled]


def run_once(
    config_path: str | Path = "config/sites.yaml",
    state_path: str | Path = "state/seen.json",
    dry_run: bool = False,
    notify_initial: bool = False,
    since_days: int = 0,
    include_seen: bool = False,
    enable_ai: bool = True,
    max_workers: int = 6,
) -> RunSummary:
    sources = load_sources(config_path)
    state = StateStore(state_path)
    summary = RunSummary()

    if not sources:
        logger.warning("Không có nguồn tin nào đang bật trong %s", config_path)
        return summary

    # Fetch all sources concurrently, but process results in config priority order
    fetched_by_id: dict[str, list[Article]] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(sources))) as pool:
        future_map = {
            pool.submit(fetch_source, source, build_http_session()): source
            for source in sources
        }
        for future in as_completed(future_map):
            source = future_map[future]
            summary.sources_checked += 1
            try:
                items = future.result()
                fetched_by_id[source.id] = items
                summary.total_fetched += len(items)
                logger.info("[%s] Quét được %d bài viết.", source.id, len(items))
            except Exception as exc:
                summary.sources_failed += 1
                summary.errors[source.id] = str(exc)
                logger.error("[%s] Lỗi khi quét %s: %s", source.id, source.url, exc)

    # Deduplicate across sources in config order
    seen_this_run: set[str] = set()
    to_notify: list[Article] = []
    to_mark_seen: list[Article] = []
    sources_to_init: list[str] = []
    cutoff_dt = (
        datetime.now(VN_TZ) - timedelta(days=since_days)
        if since_days > 0
        else None
    )

    for source in sources:
        items = fetched_by_id.get(source.id)
        if items is None:
            continue  # Source failed this run; keep previous state intact
        if not items:
            logger.warning(
                "[%s] Không tìm thấy bài viết nào (có thể web đang đổi cấu trúc hoặc tạm lỗi).",
                source.id,
            )
            continue

        is_first_run = not state.is_source_initialized(source.id)
        if is_first_run:
            sources_to_init.append(source.id)
            summary.sources_initialized += 1
            extra_note = (
                f" (gửi các bài trong {since_days} ngày gần đây)"
                if since_days > 0
                else (
                    " (và gửi email do bật --notify-initial)"
                    if notify_initial
                    else " (không gửi spam bài cũ)"
                )
            )
            logger.info(
                "[%s] Lần chạy đầu tiên: lưu %d bài hiện có làm mốc ban đầu%s.",
                source.id,
                len(items),
                extra_note,
            )

        for article in items:
            canon = article.canonical_url
            if not canon or canon in seen_this_run:
                continue
            seen_this_run.add(canon)

            already_seen = state.is_seen(article)
            if already_seen and not (include_seen and cutoff_dt is not None):
                continue

            if not already_seen:
                to_mark_seen.append(article)
            if cutoff_dt is not None:
                pub_dt = parse_published_datetime(article.published_at)
                if pub_dt is not None and pub_dt >= cutoff_dt:
                    to_notify.append(article)
            elif not is_first_run or notify_initial:
                to_notify.append(article)

    summary.new_articles = to_notify

    if dry_run:
        logger.info(
            "[DRY-RUN] Tổng cộng %d tin mới sẽ được gửi email (không gửi thật, không lưu state).",
            len(to_notify),
        )
        for art in to_notify:
            logger.info("  • [%s] %s (%s)", art.source_name, art.title, art.url)
        return summary

    # Only run AI summarization on brand-new articles that are actually going to be emailed
    if to_notify and enable_ai:
        content_selectors = {s.id: s.content_selector for s in sources if s.content_selector}
        summary.ai_summarized = enrich_articles_with_ai(
            to_notify, content_selectors_by_source=content_selectors
        )

    # Apply per-source AI importance threshold (e.g., skip "Tham khảo" PR posts on general news feeds)
    # Note: if AI summary is unavailable (no API key or transient error), we keep the article so nothing is missed.
    min_importance_by_source = {s.id: s.min_importance for s in sources if s.min_importance}
    articles_to_email: list[Article] = []
    for art in to_notify:
        threshold = min_importance_by_source.get(art.source_id, "")
        if threshold and art.ai_summary is not None:
            if not meets_min_importance(art.ai_summary.importance, threshold):
                summary.filtered_out_by_ai += 1
                logger.info(
                    "[AI-FILTER] Bỏ qua bài mức '%s' (yêu cầu >= '%s'): [%s] %s",
                    art.ai_summary.importance,
                    threshold,
                    art.source_name,
                    art.title,
                )
                continue
        articles_to_email.append(art)

    summary.new_articles = articles_to_email

    # Send email BEFORE persisting state so failed email attempts are retried on the next run
    if articles_to_email:
        send_articles_email(articles_to_email)
    elif to_notify:
        logger.info("Tất cả %d bài mới đều bị bộ lọc AI (min_importance) ẩn đi — không gửi email.", len(to_notify))

    # Mark all fetched articles (including AI-filtered ones) as seen so we don't re-call AI next run
    for source_id in sources_to_init:
        state.mark_source_initialized(source_id)
    for article in to_mark_seen:
        state.mark_seen(article)

    if state.save():
        logger.info("Đã cập nhật trạng thái vào %s.", state_path)
    else:
        logger.info("Không có tin mới nào cần cập nhật trạng thái.")

    return summary
