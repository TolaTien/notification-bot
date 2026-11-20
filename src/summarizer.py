from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

from src.content_extractor import fetch_article_content
from src.fetchers import build_http_session
from src.models import AISummary, Article

logger = logging.getLogger(__name__)

AI_TIMEOUT = 35

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý biên tập tin tức thông minh bằng tiếng Việt. "
    "Nhiệm vụ của bạn là đọc kỹ toàn bộ bài viết/thông báo và tóm tắt lại sao cho người đọc "
    "chỉ cần đọc email là nắm trọn toàn bộ thông tin quan trọng mà không cần mở trang web gốc.\n"
    "Quy tắc bắt buộc:\n"
    "- Trung thực tuyệt đối với nội dung bài gốc, KHÔNG bịa đặt hay suy diễn số liệu/ngày giờ.\n"
    "- Giữ chính xác mọi con số, ngày tháng, địa điểm, lệ phí, điều kiện, đối tượng áp dụng.\n"
    "- Viết tiếng Việt tự nhiên, súc tích, dễ quét mắt.\n"
    "- Quy tắc chấm trường 'importance' (chỉ chọn 1 trong 3 giá trị chính xác):\n"
    '  + "Cao": Thông báo ảnh hưởng trực tiếp tới lịch học, lịch thi, đăng ký môn, học phí, hạn chót, lịch nghỉ, quy định/thủ tục, văn bằng, học bổng, tuyển sinh.\n'
    '  + "Trung bình": Cuộc thi học thuật, hội thảo hữu ích, cơ hội việc làm/thực tập, chương trình đào tạo/chứng chỉ mới đáng chú ý.\n'
    '  + "Tham khảo": Tin PR, lễ kỷ niệm, gặp mặt tri ân, ký kết hợp tác, vinh danh đối tác/cá nhân, tin hoạt động truyền thông chung.\n'
    "- Chỉ trả về đúng 1 JSON object hợp lệ (không thêm lời giải thích ngoài JSON) theo schema:\n"
    "{\n"
    '  "tldr": "Tóm tắt cốt lõi trong 1-2 câu ngắn gọn.",\n'
    '  "key_points": ["Ý chính 1", "Ý chính 2", "Ý chính 3"],\n'
    '  "deadlines": ["Mốc thời gian / hạn chót cụ thể nếu có, nếu không có thì trả về []"],\n'
    '  "action_required": "Việc người đọc cần làm nếu có (ví dụ: đăng ký trước ngày...), nếu không có thì để chuỗi rỗng \\"\\"",\n'
    '  "importance": "Cao | Trung bình | Tham khảo"\n'
    "}"
)


def _build_user_prompt(article: Article, body_text: str) -> str:
    parts = [
        f"Nguồn: {article.source_name}",
        f"Tiêu đề: {article.title}",
    ]
    if article.published_at:
        parts.append(f"Ngày đăng: {article.published_at}")
    if article.attachments:
        att_names = ", ".join(name for name, _ in article.attachments)
        parts.append(f"Tài liệu đính kèm trong bài: {att_names}")
    parts.append(f"\n--- NỘI DUNG BÀI VIẾT ---\n{body_text}")
    return "\n".join(parts)


def parse_ai_json(raw_text: str) -> AISummary:
    cleaned = (raw_text or "").strip()
    # Strip optional markdown code fences like ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]

    data = json.loads(cleaned)
    tldr = str(data.get("tldr") or "").strip()
    if not tldr:
        raise ValueError("AI JSON missing non-empty 'tldr'")

    key_points = [
        str(item).strip()
        for item in (data.get("key_points") or [])
        if str(item).strip()
    ][:6]
    deadlines = [
        str(item).strip()
        for item in (data.get("deadlines") or [])
        if str(item).strip()
    ][:5]
    action_required = str(data.get("action_required") or "").strip()
    raw_imp = str(data.get("importance") or "Trung bình").strip()
    importance = raw_imp if raw_imp in ("Cao", "Trung bình", "Tham khảo") else "Trung bình"

    return AISummary(
        tldr=tldr,
        key_points=key_points,
        deadlines=deadlines,
        action_required=action_required,
        importance=importance,
    )


@dataclass(slots=True)
class AIConfig:
    provider: str  # "gemini" or "openai"
    api_key: str
    model: str
    base_url: str = ""

    @classmethod
    def from_env(cls) -> "AIConfig | None":
        provider = (os.getenv("AI_PROVIDER") or "").strip().lower()
        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        generic_key = (os.getenv("AI_API_KEY") or "").strip()
        model = (os.getenv("AI_MODEL") or "").strip()
        base_url = (os.getenv("AI_BASE_URL") or "").strip()

        if not provider:
            if gemini_key:
                provider = "gemini"
            elif generic_key:
                provider = "openai"
            else:
                return None

        if provider == "gemini":
            key = gemini_key or generic_key
            if not key:
                return None
            return cls(
                provider="gemini",
                api_key=key,
                model=model or "gemini-2.5-flash",
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta",
            )

        if provider in ("openai", "groq", "openrouter", "deepseek", "openai_compatible"):
            key = generic_key or gemini_key
            if not key:
                return None
            default_urls = {
                "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
                "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
                "openrouter": ("https://openrouter.ai/api/v1", "google/gemini-2.0-flash-001"),
                "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
                "openai_compatible": ("https://api.openai.com/v1", "gpt-4o-mini"),
            }
            def_url, def_model = default_urls[provider]
            return cls(
                provider="openai",
                api_key=key,
                model=model or def_model,
                base_url=(base_url or def_url).rstrip("/"),
            )

        logger.warning("AI_PROVIDER '%s' không được hỗ trợ; tắt tóm tắt AI.", provider)
        return None


class AISummarizer:
    def __init__(self, config: AIConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls) -> "AISummarizer | None":
        cfg = AIConfig.from_env()
        if cfg is None:
            return None
        return cls(cfg)

    def _call_gemini(self, prompt: str, session: requests.Session) -> str:
        url = (
            f"{self.config.base_url.rstrip('/')}/models/"
            f"{self.config.model}:generateContent"
        )
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        resp = session.post(
            url,
            params={"key": self.config.api_key},
            json=payload,
            timeout=AI_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini không trả về kết quả: {data}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(str(p.get("text") or "") for p in parts)

    def _call_openai_compatible(self, prompt: str, session: requests.Session) -> str:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        }
        resp = session.post(url, headers=headers, json=payload, timeout=AI_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"AI API không trả về kết quả: {data}")
        return str((choices[0].get("message") or {}).get("content") or "")

    def summarize_article(
        self,
        article: Article,
        session: requests.Session | None = None,
        content_selector: str = "",
    ) -> AISummary | None:
        sess = session or build_http_session()

        if not article.full_content:
            try:
                body, attachments = fetch_article_content(
                    article.url, sess, content_selector=content_selector
                )
                article.full_content = body
                article.attachments = attachments
            except Exception as exc:
                logger.warning(
                    "[AI] Không tải được toàn văn bài '%s': %s (sẽ dùng tóm tắt ngắn nếu có).",
                    article.title[:60],
                    exc,
                )

        body_text = article.full_content or article.summary
        if not body_text or len(body_text.strip()) < 25:
            logger.info("[AI] Bài '%s' không đủ nội dung để tóm tắt.", article.title[:60])
            return None

        prompt = _build_user_prompt(article, body_text)
        for attempt in range(3):
            try:
                if self.config.provider == "gemini":
                    raw = self._call_gemini(prompt, sess)
                else:
                    raw = self._call_openai_compatible(prompt, sess)
                summary = parse_ai_json(raw)
                article.ai_summary = summary
                return summary
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 429 and attempt < 2:
                    wait_sec = 12 * (attempt + 1)
                    logger.info(
                        "[AI] Chạm giới hạn tốc độ (429), đợi %ds rồi thử lại bài '%s'...",
                        wait_sec,
                        article.title[:50],
                    )
                    time.sleep(wait_sec)
                    continue
                logger.warning(
                    "[AI] Lỗi khi tóm tắt bài '%s' (sẽ fallback về tóm tắt gốc): %s",
                    article.title[:60],
                    exc,
                )
                return None
            except Exception as exc:
                logger.warning(
                    "[AI] Lỗi khi tóm tắt bài '%s' (sẽ fallback về tóm tắt gốc): %s",
                    article.title[:60],
                    exc,
                )
                return None
        return None


def enrich_articles_with_ai(
    articles: list[Article],
    content_selectors_by_source: dict[str, str] | None = None,
    summarizer: AISummarizer | None = None,
    max_workers: int = 4,
) -> int:
    """Enrich newly discovered articles with full content + AI summary. Returns count summarized."""
    if not articles:
        return 0

    ai = summarizer or AISummarizer.from_env()
    if ai is None:
        logger.info("Chưa cấu hình AI API key (GEMINI_API_KEY / AI_API_KEY) — gửi email theo tóm tắt gốc.")
        return 0

    selectors = content_selectors_by_source or {}
    logger.info(
        "Đang dùng AI (%s / %s) đọc & tóm tắt %d bài viết mới...",
        ai.config.provider,
        ai.config.model,
        len(articles),
    )

    summarized_count = 0
    workers = min(2 if len(articles) > 6 else max_workers, len(articles))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                ai.summarize_article,
                art,
                build_http_session(),
                selectors.get(art.source_id, ""),
            ): art
            for art in articles
        }
        for future in as_completed(futures):
            art = futures[future]
            try:
                if future.result() is not None:
                    summarized_count += 1
                    logger.info("  ✓ [AI] Đã tóm tắt xong: %s", art.title[:70])
            except Exception as exc:
                logger.warning("  ✗ [AI] Lỗi bất ngờ khi tóm tắt '%s': %s", art.title[:60], exc)

    return summarized_count
