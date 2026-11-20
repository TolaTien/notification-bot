from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.content_extractor import _extract_from_html
from src.fetchers import _extract_html_articles, _parse_rss_xml
from src.models import (
    AISummary,
    Article,
    SourceConfig,
    meets_min_importance,
    normalize_url,
    parse_published_datetime,
)
from src.notifier import build_email_bodies, build_email_subject
from src.runner import run_once
from src.state import StateStore
from src.summarizer import AIConfig, AISummarizer, ModelTarget, parse_ai_json


class TestNotificationBot(unittest.TestCase):
    def test_normalize_url_strips_tracking_and_trailing_slash(self) -> None:
        a = normalize_url("https://PTIT.edu.vn/thong-bao-1/?utm_source=fb&fbclid=123#top")
        b = normalize_url("https://ptit.edu.vn/thong-bao-1")
        self.assertEqual(a, b)

    def test_html_extractor_iig_markup(self) -> None:
        sample_html = """
        <html><body>
          <article class="cd-featured-slide">
            <a class="cd-featured-pic" href="/vi/tin-tuc/bai-1"><img src="x.jpg"/></a>
            <h3 class="cd-featured-title">
              <a href="/vi/tin-tuc/bai-1">Thông báo lịch nghỉ lễ mới nhất</a>
            </h3>
            <time datetime="2026-11-10T02:42:26.000Z">10/11/2026</time>
            <p class="cd-featured-excerpt">Nội dung tóm tắt bài viết 1</p>
          </article>
          <article class="cd-card">
            <h2 class="cd-card-title">
              <a class="cd-card-link" href="/vi/tin-tuc/chuyen-muc/quy-dinh">Chuyên mục bị loại</a>
            </h2>
          </article>
        </body></html>
        """
        source = SourceConfig(
            id="iig-test",
            name="IIG Test",
            url="https://iigvietnam.com/vi/tin-tuc",
            type="html",
            item_selector="article.cd-featured-slide, article.cd-card",
            link_selector="h2 a, h3 a, a.cd-card-link",
            date_selector="time",
            summary_selector="p.cd-featured-excerpt, p.cd-card-text",
            include_patterns=["/vi/tin-tuc/"],
            exclude_patterns=["/chuyen-muc/"],
        )
        items = _extract_html_articles(sample_html, source)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Thông báo lịch nghỉ lễ mới nhất")
        self.assertEqual(items[0].url, "https://iigvietnam.com/vi/tin-tuc/bai-1")
        self.assertIn("10/11/2026", items[0].published_at)

    def test_rss_parser_and_keyword_filter(self) -> None:
        sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
          <item>
            <title>Thông báo lịch thi học kỳ 1</title>
            <link>https://example.com/lich-thi</link>
            <description>Chi tiết lịch thi sinh viên</description>
          </item>
          <item>
            <title>Bản tin sự kiện giải trí</title>
            <link>https://example.com/su-kien</link>
          </item>
        </channel></rss>
        """
        source = SourceConfig(
            id="rss-test",
            name="RSS Test",
            url="https://example.com/feed",
            type="rss",
            keywords=["thông báo"],
        )
        items = _parse_rss_xml(sample_rss, source)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://example.com/lich-thi")

    def test_short_keyword_uses_word_boundary(self) -> None:
        sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
          <item>
            <title>Hướng dẫn cài đặt tài khoản mới</title>
            <link>https://example.com/cai-dat</link>
          </item>
          <item>
            <title>OpenAI ra mắt mô hình AI thế hệ mới</title>
            <link>https://example.com/ai-news</link>
          </item>
        </channel></rss>
        """
        source = SourceConfig(
            id="tech-test",
            name="Tech Test",
            url="https://example.com/feed",
            type="rss",
            keywords=["ai"],
        )
        items = _parse_rss_xml(sample_rss, source)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://example.com/ai-news")

    def test_content_extractor_extracts_body_and_pdf_attachments(self) -> None:
        html = """
        <html><body>
          <nav>Menu cần bỏ qua</nav>
          <div class="post-content">
            <p>Phòng Đào tạo thông báo tạm hoãn lịch phát bằng lẻ chiều thứ Sáu ngày 20/11/2026 tại Cơ sở Hà Đông.</p>
            <p>Sinh viên theo dõi lịch phát lại vào tuần sau và mang theo thẻ sinh viên bản gốc.</p>
            <a href="/wp-content/uploads/2026/11/thong-bao.pdf">Tải công văn đính kèm</a>
          </div>
          <footer>Footer cần bỏ qua</footer>
        </body></html>
        """
        text, attachments = _extract_from_html(html, "https://ptit.edu.vn/bai-viet/", ".post-content")
        self.assertIn("tạm hoãn lịch phát bằng lẻ", text)
        self.assertNotIn("Menu cần bỏ qua", text)
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0][0], "Tải công văn đính kèm")
        self.assertEqual(attachments[0][1], "https://ptit.edu.vn/wp-content/uploads/2026/11/thong-bao.pdf")

    def test_parse_ai_json_with_code_fences(self) -> None:
        raw = """```json
        {
          "tldr": "PTIT tạm hoãn phát bằng lẻ chiều 20/11/2026 tại cơ sở Hà Đông.",
          "key_points": ["Hoãn phát bằng lẻ chiều 20/11/2026", "Áp dụng tại cơ sở đào tạo Hà Đông"],
          "deadlines": ["20/11/2026: Tạm hoãn phát bằng"],
          "action_required": "Chờ thông báo lịch phát bù vào tuần tiếp theo.",
          "importance": "Cao"
        }
        ```"""
        parsed = parse_ai_json(raw)
        self.assertEqual(parsed.importance, "Cao")
        self.assertEqual(len(parsed.key_points), 2)
        self.assertEqual(len(parsed.deadlines), 1)

    def test_ai_summarizer_graceful_fallback_on_api_error(self) -> None:
        art = Article("ptit", "PTIT", "Thông báo thử nghiệm", "https://ptit.edu.vn/test", full_content="Nội dung bài viết đủ dài để gửi lên AI tóm tắt chi tiết cho sinh viên.")
        cfg = AIConfig(provider="gemini", api_key="fake-key", model="gemini-2.5-flash")
        summarizer = AISummarizer(cfg)
        fake_session = MagicMock()
        fake_session.post.side_effect = RuntimeError("Network error")

        result = summarizer.summarize_article(art, session=fake_session, retry_sleep_seconds=0)
        self.assertIsNone(result)
        self.assertIsNone(art.ai_summary)

    def test_ai_model_automatic_failover_on_429_quota_limit(self) -> None:
        import requests

        art1 = Article("ptit", "PTIT", "Bài 1", "https://ptit.edu.vn/1", full_content="Nội dung bài viết số 1 đủ dài để AI phân tích và tóm tắt.")
        art2 = Article("ptit", "PTIT", "Bài 2", "https://ptit.edu.vn/2", full_content="Nội dung bài viết số 2 đủ dài để AI phân tích và tóm tắt.")

        cfg = AIConfig(
            provider="gemini",
            api_key="fake-key",
            model="gemini-2.5-flash",
            targets=[
                ModelTarget("gemini", "fake-key", "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta"),
                ModelTarget("gemini", "fake-key", "gemini-2.5-flash-lite", "https://generativelanguage.googleapis.com/v1beta"),
            ],
        )
        summarizer = AISummarizer(cfg)

        called_urls: list[str] = []

        def fake_post(url, **_kwargs):
            called_urls.append(url)
            resp = MagicMock()
            if "gemini-2.5-flash:" in url:
                resp.status_code = 429
                err = requests.HTTPError("429 Too Many Requests", response=resp)
                resp.raise_for_status.side_effect = err
                return resp
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "tldr": "Tóm tắt thành công bằng model dự phòng.",
                                            "key_points": ["Ý 1"],
                                            "deadlines": [],
                                            "action_required": "",
                                            "importance": "Cao",
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
            return resp

        fake_session = MagicMock()
        fake_session.post.side_effect = fake_post

        # Article 1: primary hits 429 -> automatically fails over to flash-lite
        res1 = summarizer.summarize_article(art1, session=fake_session, retry_sleep_seconds=0)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.tldr, "Tóm tắt thành công bằng model dự phòng.")
        self.assertEqual(len(called_urls), 2)
        self.assertIn("gemini-2.5-flash:", called_urls[0])
        self.assertIn("gemini-2.5-flash-lite:", called_urls[1])

        # Article 2: primary is marked exhausted, so it goes STRAIGHT to flash-lite (1 call only!)
        res2 = summarizer.summarize_article(art2, session=fake_session, retry_sleep_seconds=0)
        self.assertIsNotNone(res2)
        self.assertEqual(len(called_urls), 3)
        self.assertIn("gemini-2.5-flash-lite:", called_urls[2])

    def test_gemma_model_omits_system_instruction_and_json_mime_type(self) -> None:
        art = Article("ptit", "PTIT", "Bài Gemma", "https://ptit.edu.vn/g", full_content="Nội dung bài viết kiểm tra gọi Gemma 4 trên Gemini API.")
        cfg = AIConfig(
            provider="gemini",
            api_key="fake-key",
            model="gemma-4-31b-it",
            targets=[
                ModelTarget("gemini", "fake-key", "gemma-4-31b-it", "https://generativelanguage.googleapis.com/v1beta"),
            ],
        )
        summarizer = AISummarizer(cfg)
        captured_payloads: list[dict] = []

        def fake_post(url, **kwargs):
            captured_payloads.append(kwargs["json"])
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '```json\n{"tldr":"OK Gemma","key_points":["1"],"deadlines":[],"action_required":"","importance":"Trung bình"}\n```'
                                }
                            ]
                        }
                    }
                ]
            }
            return resp

        fake_session = MagicMock()
        fake_session.post.side_effect = fake_post
        res = summarizer.summarize_article(art, session=fake_session, retry_sleep_seconds=0)
        self.assertIsNotNone(res)
        self.assertEqual(res.tldr, "OK Gemma")
        self.assertNotIn("system_instruction", captured_payloads[0])
        self.assertNotIn("responseMimeType", captured_payloads[0]["generationConfig"])

    def test_first_run_seeds_without_spam_then_notifies_new_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "sites.yaml"
            state_path = tmp_path / "seen.json"
            cfg_path.write_text(
                "sources:\n"
                "  - id: s1\n"
                "    name: Source 1\n"
                "    url: https://a.test\n"
                "    type: html\n"
                "  - id: s2\n"
                "    name: Source 2\n"
                "    url: https://b.test\n"
                "    type: html\n",
                encoding="utf-8",
            )

            art_old = Article("s1", "Source 1", "Tin cũ", "https://a.test/old")
            art_new_s1 = Article("s1", "Source 1", "Tin mới nóng hổi", "https://a.test/new")
            art_new_dup_s2 = Article("s2", "Source 2", "Tin mới nóng hổi", "https://a.test/new/")

            # Run 1: Initial seed -> should NOT send email
            with (
                patch("src.runner.fetch_source", side_effect=lambda src, _sess: [art_old]),
                patch("src.runner.send_articles_email") as mock_send,
            ):
                s1 = run_once(cfg_path, state_path, enable_ai=False)
                self.assertEqual(len(s1.new_articles), 0)
                mock_send.assert_not_called()

            # Run 2: New post appears in BOTH s1 and s2 -> should send ONCE under Source 1
            with (
                patch(
                    "src.runner.fetch_source",
                    side_effect=lambda src, _sess: [art_new_s1, art_old] if src.id == "s1" else [art_new_dup_s2, art_old],
                ),
                patch("src.runner.send_articles_email") as mock_send,
            ):
                s2 = run_once(cfg_path, state_path, enable_ai=False)
                self.assertEqual(len(s2.new_articles), 1)
                self.assertEqual(s2.new_articles[0].source_name, "Source 1")
                mock_send.assert_called_once()

            # Run 3: Same articles -> should send NOTHING
            with (
                patch("src.runner.fetch_source", return_value=[art_new_s1, art_old]),
                patch("src.runner.send_articles_email") as mock_send,
            ):
                s3 = run_once(cfg_path, state_path, enable_ai=False)
                self.assertEqual(len(s3.new_articles), 0)
                mock_send.assert_not_called()

    def test_state_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "seen.json"
            store = StateStore(state_path, max_entries=2)
            for i in range(4):
                store.mark_seen(Article("s", "S", f"T{i}", f"https://x.test/{i}"))
            self.assertTrue(store.save())
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["seen"]), 2)
            self.assertIn("https://x.test/3", saved["seen"])

    def test_email_builder_with_ai_summary_and_attachments(self) -> None:
        art = Article(
            "ptit",
            "PTIT - Thông báo",
            "Thông báo lịch thi",
            "https://ptit.edu.vn/a",
            "20/11/2026",
            attachments=[("Lịch thi PDF", "https://ptit.edu.vn/lich-thi.pdf")],
            ai_summary=AISummary(
                tldr="Đã có lịch thi học kỳ 1 năm học 2026-2027.",
                key_points=["Thi từ ngày 01/12/2026", "Mang thẻ sinh viên khi đi thi"],
                deadlines=["01/12/2026: Bắt đầu thi"],
                action_required="Kiểm tra phòng thi trên cổng đào tạo.",
                importance="Cao",
            ),
        )
        subject = build_email_subject([art])
        plain, html_body = build_email_bodies([art])
        self.assertIn("🔴", subject)
        self.assertIn("PTIT - Thông báo", subject)
        self.assertIn("Kiểm tra phòng thi", plain)
        self.assertIn("Tóm tắt nhanh", html_body)
        self.assertIn("Mốc thời gian / Hạn chót", html_body)
        self.assertIn("Tài liệu đính kèm", html_body)
        self.assertIn("https://ptit.edu.vn/lich-thi.pdf", html_body)

    def test_ai_min_importance_filters_pr_posts_and_marks_them_seen(self) -> None:
        self.assertFalse(meets_min_importance("Tham khảo", "Trung bình"))
        self.assertTrue(meets_min_importance("Trung bình", "Trung bình"))
        self.assertTrue(meets_min_importance("Cao", "Trung bình"))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "sites.yaml"
            state_path = tmp_path / "seen.json"
            cfg_path.write_text(
                "sources:\n"
                "  - id: news\n"
                "    name: Tin tức chung\n"
                "    url: https://news.test\n"
                "    type: html\n"
                '    min_importance: "Trung bình"\n',
                encoding="utf-8",
            )
            # Pre-seed source as already initialized
            state_path.write_text(
                json.dumps({"initialized_sources": ["news"], "seen": {}}),
                encoding="utf-8",
            )

            pr_post = Article("news", "Tin tức chung", "Lễ kỷ niệm 30 năm", "https://news.test/pr")
            important_post = Article("news", "Tin tức chung", "Thông báo đổi lịch thi", "https://news.test/exam")

            def fake_enrich(articles, **_kwargs):
                for a in articles:
                    if "pr" in a.url:
                        a.ai_summary = AISummary(tldr="Tin kỷ niệm", importance="Tham khảo")
                    else:
                        a.ai_summary = AISummary(tldr="Đổi lịch thi gấp", importance="Cao")
                return len(articles)

            with (
                patch("src.runner.fetch_source", return_value=[pr_post, important_post]),
                patch("src.runner.enrich_articles_with_ai", side_effect=fake_enrich),
                patch("src.runner.send_articles_email") as mock_send,
            ):
                res = run_once(cfg_path, state_path, enable_ai=True)
                self.assertEqual(res.filtered_out_by_ai, 1)
                self.assertEqual(len(res.new_articles), 1)
                self.assertEqual(res.new_articles[0].url, "https://news.test/exam")
                mock_send.assert_called_once()
                sent_list = mock_send.call_args[0][0]
                self.assertEqual([a.url for a in sent_list], ["https://news.test/exam"])

            # Verify PR post was STILL saved to seen.json so AI isn't re-invoked on next run
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("https://news.test/pr", saved["seen"])
            self.assertIn("https://news.test/exam", saved["seen"])

    def test_parse_published_datetime(self) -> None:
        dt = parse_published_datetime("20/11/2026 11:18")
        self.assertIsNotNone(dt)
        self.assertEqual((dt.year, dt.month, dt.day), (2026, 11, 20))
        self.assertIsNone(parse_published_datetime("invalid"))

    def test_since_days_with_include_seen_resends_recent_seen_articles(self) -> None:
        from datetime import datetime
        from src.models import VN_TZ

        today_str = datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "sites.yaml"
            state_path = tmp_path / "seen.json"
            cfg_path.write_text(
                "sources:\n"
                "  - id: s1\n"
                "    name: Source 1\n"
                "    url: https://a.test\n"
                "    type: html\n",
                encoding="utf-8",
            )
            state_path.write_text(
                json.dumps(
                    {
                        "initialized_sources": ["s1"],
                        "seen": {"https://a.test/recent": {"source_id": "s1", "title": "Recent"}},
                    }
                ),
                encoding="utf-8",
            )

            art_recent = Article("s1", "Source 1", "Tin hôm nay", "https://a.test/recent", published_at=today_str)
            art_ancient = Article("s1", "Source 1", "Tin rất cũ", "https://a.test/ancient", published_at="01/01/2020 08:00")

            with (
                patch("src.runner.fetch_source", return_value=[art_recent, art_ancient]),
                patch("src.runner.send_articles_email") as mock_send,
            ):
                res = run_once(cfg_path, state_path, since_days=30, include_seen=True, enable_ai=False)
                self.assertEqual([a.url for a in res.new_articles], ["https://a.test/recent"])
                mock_send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
