from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.content_extractor import _extract_from_html
from src.fetchers import _extract_html_articles, _parse_rss_xml
from src.models import AISummary, Article, SourceConfig, meets_min_importance, normalize_url
from src.notifier import build_email_bodies, build_email_subject
from src.runner import run_once
from src.state import StateStore
from src.summarizer import AIConfig, AISummarizer, parse_ai_json


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

        result = summarizer.summarize_article(art, session=fake_session)
        self.assertIsNone(result)
        self.assertIsNone(art.ai_summary)

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


if __name__ == "__main__":
    unittest.main()
