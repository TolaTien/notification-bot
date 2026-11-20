from __future__ import annotations

import html as html_lib
import logging
import os
import smtplib
from collections import defaultdict
from dataclasses import dataclass
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

from src.models import AISummary, Article, now_vn_iso

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmailConfig:
    host: str
    port: int
    user: str
    password: str
    recipients: list[str]
    sender_name: str = "News Monitor Bot"

    @classmethod
    def from_env(cls) -> "EmailConfig":
        host = (os.getenv("SMTP_HOST") or "").strip() or "smtp.gmail.com"
        port_raw = (os.getenv("SMTP_PORT") or "").strip() or "465"
        port = int(port_raw)
        user = (os.getenv("SMTP_USER") or "").strip()
        password = (os.getenv("SMTP_PASSWORD") or "").strip()
        raw_to = (os.getenv("NOTIFY_TO") or "").strip() or user
        recipients = [addr.strip() for addr in raw_to.split(",") if addr.strip()]
        sender_name = (os.getenv("SENDER_NAME") or "").strip() or "News Monitor Bot"

        missing = [
            name
            for name, val in [
                ("SMTP_USER", user),
                ("SMTP_PASSWORD", password),
                ("NOTIFY_TO", recipients),
            ]
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"Thiếu biến môi trường cấu hình email: {', '.join(missing)}. "
                "Vui lòng kiểm tra file .env hoặc GitHub Secrets."
            )
        return cls(
            host=host,
            port=port,
            user=user,
            password=password,
            recipients=recipients,
            sender_name=sender_name,
        )


def build_email_subject(articles: list[Article]) -> str:
    if len(articles) == 1:
        art = articles[0]
        short_title = art.title if len(art.title) <= 85 else art.title[:84] + "…"
        prefix = "🔴 " if art.ai_summary and art.ai_summary.importance == "Cao" else ""
        return f"{prefix}[Tin mới - {art.source_name}] {short_title}"
    source_names = list(dict.fromkeys(a.source_name for a in articles))
    sources_preview = ", ".join(source_names[:3])
    if len(source_names) > 3:
        sources_preview += f" (+{len(source_names) - 3})"
    return f"[{len(articles)} tin mới] {sources_preview}"


def _importance_badge_html(importance: str) -> str:
    styles = {
        "Cao": ("#FEE2E2", "#B91C1C", "🔴 QUAN TRỌNG"),
        "Trung bình": ("#FEF3C7", "#B45309", "🟡 ĐÁNG CHÚ Ý"),
        "Tham khảo": ("#F3F4F6", "#4B5563", "⚪ THAM KHẢO"),
    }
    bg, fg, label = styles.get(importance, styles["Trung bình"])
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:11px;font-weight:700;letter-spacing:0.3px;">'
        f"{label}</span>"
    )


def _render_ai_card_html(ai: AISummary) -> str:
    parts: list[str] = []

    # TL;DR box
    parts.append(
        f'<div style="margin:10px 0;padding:10px 12px;background:#F8FAFC;'
        f'border-left:3px solid #3B82F6;border-radius:6px;font-size:14px;color:#1F2937;line-height:1.55;">'
        f'<strong>✨ Tóm tắt nhanh:</strong> {html_lib.escape(ai.tldr)}'
        f"</div>"
    )

    # Key points
    if ai.key_points:
        li_items = "".join(
            f'<li style="margin-bottom:4px;">{html_lib.escape(point)}</li>'
            for point in ai.key_points
        )
        parts.append(
            f'<div style="margin:10px 0;font-size:14px;color:#374151;line-height:1.5;">'
            f'<div style="font-weight:600;color:#111827;margin-bottom:4px;">📌 Nội dung chính:</div>'
            f'<ul style="margin:0;padding-left:18px;">{li_items}</ul>'
            f"</div>"
        )

    # Deadlines box
    if ai.deadlines:
        dl_items = "".join(
            f'<li style="margin-bottom:3px;">{html_lib.escape(dl)}</li>'
            for dl in ai.deadlines
        )
        parts.append(
            f'<div style="margin:10px 0;padding:10px 12px;background:#FFFBEB;'
            f'border:1px solid #FDE68A;border-radius:6px;font-size:13.5px;color:#92400E;line-height:1.5;">'
            f'<div style="font-weight:700;margin-bottom:4px;">⏰ Mốc thời gian / Hạn chót:</div>'
            f'<ul style="margin:0;padding-left:18px;">{dl_items}</ul>'
            f"</div>"
        )

    # Action required box
    if ai.action_required:
        parts.append(
            f'<div style="margin:10px 0;padding:10px 12px;background:#EFF6FF;'
            f'border:1px solid #BFDBFE;border-radius:6px;font-size:13.5px;color:#1E40AF;line-height:1.5;">'
            f'<strong>✅ Việc cần làm:</strong> {html_lib.escape(ai.action_required)}'
            f"</div>"
        )

    return "".join(parts)


def _render_attachments_html(attachments: list[tuple[str, str]]) -> str:
    if not attachments:
        return ""
    links = "".join(
        f'<li style="margin-bottom:3px;">'
        f'<a href="{html_lib.escape(url, quote=True)}" target="_blank" '
        f'style="color:#2563EB;text-decoration:underline;">{html_lib.escape(label)}</a>'
        f"</li>"
        for label, url in attachments
    )
    return (
        f'<div style="margin:10px 0;font-size:13px;color:#374151;">'
        f'<div style="font-weight:600;margin-bottom:4px;">📎 Tài liệu đính kèm:</div>'
        f'<ul style="margin:0;padding-left:18px;">{links}</ul>'
        f"</div>"
    )


def build_email_bodies(articles: list[Article]) -> tuple[str, str]:
    """Return (plain_text_body, html_body) grouped by source."""
    grouped: dict[str, list[Article]] = defaultdict(list)
    for art in articles:
        grouped[art.source_name].append(art)

    # Plain-text version
    text_lines = [
        f"Phát hiện {len(articles)} tin tức / thông báo mới ({now_vn_iso()}):",
        "",
    ]
    for source_name, items in grouped.items():
        text_lines.append(f"=== {source_name} ({len(items)}) ===")
        for idx, art in enumerate(items, 1):
            date_part = f" [{art.published_at}]" if art.published_at else ""
            text_lines.append(f"{idx}. {art.title}{date_part}")
            if art.ai_summary:
                ai = art.ai_summary
                text_lines.append(f"   [Mức độ: {ai.importance}] Tóm tắt: {ai.tldr}")
                for point in ai.key_points:
                    text_lines.append(f"   - {point}")
                if ai.deadlines:
                    text_lines.append("   Mốc thời gian:")
                    for dl in ai.deadlines:
                        text_lines.append(f"     * {dl}")
                if ai.action_required:
                    text_lines.append(f"   Việc cần làm: {ai.action_required}")
            elif art.summary:
                text_lines.append(f"   Tóm tắt: {art.summary}")
            if art.attachments:
                text_lines.append("   Tài liệu đính kèm:")
                for label, att_url in art.attachments:
                    text_lines.append(f"     - {label}: {att_url}")
            text_lines.append(f"   Link gốc: {art.url}")
            text_lines.append("")
    plain_text = "\n".join(text_lines)

    # HTML version
    has_any_ai = any(a.ai_summary is not None for a in articles)
    sections_html: list[str] = []
    for source_name, items in grouped.items():
        color = html_lib.escape(items[0].source_color or "#2563EB")
        cards: list[str] = []
        for art in items:
            safe_title = html_lib.escape(art.title)
            safe_url = html_lib.escape(art.url, quote=True)
            safe_date = html_lib.escape(art.published_at or art.detected_at)

            badge_html = (
                f"&nbsp;&nbsp;{_importance_badge_html(art.ai_summary.importance)}"
                if art.ai_summary
                else ""
            )

            if art.ai_summary:
                body_section = _render_ai_card_html(art.ai_summary)
            elif art.summary:
                body_section = (
                    f'<p style="margin:8px 0 10px 0;color:#4B5563;font-size:14px;line-height:1.5;">'
                    f"{html_lib.escape(art.summary)}</p>"
                )
            else:
                body_section = ""

            attachments_html = _render_attachments_html(art.attachments)

            cards.append(
                f"""
                <div style="padding:16px 18px;margin-bottom:14px;border:1px solid #E5E7EB;
                            border-left:4px solid {color};border-radius:10px;background:#FFFFFF;">
                  <div style="font-size:12px;color:#6B7280;margin-bottom:6px;">
                    🕒 {safe_date}{badge_html}
                  </div>
                  <a href="{safe_url}" target="_blank"
                     style="color:#111827;font-size:16.5px;font-weight:700;text-decoration:none;line-height:1.4;">
                    {safe_title}
                  </a>
                  {body_section}
                  {attachments_html}
                  <div style="margin-top:10px;">
                    <a href="{safe_url}" target="_blank"
                       style="display:inline-block;padding:7px 14px;background:{color};color:#FFFFFF;
                              font-size:13px;font-weight:600;border-radius:6px;text-decoration:none;">
                      Xem bài gốc →
                    </a>
                  </div>
                </div>
                """
            )
        sections_html.append(
            f"""
            <div style="margin-bottom:22px;">
              <h2 style="margin:0 0 10px 0;font-size:14.5px;color:{color};text-transform:uppercase;letter-spacing:0.4px;">
                ● {html_lib.escape(source_name)} ({len(items)})
              </h2>
              {"".join(cards)}
            </div>
            """
        )

    subtitle_ai = " • Đã tóm tắt nội dung bằng AI" if has_any_ai else ""
    html_body = f"""<!DOCTYPE html>
<html lang="vi">
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#F3F4F6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:680px;margin:20px auto;background:#F9FAFB;border-radius:12px;overflow:hidden;border:1px solid #E5E7EB;">
    <div style="background:#111827;color:#FFFFFF;padding:18px 24px;">
      <div style="font-size:18px;font-weight:700;">🔔 Bản tin thông báo mới</div>
      <div style="font-size:13px;color:#D1D5DB;margin-top:4px;">
        Phát hiện {len(articles)} bài viết mới{subtitle_ai} • {html_lib.escape(now_vn_iso())}
      </div>
    </div>
    <div style="padding:20px 24px;">
      {"".join(sections_html)}
    </div>
    <div style="padding:12px 24px;background:#F3F4F6;color:#6B7280;font-size:12px;text-align:center;border-top:1px solid #E5E7EB;">
      Email được tổng hợp tự động bởi News Monitor Bot.
    </div>
  </div>
</body>
</html>
"""
    return plain_text, html_body


def send_email(subject: str, plain_text: str, html_body: str, config: EmailConfig) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = str(Header(subject, "utf-8"))
    msg["From"] = formataddr((str(Header(config.sender_name, "utf-8")), config.user))
    msg["To"] = ", ".join(config.recipients)

    msg.attach(MIMEText(plain_text, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    if config.port == 465:
        with smtplib.SMTP_SSL(config.host, config.port, timeout=20) as server:
            server.login(config.user, config.password)
            server.sendmail(config.user, config.recipients, msg.as_string())
    else:
        with smtplib.SMTP(config.host, config.port, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(config.user, config.password)
            server.sendmail(config.user, config.recipients, msg.as_string())

    logger.info("Đã gửi email tới %s: %s", ", ".join(config.recipients), subject)


def send_articles_email(articles: list[Article], config: EmailConfig | None = None) -> None:
    if not articles:
        return
    cfg = config or EmailConfig.from_env()
    subject = build_email_subject(articles)
    plain_text, html_body = build_email_bodies(articles)
    send_email(subject, plain_text, html_body, cfg)


def send_test_email(config: EmailConfig | None = None) -> None:
    cfg = config or EmailConfig.from_env()
    sample = Article(
        source_id="test",
        source_name="News Monitor Bot - Kiểm tra kết nối",
        title="Kết nối gửi email & giao diện tóm tắt AI hoạt động thành công!",
        url="https://github.com",
        published_at=now_vn_iso(),
        source_color="#16A34A",
        ai_summary=AISummary(
            tldr="Bot đã kết nối thành công với Gmail SMTP và sẵn sàng tự động đọc, tóm tắt các thông báo mới từ IIG và PTIT.",
            key_points=[
                "Tự động quét tin mới mỗi 5 phút 24/7 trên GitHub Actions.",
                "Đọc toàn bộ nội dung bài viết, rút gọn ý chính và trích xuất tài liệu PDF đính kèm.",
                "Tự động tách nổi bật các mốc thời gian / hạn chót quan trọng.",
            ],
            deadlines=["Mỗi 5 phút: Bot tự kiểm tra tin mới hoàn toàn tự động"],
            action_required="Không cần thao tác gì thêm — bạn chỉ việc đợi nhận bản tin khi có thông báo mới!",
            importance="Cao",
        ),
    )
    send_articles_email([sample], cfg)
