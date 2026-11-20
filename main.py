from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.models import Article
from src.notifier import build_email_bodies, send_test_email
from src.runner import run_once
from src.summarizer import AISummarizer


def configure_logging(verbose: bool = False) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bot quét tin tức & thông báo tự động từ website, tóm tắt bằng AI và gửi email tức thì."
    )
    parser.add_argument(
        "--config",
        default="config/sites.yaml",
        help="Đường dẫn file cấu hình danh sách web (mặc định: config/sites.yaml)",
    )
    parser.add_argument(
        "--state",
        default="state/seen.json",
        help="Đường dẫn file lưu lịch sử bài đã quét (mặc định: state/seen.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chạy thử để kiểm tra quét web, KHÔNG gửi email và KHÔNG lưu trạng thái.",
    )
    parser.add_argument(
        "--notify-initial",
        action="store_true",
        help="Gửi email luôn các bài ở lần chạy đầu tiên của một nguồn (dùng khi muốn test).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Tắt bước tóm tắt bằng AI (chỉ gửi tiêu đề & đoạn trích gốc).",
    )
    parser.add_argument(
        "--preview-summary",
        metavar="URL",
        default="",
        help="Đọc thử 1 đường link bài viết bất kỳ và in bản tóm tắt AI ra màn hình.",
    )
    parser.add_argument(
        "--test-email",
        action="store_true",
        help="Gửi 1 email kiểm tra cấu hình SMTP rồi thoát.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Chạy lặp liên tục trên máy cá nhân / VPS thay vì thoát sau 1 lần.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Chu kỳ quét tính bằng giây khi bật --loop (mặc định: 300 = 5 phút).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Hiển thị log chi tiết hơn để debug.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    configure_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    if args.test_email:
        send_test_email()
        logger.info("Đã gửi email kiểm tra thành công!")
        return 0

    if args.preview_summary:
        ai = AISummarizer.from_env()
        if ai is None:
            logger.error("Chưa cấu hình GEMINI_API_KEY hoặc AI_API_KEY trong file .env!")
            return 1
        art = Article(
            source_id="preview",
            source_name="Xem trước tóm tắt AI",
            title=args.preview_summary,
            url=args.preview_summary,
        )
        res = ai.summarize_article(art)
        if res is None:
            logger.error("Không thể tóm tắt bài viết này.")
            return 1
        plain, _ = build_email_bodies([art])
        print("\n" + plain)
        return 0

    if not Path(args.config).exists():
        logger.error("Không tìm thấy file cấu hình: %s", args.config)
        return 1

    if args.loop:
        logger.info("Bắt đầu chế độ chạy liên tục (mỗi %d giây). Nhấn Ctrl+C để dừng.", args.interval)
        while True:
            try:
                run_once(
                    config_path=args.config,
                    state_path=args.state,
                    dry_run=args.dry_run,
                    notify_initial=args.notify_initial,
                    enable_ai=not args.no_ai,
                )
            except Exception as exc:
                logger.exception("Lỗi trong vòng lặp quét: %s", exc)
            time.sleep(max(10, args.interval))

    summary = run_once(
        config_path=args.config,
        state_path=args.state,
        dry_run=args.dry_run,
        notify_initial=args.notify_initial,
        enable_ai=not args.no_ai,
    )
    if summary.sources_checked > 0 and summary.sources_failed == summary.sources_checked:
        logger.error("Tất cả các nguồn đều gặp lỗi trong lần quét này!")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
