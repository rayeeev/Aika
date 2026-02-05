from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.config import settings
from src.db.session import get_session
from src.llm.groq_client import GroqClient
from src.memory.store import MemoryStore
from src.memory.summarizer import MemorySummarizer
from src.utils.paths import PathResolver


logger = logging.getLogger(__name__)


class MaintenanceJobs:
    def __init__(self, resolver: PathResolver, groq: GroqClient) -> None:
        self.resolver = resolver
        self.groq = groq

    async def run_daily_summaries(self) -> None:
        local_today = datetime.now(ZoneInfo(settings.timezone)).date()
        target_day = local_today - timedelta(days=1)

        async with get_session() as session:
            store = MemoryStore(session)
            summarizer = MemorySummarizer(store, self.groq)
            users = await store.list_known_users()
            for uid in users:
                text = await summarizer.run_daily_summary(uid, target_day)
                status = "ok" if text else "skipped"
                await store.insert_job_log(
                    job_name="daily_summary",
                    status=status,
                    details=f"day={target_day.isoformat()}",
                    telegram_user_id=uid,
                )

    async def run_weekly_major_memory(self) -> None:
        local_today = datetime.now(ZoneInfo(settings.timezone)).date()

        async with get_session() as session:
            store = MemoryStore(session)
            summarizer = MemorySummarizer(store, self.groq)
            users = await store.list_known_users()
            for uid in users:
                text = await summarizer.run_weekly_major_memory(uid, local_today)
                status = "ok" if text else "skipped"
                await store.insert_job_log(
                    job_name="weekly_major_memory",
                    status=status,
                    details=f"day={local_today.isoformat()}",
                    telegram_user_id=uid,
                )

    async def cleanup_cache(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.cache_ttl_hours)
        deleted = 0

        async with get_session() as session:
            store = MemoryStore(session)
            for user_root in self.resolver.user_dir.glob("*"):
                if not user_root.is_dir():
                    continue
                try:
                    user_id = int(user_root.name)
                except ValueError:
                    continue
                cache_dir = user_root / "cache"
                if not cache_dir.exists():
                    continue
                for path in cache_dir.rglob("*"):
                    if not path.is_file():
                        continue
                    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    if modified >= cutoff:
                        continue
                    path.unlink(missing_ok=True)
                    deleted += 1
                    await store.mark_file_deleted(user_id, str(path))
                    await store.consume_pending_attachment_by_path(str(path))

            await store.insert_job_log(
                job_name="cache_cleanup",
                status="ok",
                details=f"deleted={deleted}",
            )

        logger.info("Cache cleanup complete: deleted=%s", deleted)
