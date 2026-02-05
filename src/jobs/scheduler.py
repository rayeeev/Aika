from __future__ import annotations

import logging
from datetime import datetime, timezone

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import settings
from src.db.session import get_session
from src.jobs.maintenance import MaintenanceJobs
from src.memory.store import MemoryStore
from src.support.error_triage import ErrorTriageService
from src.tools.reminders import schedule_one_off_job


logger = logging.getLogger(__name__)


class SchedulerService:
    def __init__(
        self,
        bot: Bot,
        maintenance_jobs: MaintenanceJobs,
        error_triage: ErrorTriageService,
    ) -> None:
        self.bot = bot
        self.maintenance_jobs = maintenance_jobs
        self.error_triage = error_triage
        self.scheduler = AsyncIOScheduler(timezone=settings.timezone)

    async def start(self) -> None:
        self._register_periodic_jobs()
        self.scheduler.start()
        await self.hydrate_pending_reminders()

    async def shutdown(self) -> None:
        self.scheduler.shutdown(wait=False)

    def schedule_reminder(self, reminder_id: int, run_at_utc: datetime) -> None:
        schedule_one_off_job(self.scheduler, reminder_id, run_at_utc, self.send_one_reminder)

    async def hydrate_pending_reminders(self) -> None:
        async with get_session() as session:
            store = MemoryStore(session)
            reminders = await store.pending_reminders()
            now = datetime.now(timezone.utc)
            for reminder in reminders:
                if reminder.scheduled_for <= now:
                    self.scheduler.add_job(
                        self.send_one_reminder,
                        trigger="date",
                        run_date=now,
                        args=[reminder.id],
                        id=f"reminder-{reminder.id}",
                        replace_existing=True,
                    )
                else:
                    self.schedule_reminder(reminder.id, reminder.scheduled_for)

    async def send_one_reminder(self, reminder_id: int) -> None:
        try:
            async with get_session() as session:
                store = MemoryStore(session)
                reminder = await store.get_reminder(reminder_id)
                if reminder is None or reminder.status != "pending":
                    return

                await self.bot.send_message(
                    chat_id=reminder.target_user_id,
                    text=f"⏰ Reminder: {reminder.text}",
                )
                await store.mark_reminder_sent(reminder_id)
        except Exception as exc:
            logger.exception("Failed to send reminder %s", reminder_id)
            await self.error_triage.handle_exception(
                exc,
                context=f"send_one_reminder(reminder_id={reminder_id})",
            )

    def _register_periodic_jobs(self) -> None:
        self.scheduler.add_job(
            self._run_daily_summaries_job,
            CronTrigger(hour=4, minute=0),
            id="daily-summary-job",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._run_weekly_major_job,
            CronTrigger(day_of_week="sun", hour=5, minute=0),
            id="weekly-major-job",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._run_cache_cleanup_job,
            CronTrigger(minute=0),
            id="cache-cleanup-job",
            replace_existing=True,
        )

    async def _run_daily_summaries_job(self) -> None:
        await self._run_safe("daily_summary_job", self.maintenance_jobs.run_daily_summaries)

    async def _run_weekly_major_job(self) -> None:
        await self._run_safe("weekly_major_memory_job", self.maintenance_jobs.run_weekly_major_memory)

    async def _run_cache_cleanup_job(self) -> None:
        await self._run_safe("cache_cleanup_job", self.maintenance_jobs.cleanup_cache)

    async def _run_safe(self, job_name: str, fn) -> None:
        try:
            await fn()
        except Exception as exc:
            logger.exception("Scheduled job failed: %s", job_name)
            await self.error_triage.handle_exception(exc, context=f"scheduler job failed: {job_name}")
