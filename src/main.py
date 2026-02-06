from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.types import FSInputFile

from src.agent.orchestrator import AgentOrchestrator
from src.bot.handlers import BotServices, build_router
from src.bot.middleware import AllowedUserMiddleware
from src.config import settings
from src.db.session import init_db
from src.jobs.maintenance import MaintenanceJobs
from src.jobs.scheduler import SchedulerService
from src.llm.gemini_client import GeminiClient
from src.llm.groq_client import GroqClient
from src.support.error_triage import ErrorTriageService
from src.support.solver import AdminSolveService
from src.tools.camera import CameraService
from src.tools.exec import SandboxExecutor
from src.utils.logging import configure_logging
from src.utils.paths import PathResolver


logger = logging.getLogger(__name__)


async def run() -> None:
    configure_logging(settings.log_level)
    settings.validate_runtime()

    resolver = PathResolver(settings)
    resolver.ensure_base_layout()
    await init_db()

    bot = Bot(token=settings.telegram_bot_token)

    groq_client = GroqClient(settings.groq_api_key)
    gemini_client = GeminiClient(
        api_keys=settings.all_gemini_api_keys(),
        max_retries=settings.gemini_max_retries,
        base_retry_delay_seconds=settings.gemini_retry_base_delay_seconds,
        min_quota_cooldown_seconds=settings.gemini_quota_cooldown_seconds,
        daily_quota_cooldown_seconds=settings.gemini_daily_quota_cooldown_seconds,
    )
    sandbox_executor = SandboxExecutor(settings, resolver)
    solve_service = AdminSolveService(
        bot=bot,
        groq=groq_client,
        admin_user_ids=settings.admin_user_ids,
    )

    error_triage = ErrorTriageService(
        resolver=resolver,
        groq=groq_client,
        bot=bot,
        admin_user_ids=settings.admin_user_ids,
        auto_solver_callback=solve_service.handle_auto_problem,
    )

    maintenance_jobs = MaintenanceJobs(resolver=resolver, groq=groq_client)
    scheduler_service = SchedulerService(
        bot=bot,
        maintenance_jobs=maintenance_jobs,
        error_triage=error_triage,
    )

    camera_service = CameraService(resolver=resolver)

    async def send_camera_photo(chat_id: int, file_path: str, caption: str) -> None:
        photo = FSInputFile(file_path)
        await bot.send_photo(chat_id=chat_id, photo=photo, caption=caption or None)

    orchestrator = AgentOrchestrator(
        gemini=gemini_client,
        groq=groq_client,
        resolver=resolver,
        sandbox_executor=sandbox_executor,
        schedule_reminder_callback=scheduler_service.schedule_reminder,
        camera_service=camera_service,
        send_photo_callback=send_camera_photo,
    )

    services = BotServices(
        bot=bot,
        resolver=resolver,
        orchestrator=orchestrator,
        error_triage=error_triage,
        solver=solve_service,
    )

    router = build_router(services)
    router.message.middleware(
        AllowedUserMiddleware(
            allowed_user_ids=set(settings.allowed_user_ids),
            admin_user_ids=set(settings.admin_user_ids),
        )
    )
    router.callback_query.middleware(
        AllowedUserMiddleware(
            allowed_user_ids=set(settings.allowed_user_ids),
            admin_user_ids=set(settings.admin_user_ids),
        )
    )

    dp = Dispatcher()
    dp.include_router(router)

    async def on_error(event: Any) -> bool:
        update_obj = getattr(event, "update", None)
        try:
            update_json = update_obj.model_dump_json(exclude_none=True) if update_obj else "(no update payload)"
        except Exception:
            update_json = str(update_obj)
        await error_triage.handle_exception(
            getattr(event, "exception", RuntimeError("unknown aiogram error")),
            context=f"aiogram update error: {update_json}",
        )
        return True

    dp.errors.register(on_error)

    loop = asyncio.get_running_loop()

    def loop_exception_handler(_loop, context):
        exc = context.get("exception") or RuntimeError(context.get("message", "loop exception"))
        asyncio.create_task(error_triage.handle_exception(exc, context=str(context)))

    loop.set_exception_handler(loop_exception_handler)

    await scheduler_service.start()

    logger.info("Starting Telegram long polling")
    try:
        await dp.start_polling(bot, allowed_updates=["message", "callback_query"])
    finally:
        await scheduler_service.shutdown()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(run())
