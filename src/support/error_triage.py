from __future__ import annotations

import asyncio
import logging
import re
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional

from aiogram import Bot

from src.llm.groq_client import GroqClient
from src.utils.paths import PathResolver


logger = logging.getLogger(__name__)

_FILE_RE = re.compile(r'File "([^"]+)", line (\d+)')


AutoSolverCallback = Callable[[Path, str, str, str], Awaitable[None]]


class ErrorTriageService:
    def __init__(
        self,
        resolver: PathResolver,
        groq: GroqClient,
        bot: Bot,
        admin_user_ids: list[int],
        notify_cooldown_seconds: int = 300,
        auto_solver_callback: Optional[AutoSolverCallback] = None,
    ) -> None:
        self.resolver = resolver
        self.groq = groq
        self.bot = bot
        self.admin_user_ids = admin_user_ids
        self.notify_cooldown = timedelta(seconds=notify_cooldown_seconds)
        self._last_notify_at: Optional[datetime] = None
        self._last_auto_solve_at: Optional[datetime] = None
        self.auto_solver_callback = auto_solver_callback

    async def handle_exception(
        self,
        exc: BaseException,
        context: str,
        extra_files: Optional[Iterable[str]] = None,
    ) -> Path:
        stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        touched = _extract_files_from_stack(stack)
        if extra_files:
            touched.extend(str(x) for x in extra_files)

        snippets = await asyncio.to_thread(_load_snippets, touched)

        triage_text = "Groq triage unavailable"
        try:
            triage_text = await self.groq.triage_error(stack_trace=stack, snippets=snippets)
        except Exception as triage_exc:
            logger.exception("Groq triage failed: %s", triage_exc)

        report_path = await asyncio.to_thread(
            self._write_report,
            context=context,
            stack=stack,
            snippets=snippets,
            triage=triage_text,
        )

        await self._maybe_notify_admins(report_path)
        if self.auto_solver_callback and not _looks_like_gemini_quota_issue(context=context, stack=stack):
            now = datetime.now(timezone.utc)
            if self._last_auto_solve_at is None or now - self._last_auto_solve_at >= self.notify_cooldown:
                try:
                    await self.auto_solver_callback(report_path, context, stack, triage_text)
                    self._last_auto_solve_at = now
                except Exception:
                    logger.exception("Auto-solver callback failed for report %s", report_path)
        return report_path

    def _write_report(self, context: str, stack: str, snippets: str, triage: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = self.resolver.admin_dir / "error_reports" / f"{stamp}.md"

        body = (
            f"# Error Report {stamp}\n\n"
            f"## Context\n{context}\n\n"
            "## Stack Trace\n"
            "```text\n"
            f"{stack}\n"
            "```\n\n"
            "## Relevant Snippets\n"
            "```text\n"
            f"{snippets}\n"
            "```\n\n"
            "## Groq Root Cause + Patch Suggestions\n"
            f"{triage}\n"
        )

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(body, encoding="utf-8")
        return report_path

    async def _maybe_notify_admins(self, report_path: Path) -> None:
        now = datetime.now(timezone.utc)
        if self._last_notify_at and now - self._last_notify_at < self.notify_cooldown:
            return

        message = f"Error occurred; report saved at {report_path}"
        for admin_id in self.admin_user_ids:
            try:
                await self.bot.send_message(chat_id=admin_id, text=message)
            except Exception:
                logger.exception("Failed to notify admin %s", admin_id)

        self._last_notify_at = now

    def list_recent_reports(self, limit: int = 10) -> list[Path]:
        report_dir = self.resolver.admin_dir / "error_reports"
        if not report_dir.exists():
            return []
        files = [path for path in report_dir.glob("*.md") if path.is_file()]
        files.sort(key=lambda path: path.name, reverse=True)
        return files[: max(1, limit)]


def _extract_files_from_stack(stack: str) -> list[str]:
    files: list[str] = []
    for match in _FILE_RE.finditer(stack):
        path = match.group(1)
        if path.startswith("<"):
            continue
        files.append(path)
    deduped = list(dict.fromkeys(files))
    return deduped[:8]


def _load_snippets(paths: Iterable[str], max_files: int = 6, max_lines: int = 120) -> str:
    chunks: list[str] = []
    count = 0
    for raw_path in paths:
        if count >= max_files:
            break
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        header = f"== {path} =="
        excerpt = "\n".join(lines[:max_lines])
        chunks.append(f"{header}\n{excerpt}")
        count += 1

    if not chunks:
        return "(no snippets available)"
    return "\n\n".join(chunks)


def _looks_like_gemini_quota_issue(context: str, stack: str) -> bool:
    joined = f"{context}\n{stack}".lower()
    signals = (
        "resource_exhausted",
        "quota exceeded",
        "generate_content_free_tier_requests",
        "generaterequestsperdayperprojectpermodel",
        "geminiquotaexceedederror",
    )
    return any(signal in joined for signal in signals)
