from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from aiogram import Bot, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from src.agent.orchestrator import AgentOrchestrator
from src.config import settings
from src.db.session import get_session
from src.memory.store import MemoryStore
from src.tools.exec import run_admin_command
from src.tools.files import FileService
from src.tools.reminders import parse_reminder_time
from src.utils.paths import PathResolver


@dataclass
class BotServices:
    bot: Bot
    resolver: PathResolver
    orchestrator: AgentOrchestrator
    schedule_reminder: Callable[[int, datetime], None]


MAX_INLINE_ATTACHMENT_BYTES = 5 * 1024 * 1024
MAX_INLINE_AUDIO_BYTES = 15 * 1024 * 1024
MAX_INLINE_ATTACHMENT_COUNT = 4
DEFAULT_AUDIO_PROMPT = "Transcribe this audio message and treat the transcript as the user's message, then respond."


def build_router(services: BotServices) -> Router:
    router = Router(name="main-message-router")

    @router.message()
    async def message_router(message: Message, is_admin: bool = False) -> None:
        user = message.from_user
        if user is None:
            return

        user_id = int(user.id)
        services.resolver.ensure_user_layout(user_id)

        if message.text and message.text.startswith("/"):
            await _handle_command(message, user_id, is_admin, services)
            return

        if message.photo or message.document or message.voice or message.audio:
            await _handle_attachment_message(message, user_id, services)
            return

        if message.text:
            await _handle_plain_text(message, user_id, services)

    return router


async def _handle_plain_text(message: Message, user_id: int, services: BotServices) -> None:
    text = (message.text or "").strip()
    if not text:
        return

    await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    pending_note = None
    pending_ids: list[int] = []
    pending_inline_bytes: list[tuple[bytes, str]] = []
    async with get_session() as session:
        store = MemoryStore(session)
        await store.ensure_user(user_id, is_admin=user_id in settings.admin_user_ids)
        pending = await store.list_pending_attachments(user_id)
        if pending:
            paths = [item.file_path for item in pending]
            pending_ids = [item.id for item in pending]
            pending_note = f"Pending attachments: {paths}"
            pending_inline_bytes = await _load_pending_inline_media_bytes(pending)

    allow_shared = _explicit_shared_requested(text)
    result = await services.orchestrator.handle_user_text(
        telegram_user_id=user_id,
        text=text,
        allow_shared=allow_shared,
        pending_attachment_note=pending_note,
        pending_attachment_ids=pending_ids,
        inline_attachment_bytes=pending_inline_bytes,
    )
    await _reply_chunked(message, result.text)


async def _handle_attachment_message(message: Message, user_id: int, services: BotServices) -> None:
    file_id: Optional[str] = None
    file_name = "attachment.bin"
    mime = "application/octet-stream"

    if message.photo:
        photo = message.photo[-1]
        file_id = photo.file_id
        file_name = f"photo_{photo.file_unique_id}.jpg"
        mime = "image/jpeg"
    elif message.document:
        doc = message.document
        file_id = doc.file_id
        file_name = doc.file_name or "document.bin"
        mime = doc.mime_type or mime
    elif message.voice:
        voice = message.voice
        file_id = voice.file_id
        file_name = f"voice_{voice.file_unique_id}.ogg"
        mime = voice.mime_type or "audio/ogg"
    elif message.audio:
        audio = message.audio
        file_id = audio.file_id
        file_name = audio.file_name or f"audio_{audio.file_unique_id}.bin"
        mime = audio.mime_type or "audio/mpeg"

    if file_id is None:
        return

    normalized_mime = _normalize_attachment_mime(mime)
    is_audio_message = normalized_mime.startswith("audio/")

    cache_path = services.resolver.cache_file_path(user_id, file_name)
    tg_file = await message.bot.get_file(file_id)
    await message.bot.download(file=tg_file, destination=cache_path)

    async with get_session() as session:
        store = MemoryStore(session)
        await store.ensure_user(user_id, is_admin=user_id in settings.admin_user_ids)
        await store.add_file_metadata(user_id, "user", str(cache_path), "attachment")

        caption = (message.caption or "").strip()
        if not caption and not is_audio_message:
            await store.add_pending_attachment(
                telegram_user_id=user_id,
                file_path=str(cache_path),
                file_name=file_name,
                mime_type=mime,
            )
            return

    bytes_payloads: list[tuple[bytes, str]] = []
    try:
        max_inline_size = _max_inline_bytes_for_mime(normalized_mime)
        if _is_inline_multimodal_mime(normalized_mime) and cache_path.stat().st_size <= max_inline_size:
            payload = await asyncio.to_thread(cache_path.read_bytes)
            bytes_payloads.append((payload, normalized_mime))
    except Exception:
        pass

    await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    text_prompt = caption if caption else (DEFAULT_AUDIO_PROMPT if is_audio_message else "")
    if not text_prompt:
        return

    allow_shared = _explicit_shared_requested(text_prompt)
    result = await services.orchestrator.handle_user_text(
        telegram_user_id=user_id,
        text=text_prompt,
        allow_shared=allow_shared,
        pending_attachment_note=f"Attachment path: {cache_path}",
        inline_attachment_bytes=bytes_payloads,
    )
    await _reply_chunked(message, result.text)


async def _handle_command(message: Message, user_id: int, is_admin: bool, services: BotServices) -> None:
    raw = (message.text or "").strip()
    parts = raw.split(maxsplit=1)
    cmd = parts[0].split("@")[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/help":
        await message.answer(
            "Commands:\n"
            "/help\n"
            "/status\n"
            "/run <shell command> (admin)\n"
            "/files\n"
            "/save <relative_cache_path> <dest_rel_path>\n"
            "/shared put <rel_user_path> <rel_shared_path>\n"
            "/shared get <rel_shared_path> <rel_user_path>\n"
            "/remind <when> <text>\n"
            "/remind_user <user_id> <when> <text>"
        )
        return

    if cmd == "/status":
        usage = services.resolver.user_disk_usage_bytes(user_id)
        async with get_session() as session:
            store = MemoryStore(session)
            count = await store.message_count(user_id)
        await message.answer(
            "Status: OK\n"
            f"User ID: {user_id}\n"
            f"Messages stored: {count}\n"
            f"User disk usage: {usage} bytes\n"
            f"Server time: {datetime.utcnow().isoformat()}Z"
        )
        return

    if cmd == "/run":
        if not is_admin:
            await message.answer("Admin only.")
            return
        if not rest:
            await message.answer("Usage: /run <shell command>")
            return
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        result = await run_admin_command(rest)
        text = (
            f"exit_code={result['exit_code']} timed_out={result['timed_out']}\n"
            f"STDOUT:\n{result['stdout'] or '(empty)'}\n\n"
            f"STDERR:\n{result['stderr'] or '(empty)'}"
        )
        await _reply_chunked(message, text)
        return

    if cmd == "/files":
        async with get_session() as session:
            store = MemoryStore(session)
            file_service = FileService(services.resolver, store)
            recent = await file_service.list_recent_user_files(user_id, limit=20)
        if not recent:
            await message.answer("No tracked files yet.")
            return
        lines = ["Recent files:"]
        for item in recent:
            lines.append(
                f"- [{item['scope']}] {item['path']} ({item['kind']}, {item['status']}, exists={item['exists']})"
            )
        await _reply_chunked(message, "\n".join(lines))
        return

    if cmd == "/save":
        args = rest.split(maxsplit=1)
        if len(args) != 2:
            await message.answer("Usage: /save <relative_cache_path> <dest_rel_path>")
            return
        src_rel, dst_rel = args
        async with get_session() as session:
            store = MemoryStore(session)
            file_service = FileService(services.resolver, store)
            result = await file_service.file_move(user_id, src_rel, dst_rel, location="user")
        await message.answer(str(result))
        return

    if cmd == "/shared":
        args = rest.split(maxsplit=2)
        if len(args) != 3 or args[0] not in {"put", "get"}:
            await message.answer("Usage: /shared put <rel_user_path> <rel_shared_path> or /shared get <rel_shared_path> <rel_user_path>")
            return
        mode, left, right = args
        async with get_session() as session:
            store = MemoryStore(session)
            file_service = FileService(services.resolver, store)
            if mode == "put":
                result = await file_service.copy_user_to_shared(user_id, left, right)
            else:
                result = await file_service.copy_shared_to_user(user_id, left, right)
        await message.answer(str(result))
        return

    if cmd == "/remind":
        when_raw, reminder_text = _split_when_and_text(rest)
        if not when_raw or not reminder_text:
            await message.answer("Usage: /remind <when> <text>")
            return
        try:
            run_at = parse_reminder_time(when_raw, settings.timezone)
        except Exception as exc:
            await message.answer(f"Invalid reminder time: {exc}")
            return
        async with get_session() as session:
            store = MemoryStore(session)
            reminder = await store.create_reminder(
                creator_user_id=user_id,
                target_user_id=user_id,
                scheduled_for=run_at,
                text=reminder_text,
            )
        services.schedule_reminder(reminder.id, run_at)
        await message.answer(f"Reminder set for {run_at.isoformat()}")
        return

    if cmd == "/remind_user":
        if not is_admin:
            await message.answer("Admin only.")
            return
        target, when_raw, reminder_text = _split_remind_user_args(rest)
        if target is None or not when_raw or not reminder_text:
            await message.answer("Usage: /remind_user <user_id> <when> <text>")
            return
        try:
            run_at = parse_reminder_time(when_raw, settings.timezone)
        except Exception as exc:
            await message.answer(f"Invalid reminder time: {exc}")
            return
        async with get_session() as session:
            store = MemoryStore(session)
            reminder = await store.create_reminder(
                creator_user_id=user_id,
                target_user_id=target,
                scheduled_for=run_at,
                text=reminder_text,
            )
        services.schedule_reminder(reminder.id, run_at)
        await message.answer(f"Reminder for user {target} set for {run_at.isoformat()}")
        return

    await message.answer("Unknown command. Use /help")


async def _reply_chunked(message: Message, text: str, chunk_size: int = 3800) -> None:
    if len(text) <= chunk_size:
        await message.answer(text)
        return

    for i in range(0, len(text), chunk_size):
        await message.answer(text[i : i + chunk_size])


def _explicit_shared_requested(text: str) -> bool:
    return bool(re.search(r"\bshared\b", text, flags=re.IGNORECASE))


def _split_when_and_text(raw: str) -> tuple[str, str]:
    value = raw.strip()
    if not value:
        return "", ""

    if value.lower().startswith("in "):
        parts = value.split(maxsplit=2)
        if len(parts) < 3:
            return "", ""
        return f"{parts[0]} {parts[1]}", parts[2]

    if value.lower().startswith("tomorrow"):
        parts = value.split(maxsplit=2)
        if len(parts) < 3:
            return "", ""
        return f"{parts[0]} {parts[1]}", parts[2]

    parts = value.split(maxsplit=1)
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[1]


def _split_remind_user_args(raw: str) -> tuple[Optional[int], str, str]:
    parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        return None, "", ""
    try:
        target = int(parts[0])
    except ValueError:
        return None, "", ""

    when_raw, reminder_text = _split_when_and_text(parts[1])
    return target, when_raw, reminder_text


async def _load_pending_inline_media_bytes(pending_rows) -> list[tuple[bytes, str]]:
    out: list[tuple[bytes, str]] = []
    for row in pending_rows:
        if len(out) >= MAX_INLINE_ATTACHMENT_COUNT:
            break
        mime = _normalize_attachment_mime(row.mime_type or "")
        if not _is_inline_multimodal_mime(mime):
            continue
        path = Path(row.file_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            if path.stat().st_size > _max_inline_bytes_for_mime(mime):
                continue
            payload = await asyncio.to_thread(path.read_bytes)
            out.append((payload, mime))
        except Exception:
            continue
    return out


def _is_inline_multimodal_mime(mime: str) -> bool:
    lowered = (mime or "").strip().lower()
    return lowered.startswith("image/") or lowered.startswith("audio/")


def _max_inline_bytes_for_mime(mime: str) -> int:
    lowered = (mime or "").strip().lower()
    if lowered.startswith("audio/"):
        return MAX_INLINE_AUDIO_BYTES
    return MAX_INLINE_ATTACHMENT_BYTES


def _normalize_attachment_mime(mime: str) -> str:
    lowered = (mime or "").strip().lower()
    if not lowered:
        return "application/octet-stream"
    return lowered
