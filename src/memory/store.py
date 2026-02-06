from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Sequence

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from zoneinfo import ZoneInfo

from src.config import settings
from src.db.models import (
    CompressedMemory,
    DailySummary,
    Exchange,
    FileMetadata,
    JobLog,
    MajorMemory,
    Message,
    PendingAttachment,
    Reminder,
    User,
)


@dataclass
class ExchangeBundle:
    exchange_id: int
    user_text: str
    assistant_text: str


class MemoryStore:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def ensure_user(self, telegram_user_id: int, is_admin: bool = False) -> None:
        user = await self.session.get(User, telegram_user_id)
        if user is None:
            self.session.add(User(telegram_user_id=telegram_user_id, is_admin=is_admin))
            await self.session.flush()
        else:
            user.last_seen_at = datetime.now(timezone.utc)
            user.is_admin = bool(user.is_admin or is_admin)
            await self.session.flush()

    async def store_user_message(self, telegram_user_id: int, text: str) -> int:
        msg = Message(telegram_user_id=telegram_user_id, role="user", content=text)
        self.session.add(msg)
        await self.session.flush()

        exchange = Exchange(telegram_user_id=telegram_user_id, user_message_id=msg.id)
        self.session.add(exchange)
        await self.session.flush()

        msg.exchange_id = exchange.id
        await self.session.flush()
        return exchange.id

    async def store_assistant_message(self, telegram_user_id: int, text: str, exchange_id: int) -> int:
        msg = Message(
            telegram_user_id=telegram_user_id,
            role="assistant",
            content=text,
            exchange_id=exchange_id,
        )
        self.session.add(msg)
        await self.session.flush()

        exchange = await self.session.get(Exchange, exchange_id)
        if exchange:
            exchange.assistant_message_id = msg.id
            await self.session.flush()
        return msg.id

    async def store_tool_message(self, telegram_user_id: int, text: str) -> int:
        msg = Message(telegram_user_id=telegram_user_id, role="tool", content=text)
        self.session.add(msg)
        await self.session.flush()
        return msg.id

    async def get_recent_exchanges(self, telegram_user_id: int, limit: int = 3) -> list[ExchangeBundle]:
        rows = (
            (
                await self.session.execute(
                    select(Exchange)
                    .where(
                        and_(
                            Exchange.telegram_user_id == telegram_user_id,
                            Exchange.assistant_message_id.is_not(None),
                        )
                    )
                    .order_by(Exchange.id.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )

        rows = list(reversed(rows))
        bundles: list[ExchangeBundle] = []
        for row in rows:
            user_msg = await self.session.get(Message, row.user_message_id)
            assistant_msg = await self.session.get(Message, row.assistant_message_id)
            if user_msg and assistant_msg:
                bundles.append(
                    ExchangeBundle(
                        exchange_id=row.id,
                        user_text=user_msg.content,
                        assistant_text=assistant_msg.content,
                    )
                )
        return bundles

    async def oldest_uncompressed_exchange_outside_recent(
        self,
        telegram_user_id: int,
        keep_recent: int = 3,
    ) -> Optional[ExchangeBundle]:
        rows = (
            (
                await self.session.execute(
                    select(Exchange)
                    .where(
                        and_(
                            Exchange.telegram_user_id == telegram_user_id,
                            Exchange.assistant_message_id.is_not(None),
                        )
                    )
                    .order_by(Exchange.id.desc())
                )
            )
            .scalars()
            .all()
        )

        if len(rows) <= keep_recent:
            return None

        older = rows[keep_recent:]
        for row in reversed(older):
            if row.is_compressed:
                continue
            user_msg = await self.session.get(Message, row.user_message_id)
            assistant_msg = await self.session.get(Message, row.assistant_message_id)
            if user_msg and assistant_msg:
                return ExchangeBundle(
                    exchange_id=row.id,
                    user_text=user_msg.content,
                    assistant_text=assistant_msg.content,
                )
        return None

    async def mark_exchange_compressed(self, exchange_id: int) -> None:
        exchange = await self.session.get(Exchange, exchange_id)
        if exchange:
            exchange.is_compressed = True
            await self.session.flush()

    async def add_compressed_sentence(self, telegram_user_id: int, exchange_id: int, sentence: str) -> None:
        self.session.add(
            CompressedMemory(
                telegram_user_id=telegram_user_id,
                source_exchange_id=exchange_id,
                sentence=sentence,
            )
        )
        await self.session.flush()

    async def get_compressed_sentences(self, telegram_user_id: int, limit: int = 48) -> list[str]:
        rows = (
            (
                await self.session.execute(
                    select(CompressedMemory)
                    .where(CompressedMemory.telegram_user_id == telegram_user_id)
                    .order_by(CompressedMemory.id.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        rows = list(reversed(rows))
        return [item.sentence for item in rows]

    async def add_pending_attachment(
        self,
        telegram_user_id: int,
        file_path: str,
        file_name: str,
        mime_type: Optional[str],
    ) -> None:
        self.session.add(
            PendingAttachment(
                telegram_user_id=telegram_user_id,
                file_path=file_path,
                file_name=file_name,
                mime_type=mime_type,
            )
        )
        await self.session.flush()

    async def list_pending_attachments(self, telegram_user_id: int) -> list[PendingAttachment]:
        rows = (
            (
                await self.session.execute(
                    select(PendingAttachment)
                    .where(
                        and_(
                            PendingAttachment.telegram_user_id == telegram_user_id,
                            PendingAttachment.consumed.is_(False),
                        )
                    )
                    .order_by(PendingAttachment.id.asc())
                )
            )
            .scalars()
            .all()
        )
        return rows

    async def consume_pending_attachments(self, telegram_user_id: int, attachment_ids: Sequence[int]) -> None:
        if not attachment_ids:
            return
        await self.session.execute(
            update(PendingAttachment)
            .where(
                and_(
                    PendingAttachment.telegram_user_id == telegram_user_id,
                    PendingAttachment.id.in_(list(attachment_ids)),
                )
            )
            .values(consumed=True)
        )
        await self.session.flush()

    async def consume_pending_attachment_by_path(self, file_path: str) -> None:
        await self.session.execute(
            update(PendingAttachment)
            .where(
                and_(
                    PendingAttachment.file_path == file_path,
                    PendingAttachment.consumed.is_(False),
                )
            )
            .values(consumed=True)
        )
        await self.session.flush()

    async def get_daily_summary(self, telegram_user_id: int, summary_date: date) -> Optional[str]:
        row = (
            (
                await self.session.execute(
                    select(DailySummary).where(
                        and_(
                            DailySummary.telegram_user_id == telegram_user_id,
                            DailySummary.summary_date == summary_date,
                        )
                    )
                )
            )
            .scalars()
            .first()
        )
        return row.text if row else None

    async def upsert_daily_summary(self, telegram_user_id: int, summary_date: date, text: str) -> None:
        row = (
            (
                await self.session.execute(
                    select(DailySummary).where(
                        and_(
                            DailySummary.telegram_user_id == telegram_user_id,
                            DailySummary.summary_date == summary_date,
                        )
                    )
                )
            )
            .scalars()
            .first()
        )
        if row:
            row.text = text
            row.updated_at = datetime.now(timezone.utc)
        else:
            self.session.add(
                DailySummary(
                    telegram_user_id=telegram_user_id,
                    summary_date=summary_date,
                    text=text,
                )
            )
        await self.session.flush()

    async def get_daily_summaries_in_range(
        self,
        telegram_user_id: int,
        start_date: date,
        end_date: date,
    ) -> list[str]:
        rows = (
            (
                await self.session.execute(
                    select(DailySummary)
                    .where(
                        and_(
                            DailySummary.telegram_user_id == telegram_user_id,
                            DailySummary.summary_date >= start_date,
                            DailySummary.summary_date <= end_date,
                        )
                    )
                    .order_by(DailySummary.summary_date.asc())
                )
            )
            .scalars()
            .all()
        )
        return [item.text for item in rows]

    async def get_major_memory(self, telegram_user_id: int) -> Optional[str]:
        row = await self.session.get(MajorMemory, telegram_user_id)
        return row.text if row else None

    async def set_major_memory(self, telegram_user_id: int, text: str) -> None:
        row = await self.session.get(MajorMemory, telegram_user_id)
        if row:
            row.text = text
            row.updated_at = datetime.now(timezone.utc)
        else:
            self.session.add(MajorMemory(telegram_user_id=telegram_user_id, text=text))
        await self.session.flush()

    async def list_messages_for_date(
        self,
        telegram_user_id: int,
        day: date,
        timezone_name: str = settings.timezone,
    ) -> list[Message]:
        start_local = datetime.combine(day, time.min, tzinfo=ZoneInfo(timezone_name))
        end_local = start_local + timedelta(days=1)
        start = start_local.astimezone(timezone.utc)
        end = end_local.astimezone(timezone.utc)
        rows = (
            (
                await self.session.execute(
                    select(Message)
                    .where(
                        and_(
                            Message.telegram_user_id == telegram_user_id,
                            Message.created_at >= start,
                            Message.created_at < end,
                            Message.role.in_(["user", "assistant"]),
                        )
                    )
                    .order_by(Message.id.asc())
                )
            )
            .scalars()
            .all()
        )
        return rows

    async def add_file_metadata(
        self,
        telegram_user_id: int,
        scope: str,
        file_path: str,
        kind: str,
        status: str = "active",
    ) -> None:
        self.session.add(
            FileMetadata(
                telegram_user_id=telegram_user_id,
                scope=scope,
                file_path=file_path,
                kind=kind,
                status=status,
            )
        )
        await self.session.flush()

    async def mark_file_deleted(self, telegram_user_id: int, file_path: str) -> None:
        row = (
            (
                await self.session.execute(
                    select(FileMetadata)
                    .where(
                        and_(
                            FileMetadata.telegram_user_id == telegram_user_id,
                            FileMetadata.file_path == file_path,
                            FileMetadata.status == "active",
                        )
                    )
                    .order_by(FileMetadata.id.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        if row:
            row.status = "deleted"
            row.updated_at = datetime.now(timezone.utc)
            await self.session.flush()

    async def recent_files(self, telegram_user_id: int, limit: int = 20) -> list[FileMetadata]:
        rows = (
            (
                await self.session.execute(
                    select(FileMetadata)
                    .where(FileMetadata.telegram_user_id == telegram_user_id)
                    .order_by(FileMetadata.id.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return rows

    async def create_reminder(
        self,
        creator_user_id: int,
        target_user_id: int,
        scheduled_for: datetime,
        text: str,
    ) -> Reminder:
        reminder = Reminder(
            creator_user_id=creator_user_id,
            target_user_id=target_user_id,
            scheduled_for=scheduled_for,
            text=text,
            status="pending",
        )
        self.session.add(reminder)
        await self.session.flush()
        return reminder

    async def get_reminder(self, reminder_id: int) -> Optional[Reminder]:
        return await self.session.get(Reminder, reminder_id)

    async def pending_reminders(self) -> list[Reminder]:
        rows = (
            (
                await self.session.execute(
                    select(Reminder)
                    .where(Reminder.status == "pending")
                    .order_by(Reminder.scheduled_for.asc())
                )
            )
            .scalars()
            .all()
        )
        return rows

    async def list_reminders_for_user(
        self,
        target_user_id: int,
        status: Optional[str] = "pending",
        limit: int = 20,
    ) -> list[Reminder]:
        stmt = select(Reminder).where(Reminder.target_user_id == target_user_id)
        if status:
            stmt = stmt.where(Reminder.status == status)
        rows = (
            (
                await self.session.execute(
                    stmt.order_by(Reminder.scheduled_for.asc()).limit(max(1, int(limit)))
                )
            )
            .scalars()
            .all()
        )
        return rows

    async def due_reminders(self, now_utc: datetime) -> list[Reminder]:
        rows = (
            (
                await self.session.execute(
                    select(Reminder)
                    .where(
                        and_(
                            Reminder.status == "pending",
                            Reminder.scheduled_for <= now_utc,
                        )
                    )
                    .order_by(Reminder.scheduled_for.asc())
                )
            )
            .scalars()
            .all()
        )
        return rows

    async def mark_reminder_sent(self, reminder_id: int) -> None:
        row = await self.session.get(Reminder, reminder_id)
        if row:
            row.status = "sent"
            row.sent_at = datetime.now(timezone.utc)
            await self.session.flush()

    async def list_known_users(self) -> list[int]:
        rows = (
            (
                await self.session.execute(
                    select(User.telegram_user_id).order_by(User.telegram_user_id.asc())
                )
            )
            .scalars()
            .all()
        )
        return list(rows)

    async def insert_job_log(
        self,
        job_name: str,
        status: str,
        details: str,
        telegram_user_id: Optional[int] = None,
    ) -> None:
        self.session.add(
            JobLog(
                job_name=job_name,
                status=status,
                details=details,
                telegram_user_id=telegram_user_id,
            )
        )
        await self.session.flush()

    async def message_count(self, telegram_user_id: int) -> int:
        count = (
            (
                await self.session.execute(
                    select(func.count(Message.id)).where(Message.telegram_user_id == telegram_user_id)
                )
            )
            .scalar_one()
        )
        return int(count)
