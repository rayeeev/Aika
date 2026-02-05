from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from src.llm.groq_client import GroqClient
from src.memory.store import MemoryStore


class MemorySummarizer:
    def __init__(self, store: MemoryStore, groq_client: GroqClient) -> None:
        self.store = store
        self.groq = groq_client

    async def compress_one_exchange_if_needed(self, telegram_user_id: int) -> Optional[str]:
        target = await self.store.oldest_uncompressed_exchange_outside_recent(telegram_user_id, keep_recent=3)
        if target is None:
            return None

        sentence = await self.groq.summarize_exchange(
            user_text=target.user_text,
            assistant_text=target.assistant_text,
        )
        await self.store.add_compressed_sentence(telegram_user_id, target.exchange_id, sentence)
        await self.store.mark_exchange_compressed(target.exchange_id)
        return sentence

    async def run_daily_summary(self, telegram_user_id: int, day: date) -> Optional[str]:
        messages = await self.store.list_messages_for_date(telegram_user_id, day)
        if not messages:
            return None

        transcript = [f"{m.role}: {m.content}" for m in messages]
        summary = await self.groq.summarize_daily(transcript)
        await self.store.upsert_daily_summary(telegram_user_id, day, summary)
        return summary

    async def run_weekly_major_memory(self, telegram_user_id: int, sunday_day: date) -> Optional[str]:
        start = sunday_day - timedelta(days=6)
        daily = await self.store.get_daily_summaries_in_range(telegram_user_id, start, sunday_day)
        if not daily:
            return None

        previous = await self.store.get_major_memory(telegram_user_id)
        major = await self.groq.summarize_major(daily_summaries=daily, previous_major_memory=previous)
        await self.store.set_major_memory(telegram_user_id, major)
        return major
