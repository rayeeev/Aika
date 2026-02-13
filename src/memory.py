import aiosqlite
import asyncio
import time
import os
import logging
from typing import List, Dict, Optional
from groq import Groq

logger = logging.getLogger(__name__)

# Use environment variable or default to a path relative to the project root
DB_PATH = os.getenv("AIKA_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "aika.db"))

# Maximum summary lengths to prevent memory leaks
MAX_WEEKLY_SUMMARY_LENGTH = 1000
MAX_GLOBAL_SUMMARY_LENGTH = 2000

# Messages older than this (in seconds) are moved to weekly summary
BUFFER_EXPIRY_SECONDS = 3600  # 1 hour


class Memory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.groq_client = None  # Will be set by main.py
        self.model_name = "qwen/qwen3-32b"
        self._lock = asyncio.Lock()  # Prevent race conditions

    def set_groq_client(self, client: Groq):
        self.groq_client = client

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            # Key-value store for summaries
            await db.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            await db.commit()

    async def add_message(self, role: str, content: str):
        """Adds a message and triggers rolling window + time-based expiry."""
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Step 1: Expire old messages (>1 hour) before inserting
                await self._expire_old_messages(db)

                # Step 2: Insert the new message
                await db.execute(
                    "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
                    (role, content, time.time())
                )
                await db.commit()

                # Step 3: Check buffer size — pop oldest pair if >10 messages
                count_cursor = await db.execute("SELECT COUNT(*) FROM messages")
                count_row = await count_cursor.fetchone()
                count = count_row[0]

                if count > 10:
                    oldest_cursor = await db.execute(
                        "SELECT id, role, content FROM messages ORDER BY id ASC LIMIT 2"
                    )
                    oldest_rows = await oldest_cursor.fetchall()

                    if len(oldest_rows) == 2:
                        oldest_interaction_text = (
                            f"[{oldest_rows[0][1]}]: {oldest_rows[0][2]}\n"
                            f"[{oldest_rows[1][1]}]: {oldest_rows[1][2]}"
                        )

                        await self._update_weekly_summary(oldest_interaction_text)

                        ids_to_delete = [row[0] for row in oldest_rows]
                        placeholders = ','.join(['?' for _ in ids_to_delete])
                        await db.execute(
                            f"DELETE FROM messages WHERE id IN ({placeholders})",
                            ids_to_delete
                        )
                        await db.commit()

    async def _expire_old_messages(self, db):
        """Move messages older than BUFFER_EXPIRY_SECONDS into weekly summary.
        
        If Groq summarization fails, the messages are kept in the buffer
        to prevent data loss.
        """
        cutoff = time.time() - BUFFER_EXPIRY_SECONDS
        cursor = await db.execute(
            "SELECT id, role, content FROM messages WHERE timestamp < ? ORDER BY id ASC",
            (cutoff,)
        )
        old_rows = await cursor.fetchall()
        if not old_rows:
            return

        logger.info(f"Expiring {len(old_rows)} old message(s) from buffer")

        # Summarize in pairs
        summarized_ids = []
        for i in range(0, len(old_rows) - 1, 2):
            interaction_text = (
                f"[{old_rows[i][1]}]: {old_rows[i][2]}\n"
                f"[{old_rows[i + 1][1]}]: {old_rows[i + 1][2]}"
            )
            success = await self._update_weekly_summary(interaction_text)
            if success:
                summarized_ids.extend([old_rows[i][0], old_rows[i + 1][0]])
            else:
                # Groq failed — stop expiring to preserve data
                logger.warning("Groq summarization failed, keeping remaining messages in buffer")
                break

        # Handle odd leftover message (only if all pairs succeeded)
        if len(old_rows) % 2 == 1 and len(summarized_ids) == len(old_rows) - 1:
            last = old_rows[-1]
            success = await self._update_weekly_summary(f"[{last[1]}]: {last[2]}")
            if success:
                summarized_ids.append(last[0])

        # Delete only successfully summarized messages
        if summarized_ids:
            placeholders = ','.join(['?' for _ in summarized_ids])
            await db.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                summarized_ids
            )
            await db.commit()

    async def _update_weekly_summary(self, new_interaction_text: str) -> bool:
        """Update weekly summary with new interaction. Returns True on success."""
        if not self.groq_client:
            logger.warning("Groq client not set, skipping summary update")
            return False

        current_weekly = await self.get_summary("weekly_summary") or ""

        prompt = (
            f"Update the following weekly summary with the new interaction info.\n"
            f"Constraint: Keep it exactly within 3 sentences. Max {MAX_WEEKLY_SUMMARY_LENGTH} characters.\n\n"
            f"Current Weekly Summary:\n{current_weekly}\n\n"
            f"New Interaction to Merge:\n{new_interaction_text}\n\n"
            f"New Weekly Summary:"
        )

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise summarizer."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
            )
            new_summary = chat_completion.choices[0].message.content.strip()
            if len(new_summary) > MAX_WEEKLY_SUMMARY_LENGTH:
                new_summary = new_summary[:MAX_WEEKLY_SUMMARY_LENGTH]
            await self.save_summary("weekly_summary", new_summary)
            return True
        except Exception as e:
            logger.error(f"Error updating weekly summary: {e}")
            return False

    async def execute_weekly_reset(self):
        """Trigger B: End-of-Week Reset."""
        if not self.groq_client:
            logger.warning("Groq client not set, skipping weekly reset")
            return

        current_weekly = await self.get_summary("weekly_summary") or ""
        current_global = await self.get_summary("global_summary") or ""

        if not current_weekly:
            return  # Nothing to merge

        prompt = (
            f"Merge the weekly summary into the global summary.\n"
            f"Constraint: Keep it exactly within 4 sentences. Max {MAX_GLOBAL_SUMMARY_LENGTH} characters. "
            f"Retain critical core memories.\n\n"
            f"Current Global Summary:\n{current_global}\n\n"
            f"Weekly Summary to Merge:\n{current_weekly}\n\n"
            f"New Global Summary:"
        )

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise summarizer."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
            )
            new_global = chat_completion.choices[0].message.content.strip()
            if len(new_global) > MAX_GLOBAL_SUMMARY_LENGTH:
                new_global = new_global[:MAX_GLOBAL_SUMMARY_LENGTH]

            await self.save_summary("global_summary", new_global)
            await self.save_summary("weekly_summary", "")  # Wipe clean

        except Exception as e:
            logger.error(f"Error executing weekly reset: {e}")

    async def get_recent_messages(self, limit: int = 20) -> List[Dict]:
        """Get recent messages from the buffer with timestamps."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT role, content, timestamp FROM messages ORDER BY id ASC LIMIT ?",
                (limit,)
            )
            rows = await cursor.fetchall()
            return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in rows]

    async def get_summary(self, key: str) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT value FROM summaries WHERE key = ?", (key,))
            row = await cursor.fetchone()
            return row[0] if row else None

    async def save_summary(self, key: str, value: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO summaries (key, value) VALUES (?, ?)",
                (key, value)
            )
            await db.commit()

    async def get_full_context(self) -> str:
        global_sum = await self.get_summary("global_summary") or "(Empty)"
        weekly_sum = await self.get_summary("weekly_summary") or "(Empty)"
        messages = await self.get_recent_messages()

        buffer_text = ""
        for msg in messages:
            role_label = "User" if msg["role"] == "user" else "Aika"
            buffer_text += f"[{role_label}]: {msg['content']}\n"

        return (
            f"Global Summary (Long-term):\n{global_sum}\n\n"
            f"Weekly Summary (Short-term):\n{weekly_sum}\n\n"
            f"Immediate Context (Buffer):\n{buffer_text}"
        )

    async def clear_history(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM messages")
            await db.execute("DELETE FROM summaries")
            await db.commit()
