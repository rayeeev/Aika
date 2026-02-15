import aiosqlite
import asyncio
import json
import re
import time
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from zoneinfo import ZoneInfo
from groq import Groq

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("AIKA_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "aika.db"))
TIMEZONE = ZoneInfo("America/Los_Angeles")

# --- Configuration ---
CONVERSATION_GAP_SECONDS = 1800  # 30 minutes → new conversation
IMMEDIATE_BUFFER_INTERACTIONS = 10  # 10 interactions = 20 messages max
CC_SUMMARY_MAX_SENTENCES = 3
CONVERSATION_SUMMARY_MAX_SENTENCES = 4
DAY_SUMMARY_MAX_SENTENCES = 3
GLOBAL_SUMMARY_MAX_SENTENCES = 4


class Memory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.groq_client: Optional[Groq] = None
        self.extraction_model = "qwen/qwen3-32b"
        self.filter_model = "llama-3.1-8b-instant"
        self._lock = asyncio.Lock()

    def set_groq_client(self, client: Groq):
        self.groq_client = client

    # ── Schema ──────────────────────────────────────────────

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    summary TEXT,
                    is_closed INTEGER NOT NULL DEFAULT 0
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
                    summary_text TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
                    created_at REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS day_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    summary TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS global_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_conv_closed ON conversations(is_closed)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_episodic_created ON episodic_memories(created_at)")
            await db.commit()

    # ── Conversation Lifecycle ──────────────────────────────

    async def _get_active_conversation(self, db) -> Optional[int]:
        """Get the currently open conversation ID, or None."""
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE is_closed = 0 ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def _get_last_message_time(self, db, conv_id: int) -> Optional[float]:
        """Get the timestamp of the last message in a conversation."""
        cursor = await db.execute(
            "SELECT timestamp FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
            (conv_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def _create_conversation(self, db, now: float) -> int:
        """Create a new conversation and its empty CC row."""
        cursor = await db.execute(
            "INSERT INTO conversations (started_at, is_closed) VALUES (?, 0)",
            (now,)
        )
        conv_id = cursor.lastrowid
        await db.execute(
            "INSERT INTO conversation_context (conversation_id, summary_text, updated_at) VALUES (?, '', ?)",
            (conv_id, now)
        )
        await db.commit()
        logger.info(f"Created new conversation {conv_id}")
        return conv_id

    async def add_message(self, role: str, content: str):
        """Insert a message. Handles conversation boundaries and CC overflow."""
        now = time.time()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                conv_id = await self._get_active_conversation(db)

                if conv_id is not None:
                    last_ts = await self._get_last_message_time(db, conv_id)
                    if last_ts and (now - last_ts) > CONVERSATION_GAP_SECONDS:
                        # Gap detected → close old conversation, open new one
                        await self._close_conversation_internal(db, conv_id, now)
                        conv_id = await self._create_conversation(db, now)
                else:
                    conv_id = await self._create_conversation(db, now)

                # Insert the message
                await db.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (conv_id, role, content, now)
                )
                await db.commit()

                # Check if we need to update CC summary (buffer overflow)
                await self._maybe_update_cc(db, conv_id)

    async def _maybe_update_cc(self, db, conv_id: int):
        """If the conversation has more than IMMEDIATE_BUFFER_INTERACTIONS interactions,
        fold the oldest interaction into the CC summary."""
        # Count total messages in this conversation
        cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conv_id,)
        )
        total_messages = (await cursor.fetchone())[0]
        buffer_messages = IMMEDIATE_BUFFER_INTERACTIONS * 2  # 10 interactions = 20 messages

        if total_messages <= buffer_messages:
            return  # Buffer not overflowing

        # Get messages that are outside the buffer (oldest ones to fold in)
        overflow_count = total_messages - buffer_messages
        cursor = await db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
            (conv_id, overflow_count)
        )
        overflow_msgs = await cursor.fetchall()

        if not overflow_msgs:
            return

        # Get current CC summary
        cursor = await db.execute(
            "SELECT summary_text FROM conversation_context WHERE conversation_id = ?",
            (conv_id,)
        )
        row = await cursor.fetchone()
        current_summary = row[0] if row else ""

        # Build text of overflow messages
        overflow_text = "\n".join(f"[{m[0]}]: {m[1]}" for m in overflow_msgs)

        # Call Groq to fold into summary
        new_summary = await self._summarize_cc(current_summary, overflow_text)
        if new_summary:
            await db.execute(
                "UPDATE conversation_context SET summary_text = ?, updated_at = ? WHERE conversation_id = ?",
                (new_summary, time.time(), conv_id)
            )

        # Delete the overflowed messages from the buffer
        cursor = await db.execute(
            "SELECT id FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
            (conv_id, overflow_count)
        )
        ids_to_delete = [r[0] for r in await cursor.fetchall()]
        if ids_to_delete:
            placeholders = ",".join(["?"] * len(ids_to_delete))
            await db.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", ids_to_delete)

        await db.commit()

    async def _summarize_cc(self, current_summary: str, new_messages: str) -> Optional[str]:
        """Use Groq to fold new messages into the CC summary (max 3 sentences)."""
        if not self.groq_client:
            return None

        parts = []
        if current_summary:
            parts.append(f"Current conversation summary so far:\n{current_summary}")
        parts.append(f"New messages to incorporate:\n{new_messages}")

        prompt = (
            "\n\n".join(parts) + "\n\n"
            f"Summarize everything above into {CC_SUMMARY_MAX_SENTENCES} or fewer sentences. "
            "Capture the key topics, decisions, and context. Output ONLY the summary, nothing else."
        )

        return await self._call_groq(
            prompt,
            system="You summarize conversation context concisely. Output ONLY the summary text.",
            model=self.extraction_model,
            temperature=0.3,
        )

    async def _close_conversation_internal(self, db, conv_id: int, now: float):
        """Close a conversation: get full text, extract episodic memories, summarize."""
        # Fetch ALL messages (buffer + we need the full convo for extraction)
        # But we also need the CC summary for messages that were already folded
        cursor = await db.execute(
            "SELECT summary_text FROM conversation_context WHERE conversation_id = ?",
            (conv_id,)
        )
        cc_row = await cursor.fetchone()
        cc_summary = cc_row[0] if cc_row and cc_row[0] else ""

        cursor = await db.execute(
            "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,)
        )
        messages = await cursor.fetchall()

        if not messages:
            await db.execute(
                "UPDATE conversations SET is_closed = 1, ended_at = ? WHERE id = ?",
                (now, conv_id)
            )
            await db.commit()
            return

        # Build full conversation text
        full_text_parts = []
        if cc_summary:
            full_text_parts.append(f"[Earlier in conversation summary]: {cc_summary}")
        for role, content, ts in messages:
            full_text_parts.append(f"[{role}]: {content}")
        full_text = "\n".join(full_text_parts)

        # Get conversation time range
        cursor = await db.execute(
            "SELECT started_at FROM conversations WHERE id = ?",
            (conv_id,)
        )
        started_at = (await cursor.fetchone())[0]
        start_time = datetime.fromtimestamp(started_at, tz=TIMEZONE).strftime("%H:%M")
        end_time = datetime.fromtimestamp(now, tz=TIMEZONE).strftime("%H:%M")
        date_str = datetime.fromtimestamp(started_at, tz=TIMEZONE).strftime("%Y-%m-%d")
        time_interval = f"{date_str} {start_time}–{end_time}"

        # Call Groq for summary + episodic extraction (single call)
        summary, episodic_memories = await self._extract_conversation_close(full_text, time_interval)

        # Store summary
        await db.execute(
            "UPDATE conversations SET is_closed = 1, ended_at = ?, summary = ? WHERE id = ?",
            (now, summary, conv_id)
        )

        # Store episodic memories
        for mem in episodic_memories:
            await db.execute(
                "INSERT INTO episodic_memories (content, conversation_id, created_at) VALUES (?, ?, ?)",
                (mem, conv_id, now)
            )

        await db.commit()
        logger.info(f"Closed conversation {conv_id}: {len(episodic_memories)} episodic memories extracted")

    async def _extract_conversation_close(self, full_text: str, time_interval: str) -> tuple:
        """Extract conversation summary + episodic memories via Groq."""
        if not self.groq_client:
            return (f"[{time_interval}] (no summary available)", [])

        prompt = (
            f"Here is a full conversation:\n\n{full_text}\n\n"
            f"Time interval: {time_interval}\n\n"
            "Do two things:\n"
            f"1. Summarize this conversation in {CONVERSATION_SUMMARY_MAX_SENTENCES} or fewer sentences.\n"
            "2. Extract any episodic memories worth remembering (facts, situations, decisions, promises, preferences, notable events). "
            "It's OK to return no episodic memories if there's nothing worth storing.\n\n"
            "Return JSON (no markdown, no explanation):\n"
            '{"summary": "...", "episodic": ["memory1", "memory2", ...]}\n'
        )

        raw = await self._call_groq(
            prompt,
            system="You extract conversation summaries and episodic memories. Output ONLY valid JSON.",
            model=self.extraction_model,
            temperature=0.3,
        )

        if not raw:
            return (f"[{time_interval}] (extraction failed)", [])

        try:
            data = json.loads(raw)
            summary = f"[{time_interval}] {data.get('summary', '(no summary)')}"
            episodic = data.get("episodic", [])
            if not isinstance(episodic, list):
                episodic = []
            episodic = [str(e) for e in episodic if e]
            return (summary, episodic)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Conversation close extraction parse failed: {e}")
            return (f"[{time_interval}] (parse failed)", [])

    async def close_stale_conversations(self):
        """Called periodically (e.g. every 5 min) to close conversations
        where the last message was >30 min ago."""
        now = time.time()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                conv_id = await self._get_active_conversation(db)
                if conv_id is None:
                    return

                last_ts = await self._get_last_message_time(db, conv_id)
                if last_ts and (now - last_ts) > CONVERSATION_GAP_SECONDS:
                    logger.info(f"Closing stale conversation {conv_id} (idle for {int(now - last_ts)}s)")
                    await self._close_conversation_internal(db, conv_id, now)

    # ── Context Retrieval (for prompt assembly) ─────────────

    async def get_conversation_context(self) -> Dict[str, Any]:
        """Returns the CC for the active conversation:
        {"summary": str, "messages": [{"role": ..., "content": ..., "timestamp": ...}]}
        """
        async with aiosqlite.connect(self.db_path) as db:
            conv_id = await self._get_active_conversation(db)
            if conv_id is None:
                return {"summary": "", "messages": []}

            # Get CC summary
            cursor = await db.execute(
                "SELECT summary_text FROM conversation_context WHERE conversation_id = ?",
                (conv_id,)
            )
            row = await cursor.fetchone()
            summary = row[0] if row and row[0] else ""

            # Get buffered messages (the immediate buffer)
            cursor = await db.execute(
                "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conv_id,)
            )
            messages = [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in await cursor.fetchall()]

            return {"summary": summary, "messages": messages}

    async def get_today_conversation_summaries(self) -> str:
        """Get summaries of all closed conversations from today."""
        today_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
        # Get start of today in epoch
        today_start = datetime.strptime(today_str, "%Y-%m-%d").replace(tzinfo=TIMEZONE)
        today_epoch = today_start.timestamp()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT summary FROM conversations WHERE is_closed = 1 AND started_at >= ? AND summary IS NOT NULL ORDER BY started_at ASC",
                (today_epoch,)
            )
            rows = await cursor.fetchall()

        if not rows:
            return "(No earlier conversations today)"

        parts = []
        for i, (summary,) in enumerate(rows, 1):
            parts.append(f"{i}. {summary}")
        return "\n".join(parts)

    async def get_semantic_memories(self) -> str:
        """Get all semantic memories."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT content FROM semantic_memories ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()

        if not rows:
            return "(No semantic memories)"

        return "\n".join(f"• {r[0]}" for r in rows)

    async def get_appropriate_memories(self, user_text: str) -> str:
        """Use a fast Groq model to filter episodic memories for relevance."""
        if not self.groq_client:
            return "(No episodic memory filter available)"

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, content FROM episodic_memories ORDER BY created_at DESC LIMIT 100"
            )
            rows = await cursor.fetchall()

        if not rows:
            return "(No episodic memories)"

        memories_text = "\n".join(f"[{r[0]}] {r[1]}" for r in rows)

        prompt = (
            f"User's current message: {user_text}\n\n"
            f"Here are all stored episodic memories:\n{memories_text}\n\n"
            "Which of these memories are relevant to the user's current message? "
            "Return ONLY the IDs of relevant memories as a JSON array, like [1, 5, 12]. "
            "If none are relevant, return []. Output ONLY the JSON array."
        )

        raw = await self._call_groq(
            prompt,
            system="You filter memories for relevance. Output ONLY a JSON array of IDs.",
            model=self.filter_model,
            temperature=0.1,
        )

        if not raw:
            return "(Memory filter unavailable)"

        try:
            ids = json.loads(raw)
            if not isinstance(ids, list) or not ids:
                return "(No relevant episodic memories)"

            # Fetch the relevant ones
            relevant = [r[1] for r in rows if r[0] in ids]
            if not relevant:
                return "(No relevant episodic memories)"

            return "\n".join(f"• {m}" for m in relevant)
        except (json.JSONDecodeError, Exception):
            return "(Memory filter error)"

    # ── Scheduled Jobs ──────────────────────────────────────

    async def run_daily_summary(self):
        """4:00 AM — Summarize today's conversations into a day memory.
        Also analyze episodic memories to create semantic memories."""
        if not self.groq_client:
            return

        # Use yesterday's date (since it's 4 AM, we're summarizing the previous day)
        yesterday = datetime.now(TIMEZONE) - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")
        day_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        async with aiosqlite.connect(self.db_path) as db:
            # Get all conversation summaries from yesterday
            cursor = await db.execute(
                "SELECT summary FROM conversations WHERE is_closed = 1 AND started_at >= ? AND started_at < ? AND summary IS NOT NULL ORDER BY started_at ASC",
                (day_start.timestamp(), day_end.timestamp())
            )
            conv_summaries = [r[0] for r in await cursor.fetchall()]

            if not conv_summaries:
                logger.info(f"Daily summary: no conversations found for {date_str}")
                return

            # Summarize all conversations into day memory
            summaries_text = "\n".join(f"- {s}" for s in conv_summaries)
            prompt = (
                f"Here are all conversation summaries from {date_str}:\n\n{summaries_text}\n\n"
                f"Summarize the entire day into {DAY_SUMMARY_MAX_SENTENCES} or fewer sentences. "
                "Capture the most important events, topics, and outcomes. Output ONLY the summary."
            )
            day_summary = await self._call_groq(
                prompt,
                system="You create concise day summaries. Output ONLY the summary text.",
                model=self.extraction_model,
                temperature=0.3,
            )

            if day_summary:
                full_day_summary = f"[{date_str}] {day_summary}"
                await db.execute(
                    "INSERT OR REPLACE INTO day_memories (date, summary) VALUES (?, ?)",
                    (date_str, full_day_summary)
                )
                logger.info(f"Created day memory for {date_str}")

            # Analyze episodic memories to create semantic memories
            cursor = await db.execute(
                "SELECT id, content FROM episodic_memories ORDER BY created_at DESC LIMIT 50"
            )
            episodic_rows = await cursor.fetchall()

            if episodic_rows:
                episodic_text = "\n".join(f"[{r[0]}] {r[1]}" for r in episodic_rows)
                sem_prompt = (
                    f"Here are recent episodic memories:\n\n{episodic_text}\n\n"
                    "Analyze these episodic memories and extract any general semantic knowledge "
                    "(stable facts, user preferences, recurring patterns, biographical info). "
                    "Only create semantic memories for things that are clearly established patterns or facts, not one-off events. "
                    "It's okay to create none.\n\n"
                    "Return a JSON array of strings (each is one semantic memory). "
                    "Return [] if nothing qualifies. Output ONLY the JSON array."
                )
                sem_raw = await self._call_groq(
                    sem_prompt,
                    system="You analyze episodic memories to extract semantic knowledge. Output ONLY a JSON array.",
                    model=self.extraction_model,
                    temperature=0.3,
                )

                if sem_raw:
                    try:
                        sem_memories = json.loads(sem_raw)
                        if isinstance(sem_memories, list):
                            now = time.time()
                            for mem in sem_memories:
                                if isinstance(mem, str) and mem.strip():
                                    await db.execute(
                                        "INSERT INTO semantic_memories (content, created_at) VALUES (?, ?)",
                                        (mem.strip(), now)
                                    )
                            if sem_memories:
                                logger.info(f"Created {len(sem_memories)} semantic memories from episodic analysis")
                    except json.JSONDecodeError as e:
                        logger.error(f"Semantic memory extraction parse failed: {e}")

            await db.commit()

    async def run_global_update(self):
        """4:10 AM — Merge latest day memory into global memory."""
        if not self.groq_client:
            return

        yesterday = datetime.now(TIMEZONE) - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

        async with aiosqlite.connect(self.db_path) as db:
            # Get latest day memory
            cursor = await db.execute(
                "SELECT summary FROM day_memories WHERE date = ?",
                (date_str,)
            )
            day_row = await cursor.fetchone()
            if not day_row:
                logger.info("Global update: no day memory found for yesterday")
                return

            day_summary = day_row[0]

            # Get current global memory
            cursor = await db.execute(
                "SELECT summary FROM global_memory ORDER BY id DESC LIMIT 1"
            )
            global_row = await cursor.fetchone()
            current_global = global_row[0] if global_row else ""

            # Merge
            parts = []
            if current_global:
                parts.append(f"Current global memory:\n{current_global}")
            parts.append(f"Latest day memory:\n{day_summary}")

            prompt = (
                "\n\n".join(parts) + "\n\n"
                f"Merge the above into an updated global memory of {GLOBAL_SUMMARY_MAX_SENTENCES} or fewer sentences. "
                "This should be a high-level overview of the most important ongoing context about the user, "
                "their projects, preferences, and recent developments. Output ONLY the summary."
            )
            new_global = await self._call_groq(
                prompt,
                system="You maintain a concise global memory summary. Output ONLY the summary text.",
                model=self.extraction_model,
                temperature=0.3,
            )

            if new_global:
                now = time.time()
                if global_row:
                    await db.execute(
                        "UPDATE global_memory SET summary = ?, updated_at = ? WHERE id = (SELECT id FROM global_memory ORDER BY id DESC LIMIT 1)",
                        (new_global, now)
                    )
                else:
                    await db.execute(
                        "INSERT INTO global_memory (summary, updated_at) VALUES (?, ?)",
                        (new_global, now)
                    )
                await db.commit()
                logger.info("Updated global memory")

    async def run_weekly_cleanup(self):
        """4:20 AM Sunday — Clean up episodic and semantic memories."""
        if not self.groq_client:
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Get all episodic memories
            cursor = await db.execute(
                "SELECT id, content FROM episodic_memories ORDER BY created_at ASC"
            )
            episodic_rows = await cursor.fetchall()

            # Get all semantic memories
            cursor = await db.execute(
                "SELECT id, content FROM semantic_memories ORDER BY created_at ASC"
            )
            semantic_rows = await cursor.fetchall()

            if not episodic_rows and not semantic_rows:
                logger.info("Weekly cleanup: no memories to review")
                return

            parts = []
            if episodic_rows:
                episodic_text = "\n".join(f"[E{r[0]}] {r[1]}" for r in episodic_rows)
                parts.append(f"Episodic memories:\n{episodic_text}")
            if semantic_rows:
                semantic_text = "\n".join(f"[S{r[0]}] {r[1]}" for r in semantic_rows)
                parts.append(f"Semantic memories:\n{semantic_text}")

            prompt = (
                "\n\n".join(parts) + "\n\n"
                "Review all memories above. Identify:\n"
                "1. Duplicate or near-duplicate memories (keep the better one, delete the rest)\n"
                "2. Memories that are no longer relevant or have been superseded\n"
                "3. Episodic memories that can be merged\n\n"
                "Return JSON (no markdown):\n"
                '{"delete_ids": ["E1", "S3", ...], "reason": "short explanation"}\n'
                "If nothing needs cleanup, return: {\"delete_ids\": [], \"reason\": \"all clean\"}"
            )
            raw = await self._call_groq(
                prompt,
                system="You clean up AI memory by removing duplicates and stale entries. Output ONLY valid JSON.",
                model=self.extraction_model,
                temperature=0.2,
            )

            if not raw:
                return

            try:
                data = json.loads(raw)
                delete_ids = data.get("delete_ids", [])
                if not isinstance(delete_ids, list):
                    return

                for id_str in delete_ids:
                    if not isinstance(id_str, str):
                        continue
                    if id_str.startswith("E"):
                        mem_id = int(id_str[1:])
                        await db.execute("DELETE FROM episodic_memories WHERE id = ?", (mem_id,))
                    elif id_str.startswith("S"):
                        mem_id = int(id_str[1:])
                        await db.execute("DELETE FROM semantic_memories WHERE id = ?", (mem_id,))

                await db.commit()
                reason = data.get("reason", "")
                logger.info(f"Weekly cleanup: deleted {len(delete_ids)} memories. Reason: {reason}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Weekly cleanup parse failed: {e}")

    # ── Recall Memory (Tool) ───────────────────────────────

    async def recall_memory(self, query_type: str, date: str = "", time_range: str = "") -> str:
        """Retrieve stored memories for the user.
        query_type: 'conversation' | 'day' | 'global'
        date: YYYY-MM-DD (required for 'conversation' and 'day')
        time_range: HH:MM-HH:MM (required for 'conversation')
        """
        async with aiosqlite.connect(self.db_path) as db:
            if query_type == "global":
                cursor = await db.execute(
                    "SELECT summary FROM global_memory ORDER BY id DESC LIMIT 1"
                )
                row = await cursor.fetchone()
                return row[0] if row else "No global memory stored yet."

            elif query_type == "day":
                if not date:
                    return "Error: date is required for day query (format: YYYY-MM-DD)"
                cursor = await db.execute(
                    "SELECT summary FROM day_memories WHERE date = ?",
                    (date,)
                )
                row = await cursor.fetchone()
                if row:
                    return row[0]

                # Fallback: get conversation summaries for that day
                try:
                    day_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=TIMEZONE)
                    day_start = day_dt.timestamp()
                    day_end = (day_dt + timedelta(days=1)).timestamp()
                    cursor = await db.execute(
                        "SELECT summary FROM conversations WHERE is_closed = 1 AND started_at >= ? AND started_at < ? AND summary IS NOT NULL ORDER BY started_at ASC",
                        (day_start, day_end)
                    )
                    rows = await cursor.fetchall()
                    if rows:
                        return "No day summary yet, but here are the conversation summaries:\n" + "\n".join(f"- {r[0]}" for r in rows)
                except ValueError:
                    return f"Invalid date format: {date}. Use YYYY-MM-DD."
                return f"No memories found for {date}."

            elif query_type == "conversation":
                if not date or not time_range:
                    return "Error: both date and time_range required for conversation query"
                try:
                    # Parse time range
                    parts = time_range.split("-")
                    if len(parts) != 2:
                        return f"Invalid time_range format: {time_range}. Use HH:MM-HH:MM."
                    start_str, end_str = parts
                    day_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=TIMEZONE)
                    sh, sm = map(int, start_str.strip().split(":"))
                    eh, em = map(int, end_str.strip().split(":"))
                    range_start = day_dt.replace(hour=sh, minute=sm).timestamp()
                    range_end = day_dt.replace(hour=eh, minute=em).timestamp()

                    cursor = await db.execute(
                        "SELECT summary FROM conversations WHERE is_closed = 1 AND started_at >= ? AND started_at <= ? AND summary IS NOT NULL ORDER BY started_at ASC",
                        (range_start, range_end)
                    )
                    rows = await cursor.fetchall()
                    if rows:
                        return "\n\n".join(r[0] for r in rows)
                    return f"No conversations found in time range {time_range} on {date}."
                except (ValueError, IndexError) as e:
                    return f"Error parsing query: {e}"
            else:
                return f"Unknown query_type: {query_type}. Use 'global', 'day', or 'conversation'."

    # ── Groq Helper ─────────────────────────────────────────

    async def _call_groq(self, prompt: str, system: str, model: str, temperature: float = 0.3) -> Optional[str]:
        """Call Groq API in a thread executor. Returns cleaned text or None."""
        if not self.groq_client:
            return None

        def _sync_call():
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    model=model,
                    temperature=temperature,
                )
                raw = response.choices[0].message.content or ""
                raw = raw.strip()
                # Strip <think>...</think> blocks (qwen3 thinking)
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                # Strip markdown code fences
                if raw.startswith("```"):
                    raw = re.sub(r'^```(?:json)?\s*', '', raw)
                    raw = re.sub(r'\s*```$', '', raw)
                return raw if raw else None
            except Exception as e:
                logger.error(f"Groq call failed ({model}): {e}")
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_call)

    # ── Utilities ───────────────────────────────────────────

    async def clear_all(self):
        """Wipe everything — full memory reset."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM messages")
            await db.execute("DELETE FROM conversation_context")
            await db.execute("DELETE FROM conversations")
            await db.execute("DELETE FROM episodic_memories")
            await db.execute("DELETE FROM semantic_memories")
            await db.execute("DELETE FROM day_memories")
            await db.execute("DELETE FROM global_memory")
            await db.commit()