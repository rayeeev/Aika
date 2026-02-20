import aiosqlite
import asyncio
import json
import re
import time
import os
import threading
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
MAX_EPISODIC_TOKENS = 4000    # Token budget for all episodic memories combined
MAX_SEMANTIC_TOKENS = 1500    # Token budget for all semantic memories combined
PRECALL_RECENT_MESSAGES = 20
PRECALL_MAX_SEMANTIC_CANDIDATES = 24
PRECALL_MAX_EPISODIC_CANDIDATES = 36
PRECALL_SELECTION_LIMIT = 8
COMPACTION_KEEP_MIN = 15
COMPACTION_KEEP_MAX = 20
COMPACTION_KEEP_DEFAULT = 18


class Memory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.groq_client: Optional[Groq] = None
        self.extraction_model = "qwen/qwen3-32b"
        self.filter_model = "llama-3.1-8b-instant"
        self._lock = asyncio.Lock()
        self._compaction_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()  # For sync tool methods (Gemini AFC)

    def set_groq_client(self, client: Groq):
        self.groq_client = client

    # ── Schema ──────────────────────────────────────────────

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for concurrent read/write safety
            await db.execute("PRAGMA journal_mode=WAL")
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

    async def get_active_conversation_message_count(self) -> int:
        """Return message count for current active conversation."""
        async with aiosqlite.connect(self.db_path) as db:
            conv_id = await self._get_active_conversation(db)
            if conv_id is None:
                return 0

            cursor = await db.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conv_id,)
            )
            row = await cursor.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

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
        """Insert a message and handle conversation boundaries.
        CC compaction is intentionally moved off the hot path and runs in background."""
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

            # Only delete overflow messages if summarization succeeded
            cursor = await db.execute(
                "SELECT id FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
                (conv_id, overflow_count)
            )
            ids_to_delete = [r[0] for r in await cursor.fetchall()]
            if ids_to_delete:
                placeholders = ",".join(["?"] * len(ids_to_delete))
                await db.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", ids_to_delete)

            await db.commit()
        else:
            logger.warning(f"CC summarization failed for conversation {conv_id} — overflow messages preserved")

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

        await self._enforce_memory_limits(db)
        await db.commit()
        logger.info(f"Closed conversation {conv_id}: {len(episodic_memories)} episodic memories extracted")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return max(1, len(text) // 4)

    async def _enforce_memory_limits(self, db):
        """Consolidate memories via Groq if total tokens exceed budget."""
        await self._consolidate_table(db, "episodic_memories", MAX_EPISODIC_TOKENS, "episodic")
        await self._consolidate_table(db, "semantic_memories", MAX_SEMANTIC_TOKENS, "semantic")

    async def _consolidate_table(self, db, table: str, token_budget: int, label: str):
        """Check a memory table's token usage; if over budget, ask Groq to consolidate."""
        cursor = await db.execute(f"SELECT id, content FROM {table} ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        if not rows:
            return

        total_tokens = sum(self._estimate_tokens(r[1]) for r in rows)
        if total_tokens <= token_budget:
            return

        logger.info(f"{label} memories over token budget ({total_tokens}/{token_budget}). Consolidating...")

        if not self.groq_client:
            # No Groq → fall back to deleting oldest until under budget
            running = total_tokens
            for id_, content in reversed(rows):  # oldest last in DESC order, so reversed = oldest first
                if running <= token_budget:
                    break
                running -= self._estimate_tokens(content)
                await db.execute(f"DELETE FROM {table} WHERE id = ?", (id_,))
            logger.info(f"Pruned {label} memories to ~{running} tokens (no Groq, oldest-first fallback)")
            return

        # Build memories text for Groq
        memories_text = "\n".join(f"[{r[0]}] {r[1]}" for r in rows)
        prompt = (
            f"Here are all {label} memories (currently ~{total_tokens} tokens, budget is {token_budget} tokens):\n\n"
            f"{memories_text}\n\n"
            f"Consolidate these to fit within ~{token_budget} tokens. You can:\n"
            "1. Remove memories that are redundant, outdated, or low-value\n"
            "2. Merge similar memories into shorter combined ones\n"
            "3. Summarize verbose memories more concisely\n\n"
            "Return JSON (no markdown):\n"
            '{"keep": [id1, id2, ...], "delete": [id3, id4, ...], '
            '"update": {"id": new_content, ...}, '
            '"add": ["new merged memory 1", ...]}\n\n'
            "- 'keep': IDs to keep as-is\n"
            "- 'delete': IDs to remove\n"
            "- 'update': IDs whose content should be replaced with a shorter version\n"
            "- 'add': brand new memories created by merging deleted ones (optional)\n"
            "The resulting set must fit within the token budget."
        )

        data = await self._call_groq_json(
            prompt,
            system=f"You consolidate AI {label} memories to fit a token budget. Output ONLY valid JSON.",
            model=self.extraction_model,
            temperature=0.2,
        )

        if data is None:
            # Fallback: delete oldest
            running = total_tokens
            for id_, content in reversed(rows):
                if running <= token_budget:
                    break
                running -= self._estimate_tokens(content)
                await db.execute(f"DELETE FROM {table} WHERE id = ?", (id_,))
            logger.info(f"Pruned {label} memories to ~{running} tokens (Groq failed, oldest-first fallback)")
            return

        try:
            # Process deletes
            for mem_id in data.get("delete", []):
                await db.execute(f"DELETE FROM {table} WHERE id = ?", (int(mem_id),))

            # Process updates (rewrite shorter)
            for mem_id_str, new_content in data.get("update", {}).items():
                await db.execute(
                    f"UPDATE {table} SET content = ? WHERE id = ?",
                    (str(new_content), int(mem_id_str))
                )

            # Process adds (merged memories)
            import time as _time
            now = _time.time()
            for new_mem in data.get("add", []):
                if isinstance(new_mem, str) and new_mem.strip():
                    if table == "episodic_memories":
                        await db.execute(
                            "INSERT INTO episodic_memories (content, conversation_id, created_at) VALUES (?, NULL, ?)",
                            (new_mem.strip(), now)
                        )
                    else:
                        await db.execute(
                            "INSERT INTO semantic_memories (content, created_at) VALUES (?, ?)",
                            (new_mem.strip(), now)
                        )

            deleted = len(data.get('delete', []))
            updated = len(data.get('update', {}))
            added = len(data.get('add', []))
            logger.info(f"Consolidated {label} memories: {deleted} deleted, {updated} updated, {added} added")

        except (ValueError, KeyError) as e:
            logger.error(f"{label} memory consolidation processing failed: {e}")
            # Fallback: delete oldest
            running = total_tokens
            for id_, content in reversed(rows):
                if running <= token_budget:
                    break
                running -= self._estimate_tokens(content)
                await db.execute(f"DELETE FROM {table} WHERE id = ?", (id_,))
            logger.info(f"Pruned {label} memories to ~{running} tokens (processing fallback)")

    async def _extract_conversation_close(self, full_text: str, time_interval: str) -> tuple:
        """Extract conversation summary + episodic memories via Groq with retry."""
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

        system_msg = "You extract conversation summaries and episodic memories. Output ONLY valid JSON."

        data = await self._call_groq_json(
            prompt, system=system_msg, model=self.extraction_model, temperature=0.3
        )

        if data is None:
            logger.error(f"Conversation close extraction fully failed for {time_interval}")
            return (f"[{time_interval}] (extraction failed)", [])

        summary = f"[{time_interval}] {data.get('summary', '(no summary)')}"
        episodic = data.get("episodic", [])
        if not isinstance(episodic, list):
            episodic = []
        episodic = [str(e) for e in episodic if e]
        return (summary, episodic)

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

    async def get_conversation_context(self, limit_messages: Optional[int] = None) -> Dict[str, Any]:
        """Returns CC for the active conversation.
        If limit_messages is provided, returns only the latest N messages."""
        async with aiosqlite.connect(self.db_path) as db:
            conv_id = await self._get_active_conversation(db)
            if conv_id is None:
                return {"summary": "", "messages": []}

            cursor = await db.execute(
                "SELECT summary_text FROM conversation_context WHERE conversation_id = ?",
                (conv_id,)
            )
            row = await cursor.fetchone()
            summary = row[0] if row and row[0] else ""

            if limit_messages and limit_messages > 0:
                cursor = await db.execute(
                    "SELECT role, content, timestamp FROM messages "
                    "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                    (conv_id, limit_messages)
                )
                fetched = await cursor.fetchall()
                fetched.reverse()
                messages = [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in fetched]
            else:
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
                "SELECT id, content FROM episodic_memories ORDER BY created_at DESC"
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

        data = await self._call_groq_json(
            prompt,
            system="You filter memories for relevance. Output ONLY a JSON array of IDs.",
            model=self.filter_model,
            temperature=0.1,
        )

        if data is None:
            return "(Memory filter unavailable)"

        if not isinstance(data, list) or not data:
            return "(No relevant episodic memories)"

        # Fetch the relevant ones
        relevant = [r[1] for r in rows if r[0] in data]
        if not relevant:
            return "(No relevant episodic memories)"

        return "\n".join(f"• {m}" for m in relevant)

    @staticmethod
    def _coerce_id_list(values: Any) -> List[int]:
        """Coerce a mixed list of ids into unique integers preserving order."""
        if not isinstance(values, list):
            return []
        result: List[int] = []
        seen = set()
        for value in values:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed not in seen:
                seen.add(parsed)
                result.append(parsed)
        return result

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _finalize_inference_context(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove internal helper fields before returning to caller."""
        cleaned = dict(data)
        cleaned.pop("_semantic_candidates", None)
        cleaned.pop("_episodic_candidates", None)
        return cleaned

    async def get_lightweight_inference_context(
        self,
        user_text: str,
        event_type: str = "user",
        recent_limit: int = PRECALL_RECENT_MESSAGES,
    ) -> Dict[str, Any]:
        """Fast local-only context assembly fallback (no Groq call)."""
        today_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
        today_start = datetime.strptime(today_str, "%Y-%m-%d").replace(tzinfo=TIMEZONE)
        today_epoch = today_start.timestamp()

        async with aiosqlite.connect(self.db_path) as db:
            conv_id = await self._get_active_conversation(db)
            cc_summary = ""
            recent_messages: List[Dict[str, Any]] = []

            if conv_id is not None:
                cursor = await db.execute(
                    "SELECT summary_text FROM conversation_context WHERE conversation_id = ?",
                    (conv_id,)
                )
                row = await cursor.fetchone()
                cc_summary = row[0] if row and row[0] else ""

                cursor = await db.execute(
                    "SELECT role, content, timestamp FROM messages "
                    "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                    (conv_id, max(1, recent_limit))
                )
                fetched = await cursor.fetchall()
                fetched.reverse()
                recent_messages = [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in fetched]

            cursor = await db.execute(
                "SELECT summary FROM conversations "
                "WHERE is_closed = 1 AND started_at >= ? AND summary IS NOT NULL "
                "ORDER BY started_at DESC LIMIT 6",
                (today_epoch,)
            )
            today_rows = await cursor.fetchall()
            today_rows.reverse()
            today_summaries = "\n".join(f"- {r[0]}" for r in today_rows) if today_rows else "(No earlier conversations today)"

            cursor = await db.execute(
                "SELECT summary FROM global_memory ORDER BY id DESC LIMIT 1"
            )
            global_row = await cursor.fetchone()
            global_summary = global_row[0] if global_row else "(No global memory yet)"

            cursor = await db.execute(
                "SELECT id, content FROM semantic_memories ORDER BY created_at DESC LIMIT ?",
                (PRECALL_MAX_SEMANTIC_CANDIDATES,)
            )
            semantic_rows = await cursor.fetchall()

            cursor = await db.execute(
                "SELECT id, content FROM episodic_memories ORDER BY created_at DESC LIMIT ?",
                (PRECALL_MAX_EPISODIC_CANDIDATES,)
            )
            episodic_rows = await cursor.fetchall()

        semantic_candidates = [{"id": r[0], "content": str(r[1])} for r in semantic_rows]
        episodic_candidates = [{"id": r[0], "content": str(r[1])} for r in episodic_rows]

        route = {
            "requires_tools": False,
            "complexity": "fast",
            "provider_hint": "auto",
            "model_hint": "",
        }

        context_summary = cc_summary or "(No prior summary in this conversation)"
        return {
            "conversation_id": conv_id,
            "event_type": event_type,
            "user_text": user_text,
            "cc_summary": cc_summary,
            "context_summary": context_summary,
            "recent_messages": recent_messages,
            "today_summaries": today_summaries,
            "global_summary": global_summary,
            "selected_semantic": [c["content"] for c in semantic_candidates[:PRECALL_SELECTION_LIMIT]],
            "selected_episodic": [c["content"] for c in episodic_candidates[:PRECALL_SELECTION_LIMIT]],
            "route": route,
            "_semantic_candidates": semantic_candidates,
            "_episodic_candidates": episodic_candidates,
        }

    async def prepare_inference_context(
        self,
        user_text: str,
        event_type: str = "user",
        requested_provider: str = "",
        requested_model: str = "",
        allowed_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """One Groq pre-call for memory selection + routing hints before main inference."""
        base = await self.get_lightweight_inference_context(
            user_text=user_text,
            event_type=event_type,
            recent_limit=PRECALL_RECENT_MESSAGES,
        )

        if not self.groq_client:
            return self._finalize_inference_context(base)

        semantic_candidates = base.get("_semantic_candidates", [])
        episodic_candidates = base.get("_episodic_candidates", [])

        payload = {
            "event_type": event_type,
            "user_text": user_text,
            "requested_provider": requested_provider,
            "requested_model": requested_model,
            "allowed_models": allowed_models or [],
            "conversation_context_summary": self._truncate_text(base.get("cc_summary", ""), 2400),
            "recent_messages": [
                {
                    "role": m.get("role", ""),
                    "content": self._truncate_text(str(m.get("content", "")), 300),
                }
                for m in base.get("recent_messages", [])[-PRECALL_RECENT_MESSAGES:]
            ],
            "today_summaries": self._truncate_text(base.get("today_summaries", ""), 2400),
            "global_summary": self._truncate_text(base.get("global_summary", ""), 1200),
            "semantic_candidates": [
                {"id": c["id"], "content": self._truncate_text(c["content"], 220)}
                for c in semantic_candidates
            ],
            "episodic_candidates": [
                {"id": c["id"], "content": self._truncate_text(c["content"], 220)}
                for c in episodic_candidates
            ],
        }

        prompt = (
            "Analyze the payload and prepare compact memory context for the next assistant response.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{"
            "\"context_summary\":\"string <= 6 sentences\","
            "\"semantic_ids\":[int,... up to 8],"
            "\"episodic_ids\":[int,... up to 8],"
            "\"route\":{"
            "\"requires_tools\":true/false,"
            "\"complexity\":\"fast|deep\","
            "\"provider_hint\":\"auto|groq|gemini\","
            "\"model_hint\":\"optional model name\""
            "}"
            "}\n"
            "Selection rules:\n"
            "- Pick ONLY IDs present in candidates.\n"
            "- If user asks to run commands/read-write files/logs/schedule/manage memory, set requires_tools=true.\n"
            "- Prefer provider_hint=groq unless tools are clearly required.\n"
            "- context_summary must be concise and directly useful.\n\n"
            f"Payload JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        data = await self._call_groq_json(
            prompt,
            system="You prepare AI inference context. Output ONLY valid JSON.",
            model=self.filter_model,
            temperature=0.1,
            max_retries=1,
        )

        if not isinstance(data, dict):
            return self._finalize_inference_context(base)

        semantic_ids = self._coerce_id_list(data.get("semantic_ids", []))
        episodic_ids = self._coerce_id_list(data.get("episodic_ids", []))
        semantic_map = {c["id"]: c["content"] for c in semantic_candidates}
        episodic_map = {c["id"]: c["content"] for c in episodic_candidates}

        selected_semantic = [semantic_map[mid] for mid in semantic_ids if mid in semantic_map][:PRECALL_SELECTION_LIMIT]
        selected_episodic = [episodic_map[mid] for mid in episodic_ids if mid in episodic_map][:PRECALL_SELECTION_LIMIT]

        if not selected_semantic:
            selected_semantic = [c["content"] for c in semantic_candidates[:PRECALL_SELECTION_LIMIT]]
        if not selected_episodic:
            selected_episodic = [c["content"] for c in episodic_candidates[:PRECALL_SELECTION_LIMIT]]

        route_raw = data.get("route", {}) if isinstance(data.get("route"), dict) else {}
        complexity = str(route_raw.get("complexity", "fast")).strip().lower()
        if complexity not in {"fast", "deep"}:
            complexity = "fast"

        provider_hint = str(route_raw.get("provider_hint", "auto")).strip().lower()
        if provider_hint not in {"auto", "groq", "gemini"}:
            provider_hint = "auto"

        requires_tools_raw = route_raw.get("requires_tools", False)
        if isinstance(requires_tools_raw, bool):
            requires_tools = requires_tools_raw
        elif isinstance(requires_tools_raw, str):
            requires_tools = requires_tools_raw.strip().lower() in {"true", "1", "yes", "y"}
        else:
            requires_tools = bool(requires_tools_raw)

        model_hint = str(route_raw.get("model_hint", "")).strip()
        if allowed_models:
            allowed_lookup = {m.lower(): m for m in allowed_models}
            model_hint = allowed_lookup.get(model_hint.lower(), "")

        context_summary = str(data.get("context_summary", "")).strip() or base.get("context_summary", "")
        base["context_summary"] = context_summary
        base["selected_semantic"] = selected_semantic
        base["selected_episodic"] = selected_episodic
        base["route"] = {
            "requires_tools": requires_tools,
            "complexity": complexity,
            "provider_hint": provider_hint,
            "model_hint": model_hint,
        }

        return self._finalize_inference_context(base)

    async def compact_conversation_context_if_due(self, keep_last: int = COMPACTION_KEEP_DEFAULT) -> bool:
        """Fold older messages into CC summary, keeping the most recent 15-20 messages."""
        if not self.groq_client:
            return False

        keep_last = max(COMPACTION_KEEP_MIN, min(COMPACTION_KEEP_MAX, int(keep_last)))

        async with self._compaction_lock:
            # Snapshot what to fold without keeping the main write lock during Groq call.
            async with self._lock:
                async with aiosqlite.connect(self.db_path) as db:
                    conv_id = await self._get_active_conversation(db)
                    if conv_id is None:
                        return False

                    cursor = await db.execute(
                        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                        (conv_id,)
                    )
                    total_messages = (await cursor.fetchone())[0]
                    if total_messages <= keep_last:
                        return False

                    overflow_count = total_messages - keep_last
                    cursor = await db.execute(
                        "SELECT id, role, content FROM messages "
                        "WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
                        (conv_id, overflow_count)
                    )
                    overflow_rows = await cursor.fetchall()
                    if not overflow_rows:
                        return False

                    cursor = await db.execute(
                        "SELECT summary_text FROM conversation_context WHERE conversation_id = ?",
                        (conv_id,)
                    )
                    row = await cursor.fetchone()
                    current_summary = row[0] if row and row[0] else ""

            overflow_text = "\n".join(
                f"[{role}]: {content}" for _, role, content in overflow_rows
            )
            new_summary = await self._summarize_cc(current_summary, overflow_text)
            if not new_summary:
                logger.warning(f"Background CC compaction failed for conversation {conv_id}")
                return False

            ids_to_delete = [row[0] for row in overflow_rows]
            async with self._lock:
                async with aiosqlite.connect(self.db_path) as db:
                    active_conv = await self._get_active_conversation(db)
                    if active_conv != conv_id:
                        return False

                    await db.execute(
                        "UPDATE conversation_context SET summary_text = ?, updated_at = ? WHERE conversation_id = ?",
                        (new_summary, time.time(), conv_id)
                    )

                    placeholders = ",".join(["?"] * len(ids_to_delete))
                    cursor = await db.execute(
                        f"SELECT id FROM messages WHERE id IN ({placeholders})",
                        ids_to_delete
                    )
                    existing_ids = [r[0] for r in await cursor.fetchall()]
                    if existing_ids:
                        del_placeholders = ",".join(["?"] * len(existing_ids))
                        await db.execute(
                            f"DELETE FROM messages WHERE id IN ({del_placeholders})",
                            existing_ids
                        )

                    await db.commit()

            logger.info(
                f"Compacted conversation {conv_id}: folded {len(ids_to_delete)} messages, kept latest {keep_last}"
            )
            return True

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

            # Fetch existing semantic memories for dedup
            cursor = await db.execute(
                "SELECT content FROM semantic_memories ORDER BY created_at DESC"
            )
            existing_semantic = [r[0] for r in await cursor.fetchall()]

            if episodic_rows:
                episodic_text = "\n".join(f"[{r[0]}] {r[1]}" for r in episodic_rows)
                existing_text = "\n".join(f"• {s}" for s in existing_semantic) if existing_semantic else "(none)"
                sem_prompt = (
                    f"Here are recent episodic memories:\n\n{episodic_text}\n\n"
                    f"Here are EXISTING semantic memories (DO NOT duplicate these):\n{existing_text}\n\n"
                    "Analyze the episodic memories and extract any NEW general semantic knowledge "
                    "(stable facts, user preferences, recurring patterns, biographical info) "
                    "that is NOT already captured in existing semantic memories. "
                    "Only create semantic memories for things that are clearly established patterns or facts, not one-off events. "
                    "It's OK to return none if nothing new qualifies.\n\n"
                    "Return a JSON array of strings (each is one NEW semantic memory). "
                    "Return [] if nothing qualifies. Output ONLY the JSON array."
                )
                sem_data = await self._call_groq_json(
                    sem_prompt,
                    system="You analyze episodic memories to extract semantic knowledge. Output ONLY a JSON array.",
                    model=self.extraction_model,
                    temperature=0.3,
                )

                if sem_data is not None and isinstance(sem_data, list):
                    now = time.time()
                    count = 0
                    for mem in sem_data:
                        if isinstance(mem, str) and mem.strip():
                            await db.execute(
                                "INSERT INTO semantic_memories (content, created_at) VALUES (?, ?)",
                                (mem.strip(), now)
                            )
                            count += 1
                    if count:
                        logger.info(f"Created {count} semantic memories from episodic analysis")

            await self._enforce_memory_limits(db)
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
            data = await self._call_groq_json(
                prompt,
                system="You clean up AI memory by removing duplicates and stale entries. Output ONLY valid JSON.",
                model=self.extraction_model,
                temperature=0.2,
            )

            if data is None:
                return

            delete_ids = data.get("delete_ids", [])
            if not isinstance(delete_ids, list):
                return

            for id_str in delete_ids:
                if not isinstance(id_str, str):
                    continue
                if id_str.startswith("E"):
                    try:
                        mem_id = int(id_str[1:])
                        await db.execute("DELETE FROM episodic_memories WHERE id = ?", (mem_id,))
                    except ValueError:
                        continue
                elif id_str.startswith("S"):
                    try:
                        mem_id = int(id_str[1:])
                        await db.execute("DELETE FROM semantic_memories WHERE id = ?", (mem_id,))
                    except ValueError:
                        continue

            await db.commit()
            reason = data.get("reason", "")
            logger.info(f"Weekly cleanup: deleted {len(delete_ids)} memories. Reason: {reason}")

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

    # ── Groq Helpers ────────────────────────────────────────

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

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_call)

    async def _call_groq_json(self, prompt: str, system: str, model: str,
                              temperature: float = 0.3, max_retries: int = 2) -> Optional[Any]:
        """Call Groq and parse as JSON, with retry on parse failure.
        On failure, feeds the error back to Groq so it can correct its output.
        Returns parsed JSON (dict or list) or None after all retries exhausted."""
        if not self.groq_client:
            return None

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(1 + max_retries):
            def _sync_call(msgs=messages[:]):
                try:
                    response = self.groq_client.chat.completions.create(
                        messages=msgs,
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
                    logger.error(f"Groq JSON call failed ({model}, attempt {attempt + 1}): {e}")
                    return None

            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, _sync_call)

            if raw is None:
                if attempt < max_retries:
                    logger.warning(f"Groq returned empty response (attempt {attempt + 1}/{1 + max_retries}), retrying...")
                    # Add error feedback for retry (use placeholder, not empty string)
                    messages.append({"role": "assistant", "content": "(empty response)"})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was EMPTY. I need valid JSON output. "
                            "Please try again and return ONLY valid JSON, no markdown, no explanation."
                        )
                    })
                    continue
                logger.error(f"Groq returned empty response after {1 + max_retries} attempts")
                return None

            try:
                data = json.loads(raw)
                if attempt > 0:
                    logger.info(f"Groq JSON succeeded on retry attempt {attempt + 1}")
                return data
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Groq JSON parse failed (attempt {attempt + 1}/{1 + max_retries}): {e}. "
                        f"Raw response (first 200 chars): {raw[:200]}"
                    )
                    # Feed the error back to Groq so it can fix its output
                    messages.append({"role": "assistant", "content": raw[:500]})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your response is NOT valid JSON. Parse error: {e}\n"
                            "Please fix your output and return ONLY valid JSON. "
                            "No markdown code fences, no explanation, no text before or after the JSON."
                        )
                    })
                else:
                    logger.error(
                        f"Groq JSON parse failed after {1 + max_retries} attempts. "
                        f"Last error: {e}. Last raw (first 200 chars): {raw[:200]}"
                    )
                    return None

        return None

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

    # ── Sync Methods (for Gemini AFC tools) ────────────────

    def recall_memory_sync(self, query_type: str, date: str = "", time_range: str = "") -> str:
        """Sync version of recall_memory — uses plain sqlite3 so it's safe inside AFC."""
        import sqlite3
        with self._sync_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                if query_type == "global":
                    row = conn.execute(
                        "SELECT summary FROM global_memory ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    return row[0] if row else "No global memory stored yet."

                elif query_type == "day":
                    if not date:
                        return "Error: date is required for day query (format: YYYY-MM-DD)"
                    row = conn.execute(
                        "SELECT summary FROM day_memories WHERE date = ?", (date,)
                    ).fetchone()
                    if row:
                        return row[0]
                    try:
                        day_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=TIMEZONE)
                        day_start = day_dt.timestamp()
                        day_end = (day_dt + timedelta(days=1)).timestamp()
                        rows = conn.execute(
                            "SELECT summary FROM conversations WHERE is_closed = 1 AND started_at >= ? AND started_at < ? AND summary IS NOT NULL ORDER BY started_at ASC",
                            (day_start, day_end)
                        ).fetchall()
                        if rows:
                            return "No day summary yet, but here are the conversation summaries:\n" + "\n".join(f"- {r[0]}" for r in rows)
                    except ValueError:
                        return f"Invalid date format: {date}. Use YYYY-MM-DD."
                    return f"No memories found for {date}."

                elif query_type == "conversation":
                    if not date or not time_range:
                        return "Error: both date and time_range required for conversation query"
                    try:
                        parts = time_range.split("-")
                        if len(parts) != 2:
                            return f"Invalid time_range format: {time_range}. Use HH:MM-HH:MM."
                        start_str, end_str = parts
                        day_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=TIMEZONE)
                        sh, sm = map(int, start_str.strip().split(":"))
                        eh, em = map(int, end_str.strip().split(":"))
                        range_start = day_dt.replace(hour=sh, minute=sm).timestamp()
                        range_end = day_dt.replace(hour=eh, minute=em).timestamp()
                        rows = conn.execute(
                            "SELECT summary FROM conversations WHERE is_closed = 1 AND started_at >= ? AND started_at <= ? AND summary IS NOT NULL ORDER BY started_at ASC",
                            (range_start, range_end)
                        ).fetchall()
                        if rows:
                            return "\n\n".join(r[0] for r in rows)
                        return f"No conversations found in time range {time_range} on {date}."
                    except (ValueError, IndexError) as e:
                        return f"Error parsing query: {e}"
                else:
                    return f"Unknown query_type: {query_type}. Use 'global', 'day', or 'conversation'."
            finally:
                conn.close()

    def list_memories_sync(self, memory_type: str) -> str:
        """List episodic or semantic memories with their IDs."""
        import sqlite3
        with self._sync_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                if memory_type == "episodic":
                    rows = conn.execute(
                        "SELECT id, content, created_at FROM episodic_memories ORDER BY created_at DESC"
                    ).fetchall()
                    if not rows:
                        return "No episodic memories stored."
                    parts = []
                    for id_, content, created_at in rows:
                        dt = datetime.fromtimestamp(created_at, tz=TIMEZONE).strftime("%Y-%m-%d %H:%M")
                        parts.append(f"[E{id_}] ({dt}) {content}")
                    total_tokens = sum(Memory._estimate_tokens(c) for _, c, _ in rows)
                    return f"Episodic memories ({len(rows)} items, ~{total_tokens}/{MAX_EPISODIC_TOKENS} tokens):\n" + "\n".join(parts)

                elif memory_type == "semantic":
                    rows = conn.execute(
                        "SELECT id, content, created_at FROM semantic_memories ORDER BY created_at DESC"
                    ).fetchall()
                    if not rows:
                        return "No semantic memories stored."
                    parts = []
                    for id_, content, created_at in rows:
                        dt = datetime.fromtimestamp(created_at, tz=TIMEZONE).strftime("%Y-%m-%d %H:%M")
                        parts.append(f"[S{id_}] ({dt}) {content}")
                    total_tokens = sum(Memory._estimate_tokens(c) for _, c, _ in rows)
                    return f"Semantic memories ({len(rows)} items, ~{total_tokens}/{MAX_SEMANTIC_TOKENS} tokens):\n" + "\n".join(parts)
                else:
                    return f"Unknown memory_type: {memory_type}. Use 'episodic' or 'semantic'."
            finally:
                conn.close()

    def delete_memory_sync(self, memory_type: str, memory_id: int) -> str:
        """Delete a specific episodic or semantic memory by ID."""
        import sqlite3
        with self._sync_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                if memory_type == "episodic":
                    if not conn.execute("SELECT id FROM episodic_memories WHERE id = ?", (memory_id,)).fetchone():
                        return f"Episodic memory E{memory_id} not found."
                    conn.execute("DELETE FROM episodic_memories WHERE id = ?", (memory_id,))
                    conn.commit()
                    return f"Deleted episodic memory E{memory_id}."
                elif memory_type == "semantic":
                    if not conn.execute("SELECT id FROM semantic_memories WHERE id = ?", (memory_id,)).fetchone():
                        return f"Semantic memory S{memory_id} not found."
                    conn.execute("DELETE FROM semantic_memories WHERE id = ?", (memory_id,))
                    conn.commit()
                    return f"Deleted semantic memory S{memory_id}."
                else:
                    return f"Unknown memory_type: {memory_type}. Use 'episodic' or 'semantic'."
            finally:
                conn.close()

    def edit_memory_sync(self, memory_type: str, memory_id: int, new_content: str) -> str:
        """Edit the content of a specific episodic or semantic memory."""
        import sqlite3
        with self._sync_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                if memory_type == "episodic":
                    if not conn.execute("SELECT id FROM episodic_memories WHERE id = ?", (memory_id,)).fetchone():
                        return f"Episodic memory E{memory_id} not found."
                    conn.execute("UPDATE episodic_memories SET content = ? WHERE id = ?", (new_content, memory_id))
                    conn.commit()
                    return f"Updated episodic memory E{memory_id}."
                elif memory_type == "semantic":
                    if not conn.execute("SELECT id FROM semantic_memories WHERE id = ?", (memory_id,)).fetchone():
                        return f"Semantic memory S{memory_id} not found."
                    conn.execute("UPDATE semantic_memories SET content = ? WHERE id = ?", (new_content, memory_id))
                    conn.commit()
                    return f"Updated semantic memory S{memory_id}."
                else:
                    return f"Unknown memory_type: {memory_type}. Use 'episodic' or 'semantic'."
            finally:
                conn.close()
