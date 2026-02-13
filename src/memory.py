import aiosqlite
import asyncio
import json
import math
import re
import time
import os
import logging
from typing import List, Dict, Optional, Any
from groq import Groq

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("AIKA_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "aika.db"))

# --- Configuration ---
BUFFER_MAX_MESSAGES = 20
BUFFER_EXPIRY_SECONDS = 3600  # 1 hour
MAX_RETRIEVED_MEMORIES = 10
MAX_MEMORY_CONTEXT_CHARS = 4000  # Character budget for recalled memory cards
DECAY_NODE_HALFLIFE = 0.995   # per hour
DECAY_EDGE_HALFLIFE = 0.99    # per hour (edges decay faster)
DEAD_EDGE_THRESHOLD = 0.01
ARCHIVE_STRENGTH_THRESHOLD = 0.05
ARCHIVE_DAYS = 30

# Stopwords for keyword extraction (kept minimal)
STOPWORDS = frozenset({
    "i", "me", "my", "you", "your", "we", "us", "our", "it", "its",
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "shall", "must",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "up",
    "about", "into", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then",
    "that", "this", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why", "all", "each", "every", "both",
    "just", "also", "very", "too", "only", "really", "much", "more",
    "some", "any", "than", "like", "get", "got", "go", "going",
    "thing", "things", "something", "anything", "ok", "okay", "yeah",
    "yes", "no", "hey", "hi", "hello", "please", "thanks", "thank",
    "sure", "right", "well", "now", "here", "there",
})


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text using simple tokenization."""
    tokens = re.findall(r'[a-zA-Z0-9_/.-]+', text.lower())
    keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return list(dict.fromkeys(keywords))[:20]  # Deduplicate, cap at 20


class Memory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.groq_client: Optional[Groq] = None
        self.model_name = "qwen/qwen3-32b"
        self._lock = asyncio.Lock()

    def set_groq_client(self, client: Groq):
        self.groq_client = client

    # ── Schema ──────────────────────────────────────────────

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
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    strength REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
                    target_id INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
                    weight REAL NOT NULL DEFAULT 1.0,
                    relation TEXT,
                    created_at REAL NOT NULL,
                    last_used REAL NOT NULL,
                    UNIQUE(source_id, target_id)
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_cues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
                    cue_type TEXT NOT NULL,
                    cue_value TEXT NOT NULL
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_cues_value ON memory_cues(cue_value)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_cues_node ON memory_cues(node_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_nodes_strength ON memory_nodes(strength)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id)")
            await db.commit()

    # ── Working Set (Buffer) ────────────────────────────────

    async def add_message(self, role: str, content: str):
        """Insert a message into the buffer and enforce size cap."""
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
                    (role, content, time.time())
                )
                # Delete messages older than 1 hour
                cutoff = time.time() - BUFFER_EXPIRY_SECONDS
                await db.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff,))
                # Enforce cap: keep newest BUFFER_MAX_MESSAGES
                await db.execute("""
                    DELETE FROM messages WHERE id NOT IN (
                        SELECT id FROM messages ORDER BY id DESC LIMIT ?
                    )
                """, (BUFFER_MAX_MESSAGES,))
                await db.commit()

    async def get_recent_messages(self, limit: int = 20) -> List[Dict]:
        """Get recent messages from the buffer with timestamps."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT role, content, timestamp FROM messages ORDER BY id ASC LIMIT ?",
                (limit,)
            )
            rows = await cursor.fetchall()
            return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

    # ── Ingest Pipeline ─────────────────────────────────────

    async def extract_and_store_memories(self, user_text: str, model_text: str):
        """Extract structured memories from a conversation turn via Groq, then store."""
        if not self.groq_client:
            return

        prompt = (
            "Given this conversation turn, extract any memories WORTH storing (it's okay to store nothing if there isn't anything worth storing).\n"
            "Return a JSON array (no markdown, no explanation). Each item:\n"
            '{"type":"semantic|episodic|procedural","title":"1-line","content":"2-4 sentences",'
            '"importance":1-5,"keywords":["word1","word2"],"entities":["name1"]}\n\n'
            "Rules:\n"
            "- semantic: stable facts, preferences, commitments, biographical info\n"
            "- episodic: notable events, decisions, emotional moments worth remembering\n"
            "- procedural: learned patterns about how the user wants things done\n"
            "- Most casual turns produce 0 memories. Return [] if nothing worth storing.\n"
            "- importance: 5=critical, 1=trivial. Only extract importance >= 2.\n\n"
            f"[user]: {user_text}\n[model]: {model_text}\n\n"
            "JSON array:"
        )

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You extract structured memories from conversations. Output ONLY valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.3,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()

            # Strip <think>...</think> blocks (qwen3 thinking mode)
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)

            if not raw:
                return

            memories = json.loads(raw)
            if not isinstance(memories, list):
                return

            for mem in memories:
                if not isinstance(mem, dict):
                    continue
                await self._store_extracted_memory(mem)

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Memory extraction failed: {e}")

    async def _store_extracted_memory(self, mem: Dict[str, Any]):
        """Store a single extracted memory, deduplicating by cue overlap."""
        mem_type = mem.get("type", "semantic")
        title = mem.get("title", "")[:200]
        content = mem.get("content", "")[:500]
        importance = min(max(mem.get("importance", 1), 1), 5)
        keywords = mem.get("keywords", [])[:10]
        entities = mem.get("entities", [])[:5]

        if not title or not content:
            return

        all_cues = [k.lower() for k in keywords] + [e.lower() for e in entities]
        if not all_cues:
            all_cues = extract_keywords(title + " " + content)[:8]

        now = time.time()

        async with aiosqlite.connect(self.db_path) as db:
            # Check for existing node with ≥2 shared cues
            if all_cues:
                placeholders = ','.join(['?' for _ in all_cues])
                cursor = await db.execute(f"""
                    SELECT node_id, COUNT(*) as overlap
                    FROM memory_cues
                    WHERE cue_value IN ({placeholders})
                    GROUP BY node_id
                    HAVING overlap >= 2
                    ORDER BY overlap DESC
                    LIMIT 1
                """, all_cues)
                existing = await cursor.fetchone()

                if existing:
                    # Reinforce existing node
                    node_id = existing[0]
                    await db.execute("""
                        UPDATE memory_nodes
                        SET strength = MIN(strength + ? * 0.2, 5.0),
                            last_accessed = ?,
                            access_count = access_count + 1,
                            content = ?
                        WHERE id = ?
                    """, (importance, now, content, node_id))
                    await db.commit()
                    logger.info(f"Reinforced existing memory node {node_id}: {title}")
                    return

            # Insert new node
            cursor = await db.execute("""
                INSERT INTO memory_nodes (type, title, content, strength, created_at, last_accessed, access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """, (mem_type, title, content, importance * 0.3, now, now,
                  json.dumps({"keywords": keywords, "entities": entities})))
            node_id = cursor.lastrowid

            # Insert cues
            for kw in keywords:
                await db.execute(
                    "INSERT INTO memory_cues (node_id, cue_type, cue_value) VALUES (?, 'keyword', ?)",
                    (node_id, kw.lower())
                )
            for ent in entities:
                await db.execute(
                    "INSERT INTO memory_cues (node_id, cue_type, cue_value) VALUES (?, 'entity', ?)",
                    (node_id, ent.lower())
                )

            # Link to existing nodes that share cues (association edges)
            if all_cues:
                placeholders = ','.join(['?' for _ in all_cues])
                cursor = await db.execute(f"""
                    SELECT DISTINCT node_id FROM memory_cues
                    WHERE cue_value IN ({placeholders}) AND node_id != ?
                """, all_cues + [node_id])
                related_ids = [r[0] for r in await cursor.fetchall()]

                for related_id in related_ids[:5]:  # Max 5 edges per new node
                    await db.execute("""
                        INSERT OR IGNORE INTO memory_edges (source_id, target_id, weight, relation, created_at, last_used)
                        VALUES (?, ?, 0.5, 'related_to', ?, ?)
                    """, (node_id, related_id, now, now))

            await db.commit()
            logger.info(f"Stored new {mem_type} memory: {title}")

    # ── Retrieval Pipeline ──────────────────────────────────

    async def retrieve_relevant_memories(self, user_text: str) -> str:
        """Retrieve relevant memory cards for the current turn. Returns formatted string."""
        keywords = extract_keywords(user_text)
        if not keywords:
            return "(No memories recalled)"

        now = time.time()

        async with aiosqlite.connect(self.db_path) as db:
            # Step 1: Find candidate nodes via cue matching
            placeholders = ','.join(['?' for _ in keywords])
            cursor = await db.execute(f"""
                SELECT mn.id, mn.type, mn.title, mn.content, mn.strength,
                       mn.last_accessed, mn.access_count, COUNT(mc.id) as cue_hits
                FROM memory_cues mc
                JOIN memory_nodes mn ON mc.node_id = mn.id
                WHERE mc.cue_value IN ({placeholders})
                  AND mn.type NOT LIKE 'archived_%'
                  AND mn.strength > {DEAD_EDGE_THRESHOLD}
                GROUP BY mn.id
                ORDER BY cue_hits DESC, mn.strength DESC
                LIMIT 30
            """, keywords)
            candidates = await cursor.fetchall()

            if not candidates:
                return "(No memories recalled)"

            # Step 2: Spreading activation — 1 hop from top candidates
            top_ids = [c[0] for c in candidates[:5]]
            if top_ids:
                id_placeholders = ','.join(['?' for _ in top_ids])
                cursor = await db.execute(f"""
                    SELECT mn.id, mn.type, mn.title, mn.content, mn.strength,
                           mn.last_accessed, mn.access_count, me.weight as edge_weight
                    FROM memory_edges me
                    JOIN memory_nodes mn ON (
                        (me.target_id = mn.id AND me.source_id IN ({id_placeholders}))
                        OR (me.source_id = mn.id AND me.target_id IN ({id_placeholders}))
                    )
                    WHERE mn.type NOT LIKE 'archived_%'
                      AND mn.strength > {DEAD_EDGE_THRESHOLD}
                      AND me.weight > 0.1
                    LIMIT 10
                """, top_ids + top_ids)
                hop_results = await cursor.fetchall()

                # Merge hop results into candidates (avoid duplicates)
                existing_ids = {c[0] for c in candidates}
                for hop in hop_results:
                    if hop[0] not in existing_ids:
                        candidates.append(hop[:7] + (1,))  # cue_hits=1 for hop results
                        existing_ids.add(hop[0])

            # Step 3: Score and rank
            scored = []
            for c in candidates:
                node_id, mem_type, title, content, strength, last_accessed, access_count, cue_hits = c
                hours_since = max((now - last_accessed) / 3600, 0.01)
                recency_boost = 1.0 / (1.0 + math.log(1 + hours_since))
                score = cue_hits * 1.0 + 0.6 * strength + 0.3 * recency_boost
                scored.append((score, node_id, mem_type, title, content, strength, last_accessed))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:MAX_RETRIEVED_MEMORIES]

            # Step 4: Reinforce retrieved nodes
            for item in top:
                node_id = item[1]
                await db.execute("""
                    UPDATE memory_nodes
                    SET strength = MIN(strength + 0.1, 5.0),
                        last_accessed = ?,
                        access_count = access_count + 1
                    WHERE id = ?
                """, (now, node_id))

            # Reinforce edges between co-retrieved nodes
            retrieved_ids = [item[1] for item in top]
            for i, id_a in enumerate(retrieved_ids):
                for id_b in retrieved_ids[i+1:]:
                    await db.execute("""
                        UPDATE memory_edges
                        SET weight = MIN(weight + 0.05, 5.0), last_used = ?
                        WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)
                    """, (now, id_a, id_b, id_b, id_a))

            await db.commit()

        # Step 5: Format memory cards
        if not top:
            return "(No memories recalled)"

        cards = []
        total_chars = 0
        for score, node_id, mem_type, title, content, strength, last_accessed in top:
            hours_ago = (now - last_accessed) / 3600
            if hours_ago < 1:
                time_str = f"{int(hours_ago * 60)}m ago"
            elif hours_ago < 24:
                time_str = f"{int(hours_ago)}h ago"
            else:
                time_str = f"{int(hours_ago / 24)}d ago"

            emoji = {"semantic": "💡", "episodic": "📅", "procedural": "⚙️"}.get(mem_type, "📌")
            card = (
                f"{emoji} [{mem_type.upper()}] {title}\n"
                f"   {content}\n"
                f"   strength: {strength:.1f} | last: {time_str}"
            )
            # Enforce character budget — stop adding cards if over limit
            if total_chars + len(card) > MAX_MEMORY_CONTEXT_CHARS and cards:
                logger.info(f"Memory context budget reached ({total_chars} chars, {len(cards)} cards). Skipping remaining.")
                break
            cards.append(card)
            total_chars += len(card) + 2  # +2 for "\n\n" join separator

        return "\n\n".join(cards)

    # ── Decay Pipeline ──────────────────────────────────────

    async def run_decay(self):
        """Apply time-based decay to node strengths and edge weights."""
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            # Fetch and update nodes
            cursor = await db.execute(
                "SELECT id, strength, last_accessed, access_count FROM memory_nodes WHERE strength > ?",
                (DEAD_EDGE_THRESHOLD,)
            )
            nodes = await cursor.fetchall()
            for node_id, strength, last_accessed, access_count in nodes:
                hours = max((now - last_accessed) / 3600, 0)
                # Nodes with high access_count decay slower
                effective_hours = hours / (1 + math.log(1 + access_count))
                new_strength = strength * (DECAY_NODE_HALFLIFE ** effective_hours)
                if abs(new_strength - strength) > 0.001:
                    await db.execute(
                        "UPDATE memory_nodes SET strength = ? WHERE id = ?",
                        (new_strength, node_id)
                    )

            # Decay edges
            await db.execute(f"""
                UPDATE memory_edges
                SET weight = weight * POWER({DECAY_EDGE_HALFLIFE}, (? - last_used) / 3600.0)
                WHERE weight > {DEAD_EDGE_THRESHOLD}
            """, (now,))

            # Prune dead edges
            await db.execute(f"DELETE FROM memory_edges WHERE weight < {DEAD_EDGE_THRESHOLD}")

            # Archive dead nodes
            archive_cutoff = now - (ARCHIVE_DAYS * 86400)
            await db.execute("""
                UPDATE memory_nodes
                SET type = 'archived_' || type
                WHERE strength < ?
                  AND last_accessed < ?
                  AND type NOT LIKE 'archived_%'
            """, (ARCHIVE_STRENGTH_THRESHOLD, archive_cutoff))

            await db.commit()

    # ── Nightly Consolidation ───────────────────────────────

    async def run_nightly_consolidation(self):
        """Review weak memories and consolidate. Runs as a nightly cron job."""
        if not self.groq_client:
            return

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, type, title, content, strength
                FROM memory_nodes
                WHERE strength < 0.3 AND type NOT LIKE 'archived_%'
                ORDER BY strength ASC
                LIMIT 30
            """)
            weak_nodes = await cursor.fetchall()

        if not weak_nodes:
            logger.info("Nightly consolidation: no weak memories to review")
            return

        nodes_text = "\n".join(
            f"[ID:{n[0]}] ({n[1]}) \"{n[2]}\" — {n[3]} (strength: {n[4]:.2f})"
            for n in weak_nodes
        )

        prompt = (
            "Review these weak memories and decide what to do with each.\n"
            "Return a JSON array of actions. Each item:\n"
            '{"id": <node_id>, "action": "keep|archive|delete", "reason": "1 line"}\n\n'
            "Rules:\n"
            "- archive: memory is too vague or outdated to be useful\n"
            "- delete: memory is completely irrelevant or superseded\n"
            "- keep: memory is still potentially useful despite low strength\n\n"
            f"Memories:\n{nodes_text}\n\nJSON array:"
        )

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You review and clean up AI memory. Output ONLY valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.2,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)
            if not raw:
                return

            actions = json.loads(raw)
            if not isinstance(actions, list):
                return

            async with aiosqlite.connect(self.db_path) as db:
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    node_id = action.get("id")
                    act = action.get("action", "keep")

                    if act == "archive":
                        await db.execute("""
                            UPDATE memory_nodes SET type = 'archived_' || type
                            WHERE id = ? AND type NOT LIKE 'archived_%'
                        """, (node_id,))
                    elif act == "delete":
                        await db.execute("DELETE FROM memory_nodes WHERE id = ?", (node_id,))
                        await db.execute("DELETE FROM memory_cues WHERE node_id = ?", (node_id,))
                        await db.execute(
                            "DELETE FROM memory_edges WHERE source_id = ? OR target_id = ?",
                            (node_id, node_id)
                        )

                await db.commit()
            logger.info(f"Nightly consolidation: processed {len(actions)} memories")

        except Exception as e:
            logger.error(f"Nightly consolidation failed: {e}")

    # ── Utilities ───────────────────────────────────────────

    async def clear_history(self):
        """Wipe everything — full memory reset."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM messages")
            await db.execute("DELETE FROM memory_cues")
            await db.execute("DELETE FROM memory_edges")
            await db.execute("DELETE FROM memory_nodes")
            await db.commit()
