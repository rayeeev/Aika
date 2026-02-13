# 🧠 Aika Memory Architecture v2 — Design Document

> **Status:** Design phase — not yet implemented.
> **Goal:** Replace Aika's flat 3-tier memory with a brain-inspired system that maximizes awareness (knowledge density per context token) while staying practical on a Raspberry Pi 5.

---

## 1. What's Wrong with the Current System

The current memory has three tiers: **Buffer** (10 raw messages) → **Weekly Summary** (3 sentences) → **Global Summary** (4 sentences). This has fundamental problems:

| Problem | Why it matters |
|---------|---------------|
| **No selective recall** | Every turn dumps the same last 10 messages + 2 summaries into the prompt, regardless of what's being discussed. Aika can't "remember" something from 3 weeks ago even if it's directly relevant. |
| **Lossy compression is one-way** | Once messages are summarized into the weekly summary, the original detail is gone forever. A 3-sentence summary can't capture "you mentioned wanting to buy a keyboard on Feb 3rd." |
| **No associations** | Memories have no links to each other. There's no way for a keyword or topic to trigger recall of a related past event — the "scent → story" effect doesn't exist. |
| **No concept of importance** | A casual "lol" and a critical "remember: my server password is X" are treated identically. Both get the same buffer slot and the same summarization treatment. |
| **Time-based expiry is too aggressive** | Messages older than 1 hour are force-expired regardless of whether they contained important information. A deeply important conversation at 2 PM is gone by 3 PM. |
| **Context is wasted** | The global + weekly summaries are always injected, even when irrelevant. They consume tokens without adding value on most turns. |

**The core issue:** Storage and context are conflated. The buffer IS the context. There's no retrieval — just a fixed window.

---

## 2. Design Principles

1. **Separate storage from context.** Store everything. Retrieve selectively. Context is assembled per-turn by a "composer" that acts like attention.
2. **Memories are nodes, not a log.** Instead of a message timeline, we store discrete memory nodes — facts, events, preferences, procedures — each with metadata.
3. **Associations are first-class.** Links between memories have weights that strengthen with use and decay with time. This enables pattern-completion retrieval.
4. **Strength gates retrieval, not deletion.** Don't delete weak memories — just make them harder to retrieve. Strong memories surface easily; weak ones require exact cues.
5. **Budget is sacred.** Every token in the prompt must earn its place. The Context Composer enforces a hard token budget with priority allocation.
6. **Decay is healthy.** Forgetting irrelevant associations is a feature, not a bug. It keeps the mind clean and retrieval fast.

---

## 3. Architecture Overview

```
                        ┌──────────────────────────────┐
                        │       CONTEXT COMPOSER        │
                        │   (Budgeted Prompt Assembly)  │
                        │                              │
                        │  Token Budget Allocation:    │
                        │  ┌────┐ ┌────┐ ┌────┐       │
                        │  │40% │ │40% │ │20% │       │
                        │  │Task│ │Sem.│ │Epi.│       │
                        │  └──┬─┘ └──┬─┘ └──┬─┘       │
                        └─────┼──────┼──────┼──────────┘
                              │      │      │
                    ┌─────────┘      │      └──────────┐
                    ▼                ▼                  ▼
            ┌──────────┐    ┌──────────────┐    ┌────────────┐
            │ WORKING  │    │   SEMANTIC    │    │  EPISODIC  │
            │   SET    │    │    STORE      │    │   STORE    │
            │          │    │              │    │            │
            │ Current  │    │ Stable facts │    │ Timestamped│
            │ turns +  │    │ preferences  │    │  "scenes"  │
            │ task     │    │ definitions  │    │  events    │
            │ state    │    │ commitments  │    │  moments   │
            └──────────┘    └──────────────┘    └────────────┘
                                    │                  │
                                    └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │  PROCEDURAL      │
                                    │  STORE            │
                                    │                  │
                                    │ "How I do things"│
                                    │ Format prefs     │
                                    │ Tool habits      │
                                    │ Routines         │
                                    └──────────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │      ASSOCIATION INDEX       │
                              │                              │
                              │  Keyword/entity cue index    │
                              │  Weighted edges between      │
                              │  memory nodes                │
                              │  Decay + reinforcement       │
                              └──────────────────────────────┘
```

---

## 4. Memory Node Types

Every memory is a **node** with a type, metadata, and a strength score.

### 4.1 Episodic Memory (EM)
**What:** Time-stamped "scenes" — chunks of conversation or events that happened.
**Examples:**
- "On Feb 5, Erden asked me to set up a cron job for backups."
- "At 2:30 AM, Erden was debugging SSL certificates and was frustrated."

**Properties:**
- Has a specific timestamp and duration
- Contains emotional coloring / context
- High detail, medium lifespan
- Useful when exact history matters ("what did we do last Tuesday?")

### 4.2 Semantic Memory (SM)
**What:** Distilled, stable facts extracted from episodes. The "truths" that persist.
**Examples:**
- "Erden's timezone is America/Los_Angeles."
- "Erden prefers direct, no-fluff responses."
- "The server runs on Raspberry Pi 5 at home."
- "Erden is working on a car wash business website."

**Properties:**
- No specific timestamp (timeless truths)
- High confidence, long lifespan
- Updated/corrected when contradicted by new information
- Most cost-effective memory type (highest value per token)

### 4.3 Procedural Memory (PM)
**What:** "How I do things" — learned patterns, format preferences, routines.
**Examples:**
- "When Erden asks to check server health, run: `htop`, `df -h`, `free -m`."
- "Erden prefers code without excessive comments."
- "For deployment, always check systemd service status first."

**Properties:**
- Behaviorally relevant — affects HOW Aika responds, not WHAT she knows
- Rarely needs to be shown in context (influences system prompt instead)
- Very stable, rarely decays

### 4.4 Working Set (WS)
**What:** The immediate conversation buffer. Last ~10-20 messages of the active conversation, plus any active task state (e.g., "currently diagnosing a networking issue").
**Not stored as nodes** — this is just the raw recent messages, similar to current buffer, but smarter about what counts as "active conversation" (uses time-gap detection).

---

## 5. Data Model (SQLite)

### Table: `memory_nodes`
```sql
CREATE TABLE memory_nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT NOT NULL,            -- 'episodic', 'semantic', 'procedural'
    title       TEXT NOT NULL,            -- 1-line summary for quick scanning
    content     TEXT NOT NULL,            -- Full content (2-5 sentences max)
    strength    REAL NOT NULL DEFAULT 1.0,-- Decays over time, reinforced on use
    created_at  REAL NOT NULL,           -- Unix timestamp
    last_accessed REAL NOT NULL,         -- Last time this was retrieved into context
    access_count INTEGER DEFAULT 0,      -- How many times retrieved
    source_turn INTEGER,                 -- Which conversation turn created this
    metadata    TEXT                      -- JSON: entities, tags, emotional_weight, etc.
);
```

### Table: `memory_edges`
```sql
CREATE TABLE memory_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    target_id   INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    weight      REAL NOT NULL DEFAULT 1.0,  -- Strengthens with co-retrieval, decays over time
    relation    TEXT,                        -- Optional: 'caused_by', 'related_to', 'contradicts', 'refines'
    created_at  REAL NOT NULL,
    last_used   REAL NOT NULL,
    UNIQUE(source_id, target_id)
);
```

### Table: `memory_cues`
```sql
CREATE TABLE memory_cues (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id     INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    cue_type    TEXT NOT NULL,              -- 'keyword', 'entity', 'tag', 'trigger_phrase'
    cue_value   TEXT NOT NULL               -- The actual cue text (lowercased)
);

CREATE INDEX idx_cues_value ON memory_cues(cue_value);
CREATE INDEX idx_cues_node ON memory_cues(node_id);
```

### Table: `messages` (Working Set — similar to current)
```sql
CREATE TABLE messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    role      TEXT NOT NULL,
    content   TEXT NOT NULL,
    timestamp REAL NOT NULL
);
```

### Table: `memory_blobs` (Optional — raw evidence archive)
```sql
CREATE TABLE memory_blobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id     INTEGER REFERENCES memory_nodes(id) ON DELETE SET NULL,
    raw_text    TEXT NOT NULL,              -- Original transcript chunk
    timestamp   REAL NOT NULL
);
```

---

## 6. Core Pipelines

### 6.1 Ingest Pipeline (runs after every turn, in background)

```
New turn complete (user message + model response)
    │
    ▼
┌─────────────────────────────────────────┐
│  STEP 1: Store in Working Set           │
│  (Same as current buffer insert)        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STEP 2: Extract Memories (via Groq)    │
│                                         │
│  Prompt Groq to extract from this turn: │
│  - Any new FACTS (→ semantic)           │
│  - Any EVENT worth remembering (→ epi.) │
│  - Any BEHAVIOR PATTERN (→ procedural)  │
│  - Cue keywords for each                │
│  - Importance score (1-5)               │
│                                         │
│  Output: structured JSON                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STEP 3: Deduplicate & Merge            │
│                                         │
│  Check if extracted fact already exists: │
│  - Match by cues/keywords               │
│  - If match: UPDATE existing node       │
│    (reinforce strength, merge content)  │
│  - If new: INSERT new node              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STEP 4: Link                           │
│                                         │
│  Connect new/updated nodes to existing  │
│  nodes that share entities or keywords  │
│  (INSERT into memory_edges)             │
└─────────────────────────────────────────┘
```

**Critical:** Steps 2-4 use **Groq** (fast, cheap, `qwen3-32b`) — NOT Gemini. This keeps costs near zero. The extraction prompt would look like:

```
Given this conversation turn, extract any memories worth storing.
Return JSON array. Each item has:
- type: "semantic" | "episodic" | "procedural"
- title: 1-line summary
- content: 2-4 sentences of detail
- importance: 1-5 (5 = critical, 1 = trivial)
- keywords: list of 3-8 cue words
- entities: list of named entities (people, places, projects)

Turn:
[user]: {user_message}
[model]: {model_response}

Rules:
- Only extract if there's something genuinely worth remembering.
- Most casual turns produce 0 memories. That's fine.
- Semantic: stable facts, preferences, commitments.
- Episodic: notable events, decisions, emotional moments.
- Procedural: learned patterns about how the user wants things done.
- Return empty array [] if nothing worth storing.
```

### 6.2 Retrieval Pipeline (runs at the START of every turn, before Gemini)

```
User sends message
    │
    ▼
┌─────────────────────────────────────────┐
│  STEP 1: Extract Query Cues             │
│                                         │
│  From the user's message, extract:      │
│  - Keywords (simple tokenization)       │
│  - Named entities                       │
│  - Intent category                      │
│  (This can be rule-based, no LLM needed)│
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STEP 2: Candidate Pull (Cheap Pass)    │
│                                         │
│  Query memory_cues table for matches    │
│  → Get candidate node IDs              │
│  Score: cue_matches + strength          │
│  → Top 20-30 candidates                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STEP 3: Spreading Activation           │
│  (1-2 hops on memory_edges)             │
│                                         │
│  From top candidates, follow edges:     │
│  - Hop 1: neighbors with weight > 0.3   │
│  - Hop 2 (optional): if few candidates  │
│  Add connected nodes to candidate pool  │
│  This is the "scent → story" mechanism  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STEP 4: Rank & Select                  │
│                                         │
│  Final score per candidate:             │
│  score = cue_match_score                │
│        + 0.6 * strength                 │
│        + 0.3 * recency_boost            │
│        - redundancy_penalty             │
│                                         │
│  Select top 6-12 nodes for context      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STEP 5: Reinforce                      │
│                                         │
│  Every node that made it into context:  │
│  - strength += 0.1                      │
│  - last_accessed = now                  │
│  - access_count += 1                    │
│  Every edge between co-retrieved nodes: │
│  - weight += 0.05                       │
└─────────────────────────────────────────┘
```

**Key insight:** Step 2 (candidate pull) is pure SQLite keyword matching — no embeddings needed, no LLM call. This is the "cheap pass" that handles 90% of retrieval. We can add embedding-based search as an enhancement later, but keyword + entity matching on the `memory_cues` table is fast, free, and surprisingly effective for a single-user system.

### 6.3 Decay Pipeline (runs periodically — e.g., every hour or on each turn)

```sql
-- Decay all node strengths (half-life based)
-- Nodes with high access_count decay slower
UPDATE memory_nodes
SET strength = strength * POWER(0.995, ((:now - last_accessed) / 3600.0) / (1 + LOG(1 + access_count)))
WHERE strength > 0.01;

-- Decay all edge weights
UPDATE memory_edges
SET weight = weight * POWER(0.99, (:now - last_used) / 3600.0)
WHERE weight > 0.01;

-- Archive dead nodes (strength < 0.05, not accessed in 30+ days)
-- Don't delete — just mark as archived so they can be resurrected if directly queried
UPDATE memory_nodes
SET type = 'archived_' || type
WHERE strength < 0.05
  AND last_accessed < :now - (30 * 86400)
  AND type NOT LIKE 'archived_%';

-- Delete truly dead edges
DELETE FROM memory_edges WHERE weight < 0.01;
```

**Edges decay FASTER than nodes.** This is critical. A memory can survive indefinitely if it's strong, but its associations weaken unless reinforced. This means:
- Old strong memories become "isolated" over time — retrievable only by direct cue match, not by association
- Frequently co-activated memories form strong clusters
- The association graph stays clean and fast

---

## 7. Context Composer

The Context Composer replaces the current "dump everything" approach. It runs before each Gemini API call and assembles the prompt within a **strict token budget**.

### Token Budget Allocation
```
Total context budget: ~2000 tokens (adjustable)

┌────────────────────────────────────────────────────┐
│  ZONE 1 — Task State (40%, ~800 tokens)            │
│  ├─ Current user message                           │
│  ├─ Last 4-8 messages from Working Set             │
│  └─ Active task state (if any)                     │
├────────────────────────────────────────────────────┤
│  ZONE 2 — Semantic/Procedural (40%, ~800 tokens)   │
│  ├─ Relevant facts (from retrieval pipeline)       │
│  ├─ User preferences (from retrieval pipeline)     │
│  └─ Procedural memories (format as instructions)   │
├────────────────────────────────────────────────────┤
│  ZONE 3 — Episodic Color (20%, ~400 tokens)        │
│  ├─ Related past events (only if relevant)         │
│  └─ Can be 0 if no episodes match                  │
└────────────────────────────────────────────────────┘
```

### Memory Card Format
Each retrieved memory is presented to Gemini as a compact card:

```
📌 [SEMANTIC] Erden's car wash business
   Erden runs a car wash business and is building a website for it using Next.js.
   He prefers modern, premium-looking designs with dark mode support.
   Strength: 0.92 | Last seen: 2 days ago

📌 [EPISODIC] SSL certificate debugging session (Feb 5)
   Erden was frustrated while debugging SSL certs at 2:30 AM.
   Resolved by generating a new self-signed cert with proper SAN fields.
   Strength: 0.71 | Last seen: 8 days ago

📌 [PROCEDURAL] Deployment preference
   Always check systemd status first. Use `journalctl -u <service> -f` for live logs.
   Strength: 0.85 | Last seen: 3 days ago
```

This format is compact (~50-80 tokens per card) and gives Gemini everything it needs to use the memory effectively, including confidence signals (strength) that help it weight information.

---

## 8. How "Scent → Story" Works (Associative Recall)

**Example scenario:** Erden says "remember that keyboard issue?"

1. **Cue extraction:** keywords = ["keyboard", "issue", "remember"]
2. **Cue query:** `SELECT node_id FROM memory_cues WHERE cue_value IN ('keyboard', 'issue')` → finds node #47 ("Erden mentioned wanting a mechanical keyboard") and node #82 ("Keyboard shortcut conflict in VS Code")
3. **Spreading activation:** Follow edges from #47 → finds #48 ("Erden was comparing Cherry MX switches") with edge weight 0.7, and #50 ("Budget discussion — Erden set aside $200 for peripherals") with edge weight 0.4
4. **Result:** Even though Erden only said "keyboard issue," Aika recalls the full cluster: the keyboard desire, the switch comparison, and the budget context. Exactly like smelling cookies and remembering your grandmother's kitchen.

**Why edges decay matters here:** If Erden discussed keyboards once, 6 months ago, and never again, the edges from "keyboard" to "budget" and "switches" will have decayed to near zero. Only the core memory (#47) survives if it was strong enough. The associated details fade — just like a human who vaguely remembers wanting a keyboard but not the specifics.

---

## 9. Comparison: Current vs. Proposed

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Storage** | 10 messages + 2 summaries | Unlimited nodes + edges + cues |
| **Retrieval** | Fixed window (last 10) | Cue-based, relevance-ranked |
| **Long-term memory** | 4-sentence global summary | Individual semantic/procedural nodes with full detail |
| **Associations** | None | Weighted edge graph with decay |
| **Forgetting** | Aggressive (1 hour expiry) | Gradual strength decay, edges decay faster |
| **Context assembly** | Dump everything | Budget-aware composer, only relevant memories |
| **Detail preservation** | Lost after summarization | Preserved in individual nodes indefinitely |
| **Cost per turn** | 1 Groq call (summary), 1 Gemini call | 0-1 Groq calls (extraction), 1 Gemini call, 0 for retrieval |
| **Recall from old context** | Impossible (summarized away) | Possible if cue matches and strength > threshold |

---

## 10. Implementation Phases

### Phase 1: Foundation (Minimal Viable Memory)
**Build these 3 things to get 80% of the effect:**

1. **Semantic memory extraction & storage**
   - On each turn, use Groq to extract facts/preferences → store as nodes with cues
   - Replace global/weekly summary with actual semantic nodes
   - Deduplicate on insert (match by cues)

2. **Cue-based retrieval + Context Composer**
   - On each turn, extract keywords from user message
   - Query `memory_cues` for matches → rank by strength + cue overlap
   - Assemble context with budget: working set + top semantic matches
   - Replace the current "dump summaries" approach

3. **Decay on strength + reinforcement on use**
   - Decay node strength based on time since last access
   - Reinforce when retrieved into context
   - Simple SQL UPDATE on each retrieval cycle

### Phase 2: Associations (The "Scent → Story" Layer)
4. **Memory edges with spreading activation**
   - Link co-occurring nodes (same turn, shared entities)
   - 1-hop activation during retrieval
   - Edge decay (faster than node decay)

5. **Episodic memory**
   - Store notable events as timestamped scenes
   - Retrieve when user references past events or time periods

### Phase 3: Polish
6. **Procedural memory**
   - Extract behavioral patterns over time
   - Inject as system prompt modifications (not context cards)

7. **Reconsolidation**
   - Periodically merge/refine semantic nodes that are near-duplicates
   - Update facts when contradicted by newer information

8. **Raw blobs (evidence archive)**
   - Store original transcript chunks linked to nodes
   - Allows "expanding" a memory card to full detail on demand

---

## 11. Practical Considerations for Raspberry Pi 5

### Why This Works on a Pi
- **No embeddings needed for v1.** Keyword/entity matching on SQLite with proper indexes is plenty fast for a single-user system with <10,000 memories. We can add embeddings later if needed.
- **Groq does the heavy lifting.** Memory extraction, which is the most LLM-intensive part, runs on Groq's `qwen3-32b` (free tier, fast). Gemini is only used for the actual conversation.
- **SQLite is perfect.** Single-user, single-writer, lightweight. The Pi 5 has plenty of RAM for this workload.
- **Background processing.** Like the current system, all memory operations (extraction, linking, decay) run as background tasks after the reply is sent. The user never waits.

### Token Cost Analysis
- **Current system:** ~200 tokens for summaries + ~500 tokens for buffer = ~700 tokens of memory context per turn. Quality: LOW (generic summaries).
- **Proposed system:** ~400 tokens for working set + ~400-600 tokens for 6-8 memory cards = ~800-1000 tokens. Quality: HIGH (specific, relevant, ranked).
- **Net effect:** ~30-40% more tokens spent on memory, but information density per token increases by 5-10x because we inject RELEVANT memories instead of a generic summary.

### Groq Cost
- **Current:** 1 Groq call per buffer overflow (summarization).
- **Proposed:** 1 Groq call per turn (memory extraction). Slightly more calls, but each is small and fast. Groq free tier handles this easily for a single-user bot.

---

## 12. What We're NOT Doing (And Why)

| Idea | Why we skip it |
|------|---------------|
| **Vector embeddings** | Overkill for v1. Keyword matching with good cue extraction handles single-user recall well. Can add later as Phase 4. |
| **Multiple users with shared memory** | Current system is single-user. If needed later, add a `user_id` column. |
| **LLM-based retrieval ranking** | Too expensive. SQL-based scoring (cue matches + strength + recency) is fast and free. |
| **Separate embedding model on Pi** | Adds complexity and RAM pressure. Not needed until we have >10,000 memories. |
| **Real-time consolidation** | Groq extraction after each turn is enough. Periodic cleanup (hourly/daily) handles the rest. |

---

## 13. Summary: The Philosophy

> **Current Aika:** Has amnesia with a sticky note on the fridge.
>
> **Proposed Aika:** Has a mind that naturally remembers what matters, forgets what doesn't, and can be reminded of old memories by the right cue — just like a human brain, but with the machine advantage of never truly deleting anything strong enough to persist.

The key architectural shift is: **don't compress memories into summaries — extract them into structured nodes and retrieve selectively.** Summaries destroy information. Nodes preserve it. A good retriever surfaces the right nodes at the right time, while a budget composer ensures we never waste tokens on irrelevant recall.

This is not just an optimization. It's a fundamentally different relationship between storage and attention.
