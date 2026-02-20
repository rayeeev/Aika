# Aika - Inference V4 (Groq-First)

Aika is a Telegram AI agent for Raspberry Pi with Groq-first inference, Gemini fallback, and a latency-optimized memory pipeline.

## What Changed in V4

- Groq is now the primary inference provider
- Per-turn memory path is optimized to one Groq pre-call before main generation
- Background Conversation Context compaction checks every 5 turns
- Compaction keeps latest 15-20 messages live and folds older ones into summary
- Capability-based routing chooses Gemini when tool-heavy/system actions are needed
- Prompt-level model/provider controls are supported

## Request Lifecycle (Runtime Contract)

For each incoming turn:

1. Input is debounced and buffered (0.8s)
2. **Groq PreCall** (`memory.prepare_inference_context`) runs once
3. Main call runs on Groq by default, or Gemini when routed/requested
4. If Groq fails, Gemini fallback runs and receives Groq error context
5. User/model messages are stored asynchronously
6. Every 5 turns, background compaction runs once (`compact_conversation_context_if_due`)

Foreground path is designed to avoid long blocking memory work.

## Provider and Model Routing

### Default policy

- Capability-based
- Groq first
- Gemini when tools are likely required or explicitly requested

### Prompt controls

You can place these in user prompts:

- `[PROVIDER=groq]` or `[PROVIDER=gemini]`
- `[MODEL=<groq_model_name>]`
- `[REASONING=fast]` or `[REASONING=deep]`

Controls are removed from message text before model generation.

### Hardcoded Groq allowlist

Defined in `src/main.py`:

- `llama-3.1-8b-instant`
- `llama-3.3-70b-versatile`
- `qwen/qwen3-32b`
- `deepseek-r1-distill-llama-70b`

If a requested model is not in this allowlist, Aika falls back to default model selection.

## Memory Architecture

Memory remains Groq-driven.

### Layers

- **Conversation Context (CC)**
  - `conversation_context.summary_text` (rolled summary)
  - Active recent messages in `messages`
- **Episodic memories**
- **Semantic memories**
- **Day memories**
- **Global memory**

### Live context behavior

- PreCall selects relevant semantic + episodic memories in one Groq JSON call
- Selected memories and compact context summary are injected into main prompt

### Background compaction

- Trigger: every 5 turns
- Guard: compaction only runs when active conversation already has at least 15 messages
- Action: fold oldest active-conversation messages into CC summary
- Preserve: latest 15-20 messages (default 18)
- Runs asynchronously and does not block response generation

### Conversation boundaries

- 30 minutes inactivity closes active conversation
- Close pipeline extracts:
  - conversation summary
  - episodic memories

### Scheduled jobs

- `close_stale_conversations`: every 5 minutes
- `run_daily_summary`: 4:00 AM
- `run_global_update`: 4:10 AM
- `run_weekly_cleanup`: Sunday 4:20 AM

## Sync vs Async Design

### Async

- Telegram handlers
- DB access (`aiosqlite`)
- Inference orchestration
- Background compaction and proactive tasks

### Sync (wrapped safely)

- Tool implementations for Gemini AFC
- Groq/Gemini SDK calls executed through executor where needed

## Latency Controls

Configured in `src/main.py`:

- PreCall timeout: `2.5s`
- Main call timeout: `9.0s`
- Fallback call timeout: `9.0s`
- End-to-end generation cap: `24.0s`

## Requirements

- Python 3.11+
- Telegram bot token
- Groq API key
- Gemini API key(s) for fallback/tool routing (optional but recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Required:

```bash
TELEGRAM_BOT_TOKEN=...
GROQ_API_KEY=...
ALLOWED_USER_ID=...
```

Recommended (for fallback/tool routing):

```bash
GEMINI_API_KEYS=key1,key2,key3
```

Optional:

```bash
AIKA_STARTUP_MESSAGE=true
AIKA_DB_PATH=/path/to/aika.db
```

## Run

```bash
python -m src.main
```

## Project Structure

```text
Aika/
├── src/
│   ├── main.py      # Telegram bot + inference routing + tools
│   └── memory.py    # Memory DB + Groq pre-call + compaction + summaries
├── README.md
├── requirements.txt
├── aika.db          # auto-created
└── aika.log
```

## Tool Surface (Gemini AFC)

- `execute_shell_command(cmd)`
- `read_file(path)`
- `write_file(path, content)`
- `list_directory(path)`
- `schedule_wake_up(seconds, thought, provider=\"\", model=\"\", reasoning=\"\")`
- `read_server_logs(lines)`
- `recall_memory(query_type, date, time_range)`
- `list_memories(memory_type)`
- `delete_memory(memory_type, memory_id)`
- `edit_memory(memory_type, memory_id, new_content)`

`schedule_wake_up(...)` can force delayed-task routing to a specific provider/model (for example `provider=\"gemini\"` or `model=\"groq/compound\"`) when default routing chooses the wrong main AI.

Examples:
- `schedule_wake_up(300, "Check if deploy finished")` uses default routing
- `schedule_wake_up(120, "Search latest docs and summarize", provider="groq", model="groq/compound", reasoning="deep")`
- `schedule_wake_up(600, "Run this task with Gemini", provider="gemini")`

## Monitoring

```bash
tail -f aika.log
```

If running with systemd:

```bash
sudo journalctl -u aika -f
```

## Notes

- Memory compaction and retrieval logic is Groq-based.
- Groq remains primary generation path unless routing decides otherwise.
- Gemini is used for fallback and tool-heavy turns.
