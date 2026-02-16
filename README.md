# 🤖 Aika - Autonomous AI Agent

Aika is an autonomous AI agent living on a Raspberry Pi 5, connected via Telegram. She has full system control, a conversation-based memory system, and her own personality.

## ✨ Features

- **Conversation-Based Memory** — Automatic conversation boundaries (30-min gap), rolling context summaries, episodic & semantic memory extraction, day/global memory consolidation
- **Voice Message Support** — Transcribes voice via Groq Whisper (whisper-large-v3-turbo)
- **API Key Rotation** — Multiple Gemini keys with automatic failover
- **Time-Gap Awareness** — Understands conversation breaks naturally
- **Tool Use** — Shell commands, file I/O, directory listing, server log access, memory recall
- **Self-Scheduling** — Can schedule wake-ups to remind or check on things
- **Graceful Shutdown** — Clean resource cleanup on SIGINT/SIGTERM

## 🧠 Memory Architecture

Aika uses a **conversation-centric memory system** with multiple layers of consolidation.

```
┌──────────────────────────────────────────────────────┐
│              CONTEXT COMPOSER                        │
│         (Assembled before each Gemini call)          │
│                                                      │
│    System Prompt + Tools                             │
│    + Conversation Context (CC summary + buffer)      │
│    + Today's conversation summaries                  │
│    + Semantic memories (stable facts)                │
│    + Appropriate episodic memories (Groq-filtered)   │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│           CONVERSATION LAYER                           │
│                                                        │
│  Immediate Buffer: last 10 interactions (20 msgs)      │
│  CC Summary: ≤3 sentence rolling summary of older      │
│  messages in the same conversation                     │
│                                                        │
│  30-min gap → close conversation → summarize + extract │
└──────────────┬─────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────────────────────┐
│           MEMORY LAYERS                                                   │
│                                                                           │
│  📅 Episodic  — extracted when conversations close (~4000 token budget)   │
│  💡 Semantic  — distilled from episodic at 4:00 AM (~300 token budget)    │
│  📋 Day Memory — 3-sentence summary of each day                           │
│  🌐 Global Memory — 4-sentence rolling overview                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### Lifecycle

1. **Live** — Messages go into the immediate buffer (last 10 interactions). When the buffer overflows, oldest messages are folded into a ≤3 sentence CC summary via Groq.
2. **Conversation Close** — After 30 min of inactivity, the full conversation is sent to Groq for a 4-sentence summary and episodic memory extraction.
3. **Daily (4:00 AM)** — All conversation summaries from the day are consolidated into a 3-sentence day memory. Episodic memories are analyzed to create semantic memories.
4. **Daily (4:10 AM)** — The latest day memory is merged into the global memory (4-sentence rolling summary).
5. **Weekly (4:20 AM Sunday)** — Groq reviews all episodic and semantic memories, removing duplicates and stale entries.

### Context Assembly (per prompt)

| Section | Source |
|---------|--------|
| System prompt | Static personality + tool rules |
| Today's conversations | Summaries of closed conversations from today |
| Semantic memories | All stable facts/preferences |
| Appropriate memories | Episodic memories filtered by Groq (`llama-3.1-8b-instant`) |
| Chat history | CC summary + immediate buffer messages |

## 📋 Requirements

- Python 3.11+
- Raspberry Pi 5 (or any Linux system)
- Telegram Bot Token
- Gemini API Key(s)
- Groq API Key

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd Aika
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEYS=key1,key2,key3   # Comma-separated, tries each until one works
GROQ_API_KEY=your_groq_api_key
ALLOWED_USER_ID=your_telegram_user_id

# Optional
AIKA_STARTUP_MESSAGE=true      # Set to 'false' to disable startup message
AIKA_DB_PATH=/path/to/aika.db  # Custom database path (default: ./aika.db)
```

> **💡 API Key Rotation**: You can provide multiple Gemini API keys separated by commas. If one key fails (rate limit, quota exceeded, etc.), Aika automatically tries the next one.

#### Getting Your Telegram User ID

1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. It will reply with your user ID

#### Creating a Telegram Bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow the prompts
3. Copy the bot token to your `.env` file

### 5. Run the Server

```bash
python -m src.main
```

Or for production with auto-restart:

```bash
sudo nano /etc/systemd/system/aika.service
```

Add the following:

```ini
[Unit]
Description=Aika AI Agent
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Aika
Environment=PATH=/home/pi/Aika/venv/bin
ExecStart=/home/pi/Aika/venv/bin/python -m src.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable aika
sudo systemctl start aika
```

## 📁 Project Structure

```
Aika/
├── src/
│   ├── main.py      # Bot, message handling, tools, LLM orchestration
│   └── memory.py    # Conversations, CC, episodic/semantic, day/global memory
├── .env
├── requirements.txt
├── aika.db          # SQLite (auto-created)
├── aika.log         # Server logs (cleared on restart)
└── README.md
```

## 🔧 Tools

| Tool | Description |
|------|-------------|
| `execute_shell_command(cmd)` | Run shell commands (60s timeout) |
| `read_file(path)` | Read file contents |
| `write_file(path, content)` | Write content to a file |
| `list_directory(path)` | List directory contents |
| `schedule_wake_up(seconds, thought)` | Schedule self-initiated check-in |
| `read_server_logs(lines)` | Read Aika's own server logs |
| `recall_memory(query_type, date, time_range)` | Retrieve global/day/conversation memories |
| `list_memories(memory_type)` | List episodic or semantic memories with IDs |
| `delete_memory(memory_type, memory_id)` | Delete a specific memory by type and ID |
| `edit_memory(memory_type, memory_id, new_content)` | Edit a specific memory's content |

## 🔒 Security

- **Single User** - Only responds to `ALLOWED_USER_ID`
- **No Path Traversal** - File operations use absolute paths
- **Timeout Protection** - Shell commands timeout after 60 seconds
- **Graceful Shutdown** - Clean resource cleanup

## 📊 Monitoring

Check logs:

```bash
# If running with systemd
sudo journalctl -u aika -f

# Aika's own log file (also readable by Aika via read_server_logs tool)
tail -f aika.log

# If running directly, logs output to stdout
```

## 📜 License

Private project. All rights reserved.

---

*Built with ❤️ for autonomous AI exploration*