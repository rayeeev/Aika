# 🤖 Aika - Autonomous AI Agent

Aika is an autonomous AI agent living on a Raspberry Pi 5, connected via Telegram. She has full system control, a brain-inspired memory system, and her own personality.

## ✨ Features

- **Brain-Inspired Memory** — Structured memory nodes (semantic/episodic/procedural) with cue-based retrieval, association edges, strength decay, and nightly consolidation
- **Voice Message Support** — Transcribes voice via Groq Whisper (whisper-large-v3-turbo)
- **API Key Rotation** — Multiple Gemini keys with automatic failover
- **Time-Gap Awareness** — Understands conversation breaks naturally
- **Tool Use** — Shell commands, file I/O, directory listing, server log access
- **Self-Scheduling** — Can schedule wake-ups to remind or check on things
- **Graceful Shutdown** — Clean resource cleanup on SIGINT/SIGTERM

## 🧠 Memory Architecture

Aika uses a **brain-inspired memory system** — not flat summaries, but structured knowledge nodes with associative recall.

```
┌─────────────────────────────────────────────────────┐
│              CONTEXT COMPOSER                        │
│         (Budget-aware prompt assembly)               │
│                                                     │
│    Working Set (last 20 messages)                    │
│    + Recalled Memory Cards (cue-matched, ranked)    │
└──────────────┬──────────────────────────────────────┘
               │ retrieves from
               ▼
┌─────────────────────────────────────────────────────┐
│              MEMORY NODES                            │
│                                                     │
│  💡 Semantic  — stable facts, preferences, bio      │
│  📅 Episodic  — notable events, decisions, moments  │
│  ⚙️ Procedural — behavioral patterns, tool habits   │
│                                                     │
│  Each node has: strength, access count, timestamps  │
│  Strength decays over time, reinforced on recall    │
└──────────────┬──────────────────────────────────────┘
               │ linked by
               ▼
┌─────────────────────────────────────────────────────┐
│           ASSOCIATION EDGES + CUE INDEX              │
│                                                     │
│  Keyword/entity cues for fast retrieval             │
│  Weighted edges between related memories            │
│  Spreading activation (1-hop) for "scent → story"   │
│  Edges decay faster than nodes                      │
└─────────────────────────────────────────────────────┘
```

### How It Works

1. **Ingest** — After each turn, Groq extracts structured memories as JSON (facts, events, patterns). Deduplicates by cue overlap. Creates association edges between related nodes.
2. **Retrieve** — Before each Gemini call, keywords from the user's message are matched against the cue index. Top candidates + 1-hop neighbors are scored and ranked. Top 10 are formatted as memory cards in the prompt.
3. **Decay** — Node strength decays based on time since last access (high-access nodes decay slower). Edges decay faster. Dead edges are pruned. Dead nodes are archived.
4. **Consolidation** — Nightly at 4 AM, Groq reviews weak memories and decides: keep, archive, or delete.

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
│   └── memory.py    # Memory nodes, cue retrieval, decay, consolidation
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

## 🛠️ Troubleshooting

### Bot not responding?

1. Check if the bot is running: `sudo systemctl status aika`
2. Verify your `ALLOWED_USER_ID` matches your Telegram user ID
3. Check logs for errors

### Voice messages failing?

1. Ensure `GROQ_API_KEY` is set correctly
2. Check temp directory permissions: `/tmp/aika_audio/`
3. Verify Groq API quota

### Memory not persisting?

1. Check `aika.db` exists and is writable
2. Verify `AIKA_DB_PATH` if using custom path

## 📜 License

Private project. All rights reserved.

---

*Built with ❤️ for autonomous AI exploration*