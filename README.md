# 🤖 Aika - Autonomous AI Agent

Aika is an autonomous AI agent living on a Raspberry Pi 5, connected via Telegram. She has her own personality, memory system, and can execute commands on the host system.

## ✨ Features

- **Autonomous Personality** - Straightforward, efficient, and loyal
- **Voice Message Support** - Transcribes voice messages using Groq Whisper (whisper-large-v3-turbo)
- **API Key Rotation** - Supports multiple Gemini API keys with automatic failover
- **Persistent Memory System**
  - Buffer: Last 5 interactions (10 messages), auto-expires after 1 hour of inactivity
  - Weekly Summary: Rolling compressed context
  - Global Summary: Long-term core memories
- **Time-Gap Awareness** - Understands conversation breaks (night → morning = new conversation)
- **Tool Use** - Shell commands, file read/write, directory listing, server log access
- **Self-Scheduling** - Can schedule wake-ups to remind or check on things
- **Graceful Shutdown** - Proper cleanup on SIGINT/SIGTERM

## 📋 Requirements

- Python 3.11+
- Raspberry Pi 5 (or any Linux system)
- Telegram Bot Token
- Gemini API Key(s)
- Groq API Key

## 🚀 Quick Start

### 1. Clone the Repository

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
│   ├── main.py      # Bot entry point, message handling, tools, LLM orchestration
│   └── memory.py    # Memory management (buffer, summaries, time-based expiry)
├── .env             # Environment variables (create this)
├── .gitignore
├── requirements.txt
├── aika.db          # SQLite database (auto-created)
├── aika.log         # Server logs (auto-created, rotated at 512KB)
└── README.md
```

## 🧠 Memory Architecture

Aika uses a three-tier memory system with time-based expiry:

```
┌─────────────────────────────────────────────────────┐
│                   GLOBAL SUMMARY                     │
│         (4 sentences, persists forever)              │
│         Updated weekly from weekly summary           │
└─────────────────────────────────────────────────────┘
                         ▲
                         │ Weekly Reset (Sunday midnight)
                         │
┌─────────────────────────────────────────────────────┐
│                   WEEKLY SUMMARY                     │
│       (3 sentences, resets weekly)                   │
│       Updated on buffer overflow OR time expiry      │
└─────────────────────────────────────────────────────┘
                         ▲
                         │ Overflow (>10 messages)
                         │ OR messages >1 hour old
                         │
┌─────────────────────────────────────────────────────┐
│               IMMEDIATE BUFFER                       │
│         (Last 10 messages / 5 interactions)          │
│         Auto-expires after 1 hour of inactivity      │
│         Oldest popped first on overflow              │
└─────────────────────────────────────────────────────┘
```

**Time-gap awareness**: Messages separated by >30 minutes get `[TIME GAP]` markers in the chat history, helping Aika treat overnight or long gaps as separate conversations.

## 🔧 Available Tools

| Tool | Description |
|------|-------------|
| `execute_shell_command(cmd)` | Run shell commands on the host |
| `read_file(path)` | Read file contents |
| `write_file(path, content)` | Write content to a file |
| `list_directory(path)` | List directory contents |
| `schedule_wake_up(seconds, thought)` | Schedule a self-initiated check-in |
| `read_server_logs(lines)` | Read Aika's own server logs (last N lines) |

## 🎙️ Voice Messages

1. Send a voice message to the bot
2. Transcribed using Groq's Whisper (whisper-large-v3-turbo)
3. Stored in memory with `[VOICE]` prefix
4. Processed like regular text

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

Check database:

```bash
sqlite3 aika.db "SELECT * FROM messages ORDER BY id DESC LIMIT 10;"
sqlite3 aika.db "SELECT * FROM summaries;"
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