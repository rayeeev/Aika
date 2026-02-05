# Aika Telegram Agent Server (Raspberry Pi)

Production-lean Telegram agent server using:
- `aiogram` v3 long polling
- Gemini (`google-genai`, model `gemini-3-flash-preview`) for default chat/tool orchestration
- Groq (`qwen/qwen3-32b`) for all summarization
- Groq (`openai/gpt-oss-120b`) for exception triage
- SQLite + SQLAlchemy for state/memory
- APScheduler for reminders and background jobs
- Per-invocation sandbox execution using ephemeral Docker containers

## Security Model

- Server runs in one container.
- User shell commands are **never run directly in server container** from tool calls.
- `run_shell` tool starts an ephemeral sandbox container with:
  - user folder only mounted (`/data/users/<id>` -> `/work`)
  - `network_mode=none`
  - dropped capabilities + `no-new-privileges`
  - memory/cpu/pids limits
  - timeout enforcement + forced cleanup
- Path isolation blocks absolute paths and traversal (`..`) for user/shared scopes.

## Critical Warning: Docker Socket

This service mounts `/var/run/docker.sock` to create sandbox containers. Anyone with effective control of this service can potentially gain host-level control.

Mitigations:
- Run on a dedicated Raspberry Pi.
- Restrict `ALLOWED_USER_IDS` and `ADMIN_USER_IDS` strictly.
- Keep host patched and locked down.
- Do not co-host sensitive workloads.

## Setup

1. Create bot token via `@BotFather`.
2. Create `.env` from `.env.example`.
3. Fill at minimum:
   - `TELEGRAM_BOT_TOKEN`
   - `ALLOWED_USER_IDS`
   - `ADMIN_USER_IDS`
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
4. Build images:

```bash
docker compose build aika
docker compose build sandbox-image
```

5. Start server:

```bash
docker compose up -d aika
```

Logs:

```bash
docker compose logs -f aika
```

## Raspberry Pi Install (Step by Step)

1. Update OS packages:

```bash
sudo apt update && sudo apt upgrade -y
```

2. Install Docker + Compose plugin:

```bash
sudo apt install -y docker.io docker-compose-plugin git
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

3. Re-login (or reboot) so Docker group permissions apply.

4. Clone project on the Pi:

```bash
git clone <your-repo-url> Aika
cd Aika
```

5. Create env file and set secrets:

```bash
cp .env.example .env
nano .env
```

6. Build images:

```bash
docker compose build aika
docker compose build sandbox-image
```

7. Start service:

```bash
docker compose up -d aika
```

8. Verify logs:

```bash
docker compose logs -f aika
```

9. Optional auto-start verification:

```bash
docker ps
```

## Environment Variables

- `TELEGRAM_BOT_TOKEN=`
- `ALLOWED_USER_IDS=` comma-separated Telegram IDs
- `ADMIN_USER_IDS=` comma-separated Telegram IDs (subset of allowed)
- `TELEGRAM_USER_ALIASES=` optional mapping for identity context, e.g. `123456789:Ray;987654321:Alice`
- `GEMINI_API_KEY=`
- `GEMINI_API_KEY_FALLBACK=` optional single fallback key
- `GEMINI_API_KEY_FALLBACKS=` optional comma-separated additional fallback keys
- `GROQ_API_KEY=`
- `DATA_DIR=/data`
- `USER_DIR=/data/users`
- `SHARED_DIR=/data/shared`
- `CACHE_TTL_HOURS=72`
- `SANDBOX_IMAGE=agent-sandbox-runner:latest`
- `SANDBOX_TIMEOUT_SECONDS=30`
- `SANDBOX_MEMORY=512m`
- `SANDBOX_CPUS=0.5`
- `LOG_LEVEL=INFO`
- `TIMEZONE=America/Los_Angeles`
- `GEMINI_MAX_CALLS_PER_MESSAGE=3`
- `GEMINI_MAX_RETRIES=2`
- `GEMINI_RETRY_BASE_DELAY_SECONDS=1.0`
- `GEMINI_QUOTA_COOLDOWN_SECONDS=60`
- `GEMINI_DAILY_QUOTA_COOLDOWN_SECONDS=3600`

## Commands

- `/help`
- `/status`
- `/run <shell command>` (admin only, runs in server container)
- `/files`
- `/save <relative_cache_path> <dest_rel_path>`
- `/shared put <rel_user_path> <rel_shared_path>`
- `/shared get <rel_shared_path> <rel_user_path>`
- `/remind <when> <text>`
- `/remind_user <user_id> <when> <text>` (admin)

Time parsing supports:
- `in 10m`
- `in 2h`
- `tomorrow 9am`
- ISO datetime

## Attachment Behavior

- Supported media types:
  - images (`photo`, `document` with `image/*`)
  - audio (`voice`, `audio`, `document` with `audio/*`)
- Attachment/media without caption:
  - downloaded to `/data/users/<id>/cache/...`
  - image/files: saved as pending attachment (no immediate bot reply)
  - audio (`voice`/`audio`/`audio/*`): sent immediately to Gemini with auto-transcription prompt (no pending wait)
  - on next text, pending paths are injected and eligible image/audio bytes are included for Gemini multimodal processing
  - pending items are consumed after one use
- Attachment/media with caption:
  - downloaded immediately
  - caption is sent to Gemini
  - image/audio bytes are sent in the same Gemini user content turn (multimodal)
  - local path reference is also included in prompt context

## Memory and Summarization

- Full messages stored in SQLite.
- Prompt includes:
  - major memory (<= 4 sentences)
  - compressed memory sentences
  - last 3 exchanges verbatim
  - today summary (<= 2 sentences) if present
- Rolling compression:
  - after each assistant response, oldest uncompressed exchange outside recent window is summarized into 1 sentence via Groq Qwen3-32B.
- Daily job (04:00 local): summarize previous day into exactly 2 sentences.
- Weekly job (Sunday 05:00 local): summarize last 7 daily summaries + previous major memory into exactly 4 sentences.
- Cache cleanup job (hourly): removes stale cache files and cleans references.

## Error Triage

On update or job exceptions:
- stack trace and context saved under `/data/admin/error_reports/<timestamp>.md`
- Groq GPT-OSS-120B analyzes root cause and patch suggestions
- admins receive rate-limited Telegram notification

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

If your `.env` uses Docker paths (`/data`, `/data/users`, `/data/shared`), local runs now auto-fallback to `./data` when not in Docker.  
You can also set these explicitly for local dev:

```ini
DATA_DIR=./data
USER_DIR=./data/users
SHARED_DIR=./data/shared
```

## Tests

Included tests:
- `tests/test_paths.py` path isolation
- `tests/test_sandbox_exec.py` sandbox container spec builder

Run:

```bash
pytest -q tests
```

## Troubleshooting

- If logs show `429 RESOURCE_EXHAUSTED` from Gemini:
  - This means your Gemini quota is exhausted (often free-tier daily request cap).
  - The bot now puts the exhausted key on cooldown and auto-rotates to `GEMINI_API_KEY_FALLBACK` / `GEMINI_API_KEY_FALLBACKS` if configured.
  - If all keys are exhausted, it returns a user-friendly message instead of failing updates.
  - Fix by waiting for quota reset or enabling billing/higher limits in Google AI Studio.

## Repository Layout

- `src/main.py` entrypoint
- `src/bot/handlers.py` central message router + command dispatch
- `src/agent/orchestrator.py` Gemini tool loop
- `src/tools/exec.py` sandbox executor
- `src/utils/paths.py` strict path isolation
- `src/memory/*` memory store + summarizer
- `src/jobs/*` scheduler + maintenance jobs
- `src/support/error_triage.py` incident reporting
- `sandbox/Dockerfile` sandbox runner image
