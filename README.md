# Aika Telegram Agent Server (Raspberry Pi)

Production-lean Telegram agent server using:
- `aiogram` v3 long polling
- Gemini (`google-genai`, model `gemini-3-flash-preview`) for default chat/tool orchestration
- Groq (`qwen/qwen3-32b`) for all summarization
- Groq (`openai/gpt-oss-120b`) for error triage and admin solve planning
- SQLite + SQLAlchemy for state/memory
- APScheduler for reminders and background jobs
- Per-invocation sandbox execution using ephemeral Docker containers
- Raspberry Pi camera capture tools (`rpicam-still` / `libcamera-still`) for Gemini vision tasks

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

2. Remove old/conflicting container packages:

```bash
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do sudo apt remove -y $pkg; done
```

3. Install Docker from the official Docker apt repo:

```bash
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin git
```

4. Enable Docker auto-start on reboot and add your user to docker group:

```bash
sudo systemctl enable --now docker containerd
sudo usermod -aG docker $USER
```

5. Re-login (or reboot) so Docker group permissions apply.

6. Clone project on the Pi:

```bash
git clone <your-repo-url> Aika
cd Aika
```

7. Create env file and set secrets:

```bash
cp .env.example .env
nano .env
```

8. Build images:

```bash
docker compose build aika
docker compose build sandbox-image
```

9. Start service:

```bash
docker compose up -d aika
```

10. Verify logs:

```bash
docker compose logs -f aika
```

11. Confirm auto-start behavior after reboot:

```bash
sudo reboot
# after reconnect:
systemctl is-active docker
docker compose ps
docker ps
```

`docker-compose.yml` uses `restart: unless-stopped`, so the bot container starts automatically when Docker starts after reboot.

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
- `CAMERA_STILL_COMMAND=rpicam-still`
- `CAMERA_CAPTURE_TIMEOUT_SECONDS=20`
- `CAMERA_CAPTURE_WARMUP_MS=1000`
- `GEMINI_MAX_CALLS_PER_MESSAGE=6`
- `GEMINI_MAX_TOOL_CALLS_PER_MESSAGE=10`
- `GEMINI_MAX_RETRIES=2`
- `GEMINI_RETRY_BASE_DELAY_SECONDS=1.0`
- `GEMINI_QUOTA_COOLDOWN_SECONDS=60`
- `GEMINI_DAILY_QUOTA_COOLDOWN_SECONDS=3600`

## Commands

- `/status`
- `/problems` (admin only, shows latest error reports)
- `/solve <prompt>` (admin only, uses GPT OSS 120B to propose shell actions)

`/solve` behavior:
- GPT OSS 120B generates a command plan.
- Before each command is run, bot sends clickable `Yes/No` approval buttons.
- Commands run from `/` inside the server container only after approval.
- Final message includes a complete execution summary.

## Gemini Camera Tools

Gemini can now call two camera tools:
- `camera_capture_for_ai`
  - captures a fresh image
  - attaches it back to Gemini in the same message flow for visual analysis
  - use for requests like "is someone in the kitchen?"
- `camera_capture_to_chat`
  - captures a fresh image
  - sends the image directly to Telegram chat
  - use for requests like "send me the current kitchen photo"

Both tools use one-shot camera commands (`rpicam-still`/`libcamera-still`) so the camera process exits after capture instead of staying active.

### Camera Prerequisites (Raspberry Pi)

- Container must be able to access camera devices and camera userspace stack.
- `rpicam-still` (or `libcamera-still`) must exist inside the running container.
- If camera tools fail with binary/device errors, run:

```bash
docker compose exec aika which rpicam-still
docker compose exec aika which libcamera-still
docker compose exec aika ls -l /dev/video* /dev/vchiq
```

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
- non-quota errors can trigger an auto-solve plan (still requires per-command Yes/No approval)

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

## Updating On Raspberry Pi

If code is already on Pi and container is running:

1. Pull latest code:

```bash
cd ~/Aika
git pull
```

2. Rebuild and restart bot container:

```bash
docker compose up -d --build aika
```

3. Confirm service is healthy:

```bash
docker compose ps
docker compose logs -f --tail=200 aika
```

4. If database migrations or env keys changed, update `.env` before step 2.

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
- If you get `per-message Gemini turn limit`:
  - Increase `GEMINI_MAX_CALLS_PER_MESSAGE` and/or `GEMINI_MAX_TOOL_CALLS_PER_MESSAGE`.
  - Split large tool-heavy requests into smaller steps.
- If `docker compose build` fails with `client version ... is too new. Maximum supported API version ...`:
  - This is a Docker client/daemon version mismatch.
  - Install Docker using the Raspberry Pi steps above (official Docker repo) so client + daemon are compatible.
  - Quick temporary workaround: `export DOCKER_API_VERSION=1.41` before running `docker compose`.

## Repository Layout

- `src/main.py` entrypoint
- `src/bot/handlers.py` central message router + command dispatch
- `src/agent/orchestrator.py` Gemini tool loop
- `src/tools/exec.py` sandbox executor
- `src/utils/paths.py` strict path isolation
- `src/memory/*` memory store + summarizer
- `src/jobs/*` scheduler + maintenance jobs
- `src/support/error_triage.py` incident reporting
- `src/support/solver.py` admin solve sessions and command approvals
- `sandbox/Dockerfile` sandbox runner image
