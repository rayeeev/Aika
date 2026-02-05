You are Codex. Implement a complete, production-lean Telegram agent server for a Raspberry Pi.

GOAL
Build a dockerized Python service that:
- Runs as a single Docker container (the server itself is sandboxed in Docker).
- Uses Telegram long polling (no webhook/HTTPS server required).
- Default behavior: route user messages to Google Gemini using the newest Google GenAI Python SDK.
- Supports commands with separate logic paths.
- Manages per-user files/images safely with strict folder isolation.
- Executes shell commands ONLY inside a per-invocation sandbox container that mounts ONLY that user’s folder.
- Runs background maintenance + summarization jobs heavily to minimize Gemini token usage.
- Uses Groq for summarization and background technical support.

TECH STACK (must use)
- Python 3.13
- Telegram: aiogram v3 (async) long polling (if it's needed)
- Gemini: google-genai SDK (Gemini Developer API) using model "gemini-3-flash-preview"
- Groq: official groq Python SDK using:
  - model "qwen/qwen3-32b" for all summarizations
  - model "openai/gpt-oss-120b" for background technical support on exceptions/errors
- Storage: SQLite (SQLAlchemy) for users, messages, summaries, reminders, file metadata, job logs
- Scheduler: APScheduler AsyncIO for reminders + background maintenance jobs
- Container orchestration: docker-compose for running the server
- Sandbox execution: server container is allowed to spawn ephemeral sandbox containers using /var/run/docker.sock mount (explicitly document the security tradeoff). Use python docker SDK (preferred) or docker CLI as fallback.

REPO OUTPUT (must generate all files)
1) docker-compose.yml
2) Dockerfile
3) requirements.txt (or pyproject.toml + lock; but keep simple)
4) .env.example
5) README.md (setup + run + security notes)
6) src/ package with clear modules:
   - src/main.py (entrypoint)
   - src/config.py
   - src/bot/handlers.py (Telegram routing + commands)
   - src/bot/middleware.py (user filtering + role)
   - src/llm/gemini_client.py
   - src/llm/groq_client.py
   - src/agent/orchestrator.py (planning + tool dispatch)
   - src/tools/files.py
   - src/tools/exec.py
   - src/tools/reminders.py
   - src/memory/summarizer.py
   - src/memory/store.py
   - src/db/models.py
   - src/db/session.py
   - src/jobs/scheduler.py
   - src/jobs/maintenance.py
   - src/support/error_triage.py
   - src/utils/paths.py (critical: path isolation)
   - src/utils/logging.py
7) Provide basic unit tests for path isolation and sandbox exec command building (optional but preferred).

CONFIG (.env)
- TELEGRAM_BOT_TOKEN=
- ALLOWED_USER_IDS=comma-separated numeric Telegram user ids
- ADMIN_USER_IDS=comma-separated numeric ids (subset)
- GEMINI_API_KEY=
- GROQ_API_KEY=
- DATA_DIR=/data
- USER_DIR=/data/users
- SHARED_DIR=/data/shared
- CACHE_TTL_HOURS=72
- SANDBOX_IMAGE=agent-sandbox-runner:latest
- SANDBOX_TIMEOUT_SECONDS=30
- SANDBOX_MEMORY=512m
- SANDBOX_CPUS=0.5
- LOG_LEVEL=INFO
- TIMEZONE=America/Los_Angeles

FILESYSTEM LAYOUT inside container
- /data/users/<telegram_user_id>/
   - files/        (long-lived user files)
   - cache/        (ephemeral attachments)
   - logs/
- /data/shared/    (shared folder)
- /data/admin/
   - error_reports/
   - job_logs/

TELEGRAM MESSAGE LOGIC (core requirement)
Implement a central “message_router” (single file / module) that:
1) Filters users:
   - Only allow messages from ALLOWED_USER_IDS.
   - If not allowed: don't spend compute on it. Don't reply.
2) Assigns handling:
   - If message is a command (starts with "/"): route to command handlers (no Gemini by default).
   - Else default: AI flow using Gemini (Gemini can use tools through function calling).
3) File/image handling:
   - If user sends file/photo WITHOUT caption:
       - Download into /data/users/<id>/cache/<timestamp>_<originalname>
       - Store metadata in DB as “pending_attachment” for that user
       - IMPORTANT: on the user’s NEXT text message, automatically inject a short note into the prompt like:
         “Pending attachments: [local_path1, local_path2]”
         Then clear pending attachments after they’re used once.
   - If user sends file/photo WITH caption (or text attached):
       - Download file to cache
       - Immediately send to Gemini with the caption as the user prompt, plus the local path reference.
       - If Gemini supports multimodal input (image bytes), pass the image bytes too; otherwise include path text.
4) Maintain a per-user conversation state with heavy summarization (see MEMORY below).

COMMANDS (must implement)
- /help : show short help
- /status : show server status + user folder disk usage
- /run <shell command> :
    - Only admins by default
    - Execute command inside whole container
    - Return stdout/stderr (truncate to safe size) + exit code
- /files : list recent files in user files/ and cache/
- /save <relative_cache_path> <dest_rel_path> : move from cache to files (within user dir)
- /shared put <rel_user_path> <rel_shared_path> :
    - Requires explicit command; copy from user folder to shared
- /shared get <rel_shared_path> <rel_user_path> :
    - Copy from shared to user folder
- /remind <when> <text> :
    - Parse natural-ish times minimally (support “in 10m”, “in 2h”, “tomorrow 9am”, ISO)
    - Store reminder in DB and schedule send
- /remind_user <user_id> <when> <text> :
    - Parse natural-ish times minimally (support “in 10m”, “in 2h”, “tomorrow 9am”, ISO)
    - Store reminder in DB and schedule send

TOOLS (available to Gemini via function calling)
Expose tools with strict permissions & path isolation:
1) file_write(path_rel, content, location="user"|"shared")
2) file_read(path_rel, location="user"|"shared")
3) file_list(path_rel_dir="", location="user"|"shared")
4) file_move(src_rel, dst_rel, location="user")  (no cross-user)
5) run_shell(cmd, timeout_seconds=..., workdir_rel="")  -> ALWAYS runs in sandbox container mounting only that user folder.
6) set_reminder(when, text, target_user_id=optional)
Rules:
- By default, tools operate in the user’s folder only.
- Shared folder operations ONLY when explicitly requested by user text or by /shared command; if Gemini suggests shared access, require confirmation step (return a “needs_confirmation” tool result and ask user).
- Absolute paths are forbidden. Any path traversal must be blocked.

PATH ISOLATION (non-negotiable)
Implement a helper that resolves requested paths and enforces:
- Must be within /data/users/<id>/... for user scope
- Must be within /data/shared/... for shared scope
- Must never allow “..” escape after resolve()
- Must never allow access to other users’ folders
Write tests for this.

SANDBOX EXECUTION (non-negotiable)
Do NOT execute user commands directly in the server container.
Instead, for each run_shell tool call:
- Start ephemeral container using SANDBOX_IMAGE
- Mount /data/users/<id> as /work:rw
- Set workdir to /work/(workdir_rel)
- Disable network: network_mode="none"
- Drop privileges where possible, set no-new-privileges
- Set resource limits: memory, cpu, pids limit
- Enforce timeout; kill container on timeout
- Capture stdout/stderr with size caps; then remove container
Provide the Dockerfile for SANDBOX_IMAGE too (multi-stage or separate service):
- Minimal tools: bash/sh, coreutils, python3, node optional
- Non-root user inside runner image

LLM ORCHESTRATION
Default chat flow:
- Build prompt context efficiently:
  - Include current “major memory” (<= 4 sentences)
  - Include last N recent turns (recommend 4 turns)
  - Include pending attachments paths (if any)
  - Include today summary (<= 2 sentences) if exists
- Call Gemini model "gemini-3-flash-preview"
- Use function calling for tools: Gemini can request tools; orchestrator validates and executes, then feeds tool results back to Gemini for final answer.

GROQ USAGE
- Summarization (ALL summarizations): use Groq model "qwen/qwen3-32b" only.
- Error technical support: use Groq model "openai/gpt-oss-120b" only, and ONLY when exceptions occur.

MEMORY + SUMMARIZATION REQUIREMENTS (heavy optimization)
Implement background and incremental summarization to reduce Gemini token usage:

A) Rolling “short memory” compression:
- Maintain full message log in DB.
- Also maintain a “compressed_memory” list of short sentences.
- Rule: keep only the last 3 user-assistant exchanges verbatim in the Gemini prompt.
- Every time a new assistant reply is stored:
   - If there are more than 3 exchanges older than the recent window, take the oldest remaining exchange that is still uncompressed and summarize that single exchange (user+assistant) into ONE short sentence using Groq Qwen3-32B.
   - Store that sentence into compressed_memory and mark that exchange as compressed.
This approximates: “the fourth prompt/answer from the end becomes one short sentence.”

B) Daily summarization (once a day per user):
- At 04:00 local time, for each user:
   - Summarize that day’s conversation into exactly TWO sentences using Groq Qwen3-32B.
   - Store as daily_summary(date, text).

C) Weekly major memory summarization (once a week per user):
- Every Sunday 05:00 local time:
   - Take the last 7 daily summaries + previous major_memory (if any)
   - Summarize into exactly FOUR sentences using Groq Qwen3-32B.
   - Store as major_memory (overwrite previous major_memory).

D) Cache cleanup (background):
- Every hour: delete files in /cache older than CACHE_TTL_HOURS, and remove DB references.

BACKGROUND TECHNICAL SUPPORT (errors)
- Add a global exception handler around update processing and scheduled jobs.
- On exception:
   - Save stack trace + context into /data/admin/error_reports/<timestamp>.md
   - Call Groq GPT OSS 120B with:
       - stack trace
       - relevant module code snippets (read-only: the files touched, limited to top N lines)
       - ask for a short root cause + patch suggestions
   - Append response to the error report file
   - Notify ADMIN_USER_IDS in Telegram with a short “Error occurred; report saved at …” (do NOT spam; rate limit notifications).

SECURITY & SAFETY BEHAVIOR
- Enforce per-user isolation strictly.
- All destructive operations should require explicit command or explicit user confirmation.
- Truncate large outputs.
- Never log secrets.
- Provide clear notes in README about docker.sock risk and how to mitigate (e.g., run on dedicated Pi, restrict admins).

README MUST INCLUDE
- Setup steps (create bot, set env vars)
- docker-compose up
- How to add allowed users
- How sandbox runner works
- Security warnings
- Example commands and example interactions

IMPLEMENTATION DETAILS (must do)
- Use structured logging.
- Use asyncio end-to-end.
- Design handlers so Telegram stays responsive (send “typing…” and progress for long tasks).
- Ensure all API clients are initialized once and reused.
- Provide retry/backoff for Gemini/Groq transient errors.
- Provide a minimal “health summary” for /status.

DELIVERABLE
Output the entire repository contents (all files) with correct code, ready to run on a Raspberry Pi using docker-compose.
Do NOT omit code. Do NOT leave TODOs for core functionality.
