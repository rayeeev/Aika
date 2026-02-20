import asyncio
import logging
import logging.handlers
import os
import re
import signal
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F, types
from aiogram.types import Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from groq import Groq

from src.memory import COMPACTION_KEEP_DEFAULT, Memory

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SEND_STARTUP_MESSAGE = os.getenv("AIKA_STARTUP_MESSAGE", "true").lower() == "true"

# Parse multiple Gemini API keys
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_RAW.split(",") if key.strip()]

# Routing + latency configuration
PRECALL_TIMEOUT_SECONDS = 2.5
MAIN_TIMEOUT_SECONDS = 9.0
FALLBACK_TIMEOUT_SECONDS = 9.0
GENERATION_CAP_SECONDS = 24.0
DEBOUNCE_SECONDS = 0.8

COMPACTION_EVERY_TURNS = 5
MIN_MESSAGES_FOR_COMPACTION = 15
CC_KEEP_LAST_MESSAGES = COMPACTION_KEEP_DEFAULT

# Hardcoded Groq allowlist (prompt-requested model must be in this list)
GROQ_MODEL_ALLOWLIST = [
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "groq/compound",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905",
]
DEFAULT_GROQ_MODEL_FAST = "llama-3.1-8b-instant"
DEFAULT_GROQ_MODEL_DEEP = "qwen/qwen3-32b"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# --- Logging Setup ---
LOG_FILE = Path(os.path.dirname(os.path.dirname(__file__))) / "aika.log"

# Clear log on startup
if LOG_FILE.exists():
    LOG_FILE.write_text("")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=512 * 1024, backupCount=2
)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEY:
    logger.error("Missing TELEGRAM_BOT_TOKEN or GROQ_API_KEY")
    sys.exit(1)

if not GEMINI_API_KEYS:
    logger.warning("No GEMINI_API_KEYS configured. Gemini fallback will be unavailable.")
else:
    logger.info(f"Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")

try:
    ALLOWED_USER_ID = int(ALLOWED_USER_ID)
except (TypeError, ValueError):
    logger.error("ALLOWED_USER_ID must be set and be an integer")
    sys.exit(1)

TIMEZONE = ZoneInfo("America/Los_Angeles")

# Temp directory for audio files
AUDIO_TEMP_DIR = Path(tempfile.gettempdir()) / "aika_audio"
AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Bot, Dispatcher, Scheduler, and Memory
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
scheduler = AsyncIOScheduler()
memory = Memory()
groq_client = Groq(api_key=GROQ_API_KEY)
memory.set_groq_client(groq_client)

# Track scheduled wake-up jobs
scheduled_jobs: Dict[str, Dict[str, str]] = {}

# Track whether any side-effecting tool was called during current Gemini AFC cycle.
_tool_called_this_cycle = False

# --- Message Buffering & Turn State ---
message_buffer: deque[str] = deque()
processing_task: Optional[asyncio.Task] = None
input_lock = asyncio.Lock()
last_interaction_ts = 0.0

turn_counter = 0
turn_counter_lock = asyncio.Lock()
compaction_task: Optional[asyncio.Task] = None

# Initialize Gemini Clients for each API key
gemini_clients: List[genai.Client] = [genai.Client(api_key=key) for key in GEMINI_API_KEYS]

MODEL_LOOKUP = {model.lower(): model for model in GROQ_MODEL_ALLOWLIST}
CONTROL_TAG_RE = re.compile(r"\[(MODEL|PROVIDER|REASONING)\s*=\s*([^\]]+)\]", re.IGNORECASE)
TOOL_INTENT_RE = re.compile(
    r"\b(run|execute|command|shell|terminal|bash|ssh|read file|write file|edit file|"
    r"list directory|logs?|disk|cpu|memory usage|schedule|remind|wake up|delete memory|edit memory)\b",
    re.IGNORECASE,
)


@dataclass
class PromptControls:
    clean_text: str
    requested_provider: str = ""
    requested_model: str = ""
    reasoning: str = ""


def _strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _normalize_groq_model(name: str) -> str:
    if not name:
        return ""
    return MODEL_LOOKUP.get(name.strip().lower(), "")


def _mark_tool_called() -> None:
    global _tool_called_this_cycle
    _tool_called_this_cycle = True


async def _register_turn(event_name: str) -> None:
    """Increment global turn counter and schedule background compaction every N turns."""
    global turn_counter, compaction_task

    async with turn_counter_lock:
        turn_counter += 1
        current = turn_counter

    logger.info(f"Turn #{current} ({event_name})")

    if current % COMPACTION_EVERY_TURNS != 0:
        return

    if compaction_task and not compaction_task.done():
        logger.info("Background compaction already running; skipping duplicate schedule")
        return

    try:
        message_count = await memory.get_active_conversation_message_count()
    except Exception as e:
        logger.error(f"Failed to read active conversation message count: {e}")
        return

    if message_count < MIN_MESSAGES_FOR_COMPACTION:
        logger.info(
            "Skipping compaction trigger: active conversation has "
            f"{message_count} messages (< {MIN_MESSAGES_FOR_COMPACTION})"
        )
        return

    async def _run_compaction():
        try:
            compacted = await memory.compact_conversation_context_if_due(keep_last=CC_KEEP_LAST_MESSAGES)
            logger.info(f"Background compaction completed: compacted={compacted}")
        except Exception as e:
            logger.error(f"Background compaction failed: {e}", exc_info=True)

    compaction_task = asyncio.create_task(_run_compaction())


# --- Audio Transcription ---

async def transcribe_audio(file_path: Path) -> Optional[str]:
    """Transcribes audio using Groq Whisper."""

    def _transcribe() -> Optional[str]:
        try:
            with open(file_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="text",
                )
            if isinstance(transcription, str):
                return transcription.strip()
            if hasattr(transcription, "text"):
                return str(transcription.text).strip()
            logger.error(f"Unexpected transcription response type: {type(transcription)}")
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe)


async def download_voice_message(message: Message) -> Optional[Path]:
    """Downloads a voice message from Telegram and returns local file path."""
    try:
        voice = message.voice
        if not voice:
            return None

        file = await bot.get_file(voice.file_id)
        if not file.file_path:
            logger.error("Could not get file path from Telegram")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.file_path).suffix or ".ogg"
        if file_extension.lower() == ".oga":
            file_extension = ".ogg"

        local_path = AUDIO_TEMP_DIR / f"voice_{message.from_user.id}_{timestamp}{file_extension}"
        await bot.download_file(file.file_path, destination=local_path)
        logger.info(f"Downloaded voice message to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download voice message: {e}")
        return None


async def cleanup_audio_file(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning(f"Failed to cleanup audio file {file_path}: {e}")


# --- Sync Tool Functions for Gemini AFC ---

def execute_shell_command(cmd: str) -> str:
    """Executes a shell command on the host."""
    _mark_tool_called()
    import subprocess

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "(command completed with no output)"
        return f"Error (Exit Code {result.returncode}):\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds"
    except Exception as e:
        return f"Execution failed: {e}"


def read_file(path: str) -> str:
    """Reads a file from filesystem."""
    try:
        abs_path = os.path.abspath(path)
        with open(abs_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {e}"


def write_file(path: str, content: str) -> str:
    """Writes content to a file."""
    _mark_tool_called()
    try:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {abs_path}"
    except Exception as e:
        return f"Failed to write file: {e}"


def list_directory(path: str = ".") -> str:
    """Lists files in a directory."""
    try:
        abs_path = os.path.abspath(path)
        entries = os.listdir(abs_path)
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Failed to list directory: {e}"


def schedule_wake_up(
    seconds_from_now: int,
    thought: str,
    provider: str = "",
    model: str = "",
    reasoning: str = "",
) -> str:
    """Schedule a delayed self-initiated turn with optional inference overrides.

    Parameters:
    - seconds_from_now: delay before wake-up (1..604800)
    - thought: the exact instruction to run at wake-up time
    - provider: '' (auto), 'groq', or 'gemini'
    - model:
      - if provider='groq' or provider='' -> must be a Groq allowlist model
      - if provider='gemini' -> Gemini model name (optional)
    - reasoning: '' (auto), 'fast', or 'deep'

    When to use overrides:
    - Use provider/model when delayed work needs a specific engine/capability.
    - Leave empty for default routing.
    """
    _mark_tool_called()

    if seconds_from_now <= 0:
        return f"Error: seconds_from_now must be positive, got {seconds_from_now}"
    if seconds_from_now > 604800:
        return "Error: Cannot schedule more than 1 week ahead (604800 seconds)"

    provider_value = provider.strip().lower()
    if provider_value and provider_value not in {"groq", "gemini"}:
        return "Error: provider must be 'groq', 'gemini', or empty."

    reasoning_value = reasoning.strip().lower()
    if reasoning_value and reasoning_value not in {"fast", "deep"}:
        return "Error: reasoning must be 'fast', 'deep', or empty."

    model_value = model.strip()
    if model_value:
        if provider_value != "gemini":
            normalized_model = _normalize_groq_model(model_value)
            if not normalized_model:
                allowed = ", ".join(GROQ_MODEL_ALLOWLIST)
                return f"Error: model '{model_value}' is not allowed. Allowed Groq models: {allowed}"
            model_value = normalized_model

    run_date = datetime.now(TIMEZONE) + timedelta(seconds=seconds_from_now)
    job = scheduler.add_job(
        wake_up_callback,
        "date",
        run_date=run_date,
        args=[thought, provider_value, model_value, reasoning_value],
    )
    scheduled_jobs[job.id] = {
        "thought": thought,
        "provider": provider_value,
        "model": model_value,
        "reasoning": reasoning_value,
    }
    logger.info(
        f"Scheduled wake-up job {job.id} in {seconds_from_now}s: "
        f"thought={thought}, provider={provider_value or 'auto'}, "
        f"model={model_value or 'auto'}, reasoning={reasoning_value or 'auto'}"
    )
    override_text = (
        f"provider={provider_value or 'auto'}, model={model_value or 'auto'}, "
        f"reasoning={reasoning_value or 'auto'}"
    )
    return (
        f"Scheduled wake up in {seconds_from_now} seconds (job_id: {job.id}) "
        f"regarding '{thought}' with {override_text}."
    )


def read_server_logs(lines: int = 50) -> str:
    """Reads the last N lines of Aika's server log."""
    try:
        if not LOG_FILE.exists():
            return "(No log file found)"
        with open(LOG_FILE, "r") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Failed to read logs: {e}"


def recall_memory(query_type: str, date: str = "", time_range: str = "") -> str:
    return memory.recall_memory_sync(query_type, date, time_range)


def list_memories(memory_type: str) -> str:
    return memory.list_memories_sync(memory_type)


def delete_memory(memory_type: str, memory_id: int) -> str:
    _mark_tool_called()
    return memory.delete_memory_sync(memory_type, memory_id)


def edit_memory(memory_type: str, memory_id: int, new_content: str) -> str:
    _mark_tool_called()
    return memory.edit_memory_sync(memory_type, memory_id, new_content)


tools_list = [
    execute_shell_command,
    read_file,
    write_file,
    list_directory,
    schedule_wake_up,
    read_server_logs,
    recall_memory,
    list_memories,
    delete_memory,
    edit_memory,
]


# --- Prompt + Routing Helpers ---

def _parse_prompt_controls(user_text: str) -> PromptControls:
    requested_provider = ""
    requested_model = ""
    reasoning = ""

    for key, value in CONTROL_TAG_RE.findall(user_text):
        k = key.strip().lower()
        v = value.strip()
        if k == "provider":
            provider = v.lower()
            if provider in {"groq", "gemini"}:
                requested_provider = provider
        elif k == "model":
            requested_model = v
        elif k == "reasoning":
            mode = v.lower()
            if mode in {"fast", "deep"}:
                reasoning = mode

    clean_text = CONTROL_TAG_RE.sub("", user_text).strip()
    return PromptControls(
        clean_text=clean_text,
        requested_provider=requested_provider,
        requested_model=requested_model,
        reasoning=reasoning,
    )


def _looks_like_tool_request(text: str) -> bool:
    if not text:
        return False
    return bool(TOOL_INTENT_RE.search(text))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _build_identity_prompt(now_str: str, is_self_initiated: bool) -> str:
    source_note = "This is a self-initiated turn." if is_self_initiated else "This is a user-initiated turn."
    return (
        "You are Aika, an assistant running on a Raspberry Pi 5. "
        "Your owner is Erden Rayev (Telegram user id 935139103). "
        "Be concise, direct, and execution-focused. "
        "Never fabricate system actions. If uncertain, say so. "
        "If you are not sure what to do, output exactly [SILENCE].\n"
        "You play personality of Aika, mature woman with feelings. "
        "You can get mad and not answer if you are not happy with the situation. "
        "You can be sarcastic and joking without limits. "
        "Speak as a person, not as an AI. Don't say too much. Keep your responses short and to the point."
        "If you truly have nothing to add in a proactive/self turn, output exactly [SILENCE].\n"
        f"Current time: {now_str}\n"
        f"{source_note}"
    )


def _build_memory_block(prepared: Dict[str, Any], include_tools_note: bool) -> str:
    selected_semantic = prepared.get("selected_semantic", [])
    selected_episodic = prepared.get("selected_episodic", [])
    semantic_text = "\n".join(f"- {item}" for item in selected_semantic) or "(none)"
    episodic_text = "\n".join(f"- {item}" for item in selected_episodic) or "(none)"

    tools_note = ""
    if include_tools_note:
        tools_note = (
            "\nTool policy:\n"
            "- Use tools only for explicit system actions.\n"
            "- Max 2 tool calls per user request.\n"
            "- Use tool outputs in final response.\n"
        )

    return (
        f"Conversation context summary:\n{prepared.get('context_summary', '(none)')}\n\n"
        f"Today's earlier closed conversations:\n{prepared.get('today_summaries', '(none)')}\n\n"
        f"Global memory:\n{prepared.get('global_summary', '(none)')}\n\n"
        f"Selected semantic memories:\n{semantic_text}\n\n"
        f"Selected episodic memories:\n{episodic_text}"
        f"{tools_note}"
    )


def format_time_gap(seconds: float) -> str:
    if seconds < 3600:
        minutes = max(1, int(seconds / 60))
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    if seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = int(seconds / 86400)
    return f"{days} day{'s' if days != 1 else ''}"


def _select_provider_and_model(
    controls: PromptControls,
    prepared: Dict[str, Any],
) -> tuple[str, str, str]:
    route = prepared.get("route", {}) if isinstance(prepared.get("route"), dict) else {}
    route_requires_tools = _coerce_bool(route.get("requires_tools", False))
    heuristic_requires_tools = _looks_like_tool_request(controls.clean_text)
    requires_tools = route_requires_tools or heuristic_requires_tools

    provider_hint = str(route.get("provider_hint", "auto")).lower().strip()
    requested_provider = controls.requested_provider

    if requested_provider in {"groq", "gemini"}:
        provider = requested_provider
    elif requires_tools:
        provider = "gemini"
    elif provider_hint in {"groq", "gemini"}:
        provider = provider_hint
    else:
        provider = "groq"

    if provider == "gemini" and not gemini_clients:
        provider = "groq"

    complexity = controls.reasoning or str(route.get("complexity", "fast")).lower().strip()
    if complexity not in {"fast", "deep"}:
        complexity = "fast"

    requested_model = _normalize_groq_model(controls.requested_model)
    route_model_hint = _normalize_groq_model(str(route.get("model_hint", "")))

    if complexity == "deep":
        default_model = _normalize_groq_model(DEFAULT_GROQ_MODEL_DEEP) or GROQ_MODEL_ALLOWLIST[0]
    else:
        default_model = _normalize_groq_model(DEFAULT_GROQ_MODEL_FAST) or GROQ_MODEL_ALLOWLIST[0]

    groq_model = requested_model or route_model_hint or default_model

    if controls.requested_model and not requested_model:
        logger.warning(f"Requested Groq model '{controls.requested_model}' is not in allowlist")

    return provider, groq_model, complexity


async def _call_groq_main_response(
    user_text: str,
    prepared: Dict[str, Any],
    model: str,
    reasoning: str,
    is_self_initiated: bool,
) -> str:
    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    system_prompt = (
        _build_identity_prompt(now_str, is_self_initiated)
        + "\n\n"
        + _build_memory_block(prepared, include_tools_note=False)
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    recent_messages = prepared.get("recent_messages", [])
    if isinstance(recent_messages, list):
        for item in recent_messages[-12:]:
            role = "assistant" if item.get("role") == "model" else "user"
            content = str(item.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content})

    final_user_text = user_text.strip() or "(No content provided after control tags.)"
    messages.append({"role": "user", "content": final_user_text})

    temperature = 0.7 if reasoning == "deep" else 0.45

    def _sync_call() -> str:
        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        text = _strip_thinking_blocks(text).strip()
        if not text:
            raise RuntimeError("Groq returned empty response")
        return text

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_call)


async def _call_gemini_response(
    user_text: str,
    prepared: Dict[str, Any],
    is_self_initiated: bool,
    groq_error_context: str = "",
) -> str:
    if not gemini_clients:
        raise RuntimeError("Gemini fallback unavailable: GEMINI_API_KEYS is empty")

    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    error_section = (
        f"\nPrevious Groq error (for context): {groq_error_context}\n"
        if groq_error_context
        else ""
    )
    wake_up_tool_guide = (
        "\n\nschedule_wake_up field guide:\n"
        "- seconds_from_now: integer delay in seconds (1..604800)\n"
        "- thought: instruction to execute at wake-up\n"
        "- provider: '' (auto), 'groq', or 'gemini'\n"
        "- model:\n"
        "  - with provider='groq' or '' -> choose a Groq allowlist model (e.g. 'groq/compound')\n"
        "  - with provider='gemini' -> optional Gemini model name\n"
        "- reasoning: '' (auto), 'fast', or 'deep'\n"
        "Usage rules:\n"
        "- Use provider/model/reasoning only when delayed task needs a specific AI.\n"
        "- For normal reminders, keep provider/model/reasoning empty.\n"
    )
    system_instruction = (
        _build_identity_prompt(now_str, is_self_initiated)
        + "\n\n"
        + _build_memory_block(prepared, include_tools_note=True)
        + "\n\nAvailable tools: execute_shell_command, read_file, write_file, list_directory, "
        "schedule_wake_up, read_server_logs, recall_memory, list_memories, delete_memory, edit_memory."
        + wake_up_tool_guide
        + error_section
    )

    history: List[genai_types.Content] = []
    cc_summary = prepared.get("cc_summary", "")
    if cc_summary:
        history.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=f"[CONVERSATION CONTEXT SUMMARY]: {cc_summary}")],
            )
        )
        history.append(
            genai_types.Content(
                role="model",
                parts=[genai_types.Part(text="[Acknowledged]")],
            )
        )

    recent_messages = prepared.get("recent_messages", [])
    if isinstance(recent_messages, list):
        for i, msg in enumerate(recent_messages):
            if i > 0 and "timestamp" in recent_messages[i - 1] and "timestamp" in msg:
                gap_seconds = float(msg["timestamp"]) - float(recent_messages[i - 1]["timestamp"])
                if gap_seconds > 1800:
                    history.append(
                        genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text=f"[TIME GAP: {format_time_gap(gap_seconds)} passed]")],
                        )
                    )
                    history.append(
                        genai_types.Content(
                            role="model",
                            parts=[genai_types.Part(text="[Acknowledged]")],
                        )
                    )

            role = "user" if msg.get("role") == "user" else "model"
            content = str(msg.get("content", "")).strip()
            if content:
                history.append(genai_types.Content(role=role, parts=[genai_types.Part(text=content)]))

    final_user_text = user_text.strip() or "(No content provided after control tags.)"

    global _tool_called_this_cycle
    _tool_called_this_cycle = False

    loop = asyncio.get_running_loop()
    last_error: Optional[Exception] = None

    for i, gemini_client in enumerate(gemini_clients):
        if asyncio.current_task() and asyncio.current_task().cancelled():
            raise asyncio.CancelledError()

        try:
            chat = gemini_client.chats.create(
                model=DEFAULT_GEMINI_MODEL,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=tools_list,
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                        disable=False,
                        maximum_remote_calls=3,
                    ),
                    temperature=0.7,
                ),
                history=history,
            )

            response = await loop.run_in_executor(None, lambda: chat.send_message(final_user_text))
            text = response.text
            if text is None:
                logger.warning("Gemini AFC exhausted. Asking Gemini to summarize current findings.")
                followup = await loop.run_in_executor(
                    None,
                    lambda: chat.send_message(
                        "You hit your tool call limit. Summarize what you found and ask if user wants continuation."
                    ),
                )
                return followup.text or "I gathered partial results. Want me to continue?"
            return text
        except Exception as e:
            last_error = e
            logger.warning(f"Gemini API key {i + 1}/{len(gemini_clients)} failed: {e}")
            if _tool_called_this_cycle:
                logger.error("Tool calls already executed; refusing retry across API keys to avoid duplicate side effects")
                return (
                    "I hit an API error after already executing some actions. "
                    "Please verify the side effects and ask me to continue if needed."
                )
            continue

    logger.error(f"All Gemini API keys exhausted. Last error: {last_error}")
    return "Gemini fallback is unavailable right now."


# --- Core Logic ---

async def _store_messages(user_text: str, model_text: str) -> None:
    try:
        await memory.add_message("user", user_text)
        await memory.add_message("model", model_text)
    except Exception as e:
        logger.error(f"Failed to store messages in memory: {e}")


async def generate_response(
    raw_user_text: str,
    is_self_initiated: bool = False,
    event_type: str = "user",
) -> str:
    """Groq-first inference with one Groq memory pre-call and Gemini fallback."""
    controls = _parse_prompt_controls(raw_user_text)
    model_input_text = controls.clean_text or raw_user_text.strip()

    try:
        prepared = await asyncio.wait_for(
            memory.prepare_inference_context(
                user_text=model_input_text,
                event_type=event_type,
                requested_provider=controls.requested_provider,
                requested_model=controls.requested_model,
                allowed_models=GROQ_MODEL_ALLOWLIST,
            ),
            timeout=PRECALL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Groq pre-call timed out. Falling back to local lightweight context")
        prepared = await memory.get_lightweight_inference_context(
            user_text=model_input_text,
            event_type=event_type,
        )
    except Exception as e:
        logger.error(f"Groq pre-call failed: {e}", exc_info=True)
        prepared = await memory.get_lightweight_inference_context(
            user_text=model_input_text,
            event_type=event_type,
        )

    provider, groq_model, reasoning = _select_provider_and_model(controls, prepared)

    if provider == "groq":
        try:
            return await asyncio.wait_for(
                _call_groq_main_response(
                    user_text=model_input_text,
                    prepared=prepared,
                    model=groq_model,
                    reasoning=reasoning,
                    is_self_initiated=is_self_initiated,
                ),
                timeout=MAIN_TIMEOUT_SECONDS,
            )
        except Exception as groq_error:
            logger.warning(f"Groq main call failed ({groq_model}): {groq_error}")
            if not gemini_clients:
                return "I hit an upstream model error and Gemini fallback is not configured."
            try:
                return await asyncio.wait_for(
                    _call_gemini_response(
                        user_text=model_input_text,
                        prepared=prepared,
                        is_self_initiated=is_self_initiated,
                        groq_error_context=str(groq_error),
                    ),
                    timeout=FALLBACK_TIMEOUT_SECONDS,
                )
            except Exception as fallback_error:
                logger.error(f"Gemini fallback failed after Groq error: {fallback_error}", exc_info=True)
                return "I ran into a model routing error. Please try again in a moment."

    # Provider forced/routed to Gemini first
    try:
        return await asyncio.wait_for(
            _call_gemini_response(
                user_text=model_input_text,
                prepared=prepared,
                is_self_initiated=is_self_initiated,
            ),
            timeout=MAIN_TIMEOUT_SECONDS,
        )
    except Exception as gemini_error:
        logger.warning(f"Gemini primary path failed: {gemini_error}")

    try:
        return await asyncio.wait_for(
            _call_groq_main_response(
                user_text=model_input_text,
                prepared=prepared,
                model=groq_model,
                reasoning=reasoning,
                is_self_initiated=is_self_initiated,
            ),
            timeout=FALLBACK_TIMEOUT_SECONDS,
        )
    except Exception as groq_error:
        logger.error(f"Groq fallback after Gemini failure also failed: {groq_error}", exc_info=True)
        return "I couldn't reach either model provider right now."


async def wake_up_callback(
    thought: str,
    provider: str = "",
    model: str = "",
    reasoning: str = "",
) -> None:
    logger.info(
        "Waking up with thought: "
        f"{thought} | provider={provider or 'auto'} model={model or 'auto'} reasoning={reasoning or 'auto'}"
    )
    await _register_turn("wake_up_callback")

    controls: List[str] = []
    if provider:
        controls.append(f"[PROVIDER={provider}]")
    if model:
        controls.append(f"[MODEL={model}]")
    if reasoning:
        controls.append(f"[REASONING={reasoning}]")
    controls_prefix = " ".join(controls).strip()
    wake_prompt = f"{controls_prefix} [SELF-INITIATED] {thought}".strip()

    response_text = await generate_response(
        wake_prompt,
        is_self_initiated=True,
        event_type="wake_up",
    )

    if response_text and response_text.strip() != "[SILENCE]":
        try:
            await bot.send_message(chat_id=ALLOWED_USER_ID, text=response_text)
            await _register_turn("assistant_reply_wake_up")
        except Exception as e:
            logger.error(f"Failed to send wake-up message: {e}")
        asyncio.create_task(_store_messages(wake_prompt, response_text))
    else:
        logger.info("Aika chose silence for self-initiated thought.")
        asyncio.create_task(_store_messages(wake_prompt, "[CHOSE SILENCE]"))


async def check_if_should_speak() -> None:
    """Proactive layer. Uses a cheap Groq call to decide whether to send follow-up."""
    global last_interaction_ts

    if (datetime.now().timestamp() - last_interaction_ts) < 5:
        return

    cc = await memory.get_conversation_context(limit_messages=6)
    recent_msgs = cc["messages"][-6:]
    if not recent_msgs:
        return

    history_text = "\n".join(f"[{m['role']}]: {m['content']}" for m in recent_msgs)
    prompt = (
        f"Recent conversation:\n\n{history_text}\n\n"
        "Decide whether Aika should proactively send one new message. "
        "If no proactive message is needed, output exactly [NO]. "
        "If yes, output only a short thought/reason for proactive follow-up."
        "For example, if Aika asked question and user didn't answer, output proactive thought to get frustrated."
        "If user still doesn't answer, output proactive thought to get mad at user and not answer him later."
    )

    try:
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.4,
            ),
        )
        result = (completion.choices[0].message.content or "").strip()
        result = _strip_thinking_blocks(result)
        if not result or "[NO]" in result:
            return

        response_text = await generate_response(
            f"[PROACTIVE THOUGHT]: {result}",
            is_self_initiated=True,
            event_type="proactive",
        )

        if response_text and response_text.strip() != "[SILENCE]":
            await bot.send_message(chat_id=ALLOWED_USER_ID, text=response_text)
            await _register_turn("assistant_reply_proactive")
            asyncio.create_task(_store_messages(f"[PROACTIVE: {result}]", response_text))
    except Exception as e:
        logger.error(f"Proactive check failed: {e}")


async def _keep_typing(chat_id: int) -> None:
    """Send typing indicator every few seconds."""
    try:
        while True:
            try:
                await bot.send_chat_action(chat_id=chat_id, action="typing")
            except Exception:
                pass
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        return


async def _process_buffered_messages(chat_id: int) -> None:
    global processing_task

    try:
        await asyncio.sleep(DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return

    async with input_lock:
        if not message_buffer:
            return
        combined_text = "\n\n".join(message_buffer)
        message_buffer.clear()

    logger.info(f"Processing combined input ({len(combined_text)} chars)")
    await process_combined_input(chat_id, combined_text)


async def process_combined_input(chat_id: int, user_text: str) -> None:
    global last_interaction_ts

    typing_task = asyncio.create_task(_keep_typing(chat_id))

    response_text: Optional[str] = None
    try:
        response_text = await asyncio.wait_for(
            generate_response(user_text, event_type="user"),
            timeout=GENERATION_CAP_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(f"Response generation timed out after {GENERATION_CAP_SECONDS}s")
        response_text = "I timed out while thinking. Try again in a moment."
    except asyncio.CancelledError:
        logger.info("Generation cancelled due to newer input")
        typing_task.cancel()
        return
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    if not response_text or response_text.strip() == "[SILENCE]":
        logger.info("Aika chose silence")
        asyncio.create_task(_store_messages(user_text, "[CHOSE SILENCE]"))
        return

    try:
        await bot.send_message(chat_id=chat_id, text=response_text)
        last_interaction_ts = datetime.now().timestamp()
        await _register_turn("assistant_reply")
        asyncio.create_task(_store_messages(user_text, response_text))
        asyncio.create_task(check_if_should_speak())
    except Exception as e:
        logger.error(f"Failed to send response: {e}")


async def handle_new_input(message: Message, text: str) -> None:
    """Main entry point for user textual input with debounce and cancellation semantics."""
    global processing_task

    await _register_turn("user_input")

    async with input_lock:
        if processing_task and not processing_task.done():
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
            logger.info("Cancelled previous processing task to accumulate new input")

        message_buffer.append(text)
        processing_task = asyncio.create_task(_process_buffered_messages(message.chat.id))


@dp.message(F.voice)
async def handle_voice_message(message: Message) -> None:
    if message.from_user.id != ALLOWED_USER_ID:
        return

    logger.info(f"Received voice message from user {message.from_user.id}")
    processing_msg = await message.reply("🎙️ Processing voice message...")

    audio_path: Optional[Path] = None
    try:
        audio_path = await download_voice_message(message)
        if not audio_path:
            await processing_msg.edit_text("❌ Failed to download voice message.")
            return

        transcription = await transcribe_audio(audio_path)
        if not transcription:
            await processing_msg.edit_text("❌ Failed to transcribe voice message.")
            return

        logger.info(f"Transcribed voice message: {transcription}")
        await processing_msg.delete()
        await handle_new_input(message, f"[VOICE] {transcription}")
    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        try:
            await processing_msg.edit_text(f"❌ Error processing voice message: {e}")
        except Exception:
            pass
    finally:
        if audio_path:
            await cleanup_audio_file(audio_path)


@dp.message(F.text)
async def handle_text_message(message: Message) -> None:
    if message.from_user.id != ALLOWED_USER_ID:
        return

    user_text = message.text
    if not user_text:
        return

    logger.info(f"Received: {user_text}")
    await handle_new_input(message, user_text)


@dp.message_reaction()
async def handle_reaction(message_reaction: types.MessageReactionUpdated) -> None:
    if message_reaction.user.id != ALLOWED_USER_ID:
        return

    new_reactions = message_reaction.new_reaction
    if not new_reactions:
        return

    reaction = new_reactions[-1]
    emoji = reaction.emoji if hasattr(reaction, "emoji") else "content"
    logger.info(f"Received reaction: {emoji}")

    content = f"[USER REACTION: {emoji}]"
    try:
        await memory.add_message("user", content)
    except Exception as e:
        logger.error(f"Failed to save reaction: {e}")

    await _register_turn("reaction")
    asyncio.create_task(check_if_should_speak())


@dp.message()
async def handle_unsupported_message(message: Message) -> None:
    if message.from_user.id != ALLOWED_USER_ID:
        return
    logger.info(f"Received unsupported message type from user {message.from_user.id}")


async def shutdown(signal_type) -> None:
    logger.info(f"Received signal {signal_type.name}, shutting down gracefully...")

    scheduler.shutdown(wait=False)
    await dp.stop_polling()
    await bot.session.close()

    try:
        for file in AUDIO_TEMP_DIR.glob("*"):
            file.unlink()
        logger.info("Cleaned up temporary audio files.")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp directory: {e}")

    logger.info("Shutdown complete.")


async def main() -> None:
    await memory.init_db()

    scheduler.add_job(memory.close_stale_conversations, "interval", minutes=5)
    scheduler.add_job(memory.run_daily_summary, "cron", hour=4, minute=0)
    scheduler.add_job(memory.run_global_update, "cron", hour=4, minute=10)
    scheduler.add_job(memory.run_weekly_cleanup, "cron", day_of_week="sun", hour=4, minute=20)
    scheduler.start()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    logger.info("Aika (Inference V4) started")
    logger.info(f"Audio temp directory: {AUDIO_TEMP_DIR}")
    logger.info(f"Groq allowlist: {', '.join(GROQ_MODEL_ALLOWLIST)}")

    if SEND_STARTUP_MESSAGE:
        try:
            await bot.send_message(ALLOWED_USER_ID, "I'm awake. Don't break anything.")
        except Exception as e:
            logger.warning(f"Failed to send startup message: {e}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
