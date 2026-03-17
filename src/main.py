import asyncio
import json
import logging
import logging.handlers
import os
import re
import shlex
import signal
import sys
import tempfile
import uuid
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F, types
from aiogram.types import Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from google import genai

from groq import Groq

from src.memory import COMPACTION_KEEP_DEFAULT, Memory

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")
ALLOWED_USER_ID_ENV = os.getenv("ALLOWED_USER_ID")
GROQ_API_KEYS_RAW = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
SEND_STARTUP_MESSAGE = os.getenv("AIKA_STARTUP_MESSAGE", "true").lower() == "true"

# Parse API keys
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_RAW.split(",") if key.strip()]
GROQ_API_KEYS = [key.strip() for key in GROQ_API_KEYS_RAW.split(",") if key.strip()]

# Routing + latency configuration
PRECALL_TIMEOUT_SECONDS = 4.0

FALLBACK_TIMEOUT_SECONDS = 9.0
GENERATION_CAP_SECONDS = 24.0
DEBOUNCE_SECONDS = 0.8
DEFAULT_TEMPERATURE = 0.7

COMPACTION_EVERY_TURNS = 5
MIN_MESSAGES_FOR_COMPACTION = 15
CC_KEEP_LAST_MESSAGES = COMPACTION_KEEP_DEFAULT

# Models fallback chain
TARGET_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct-0905",
]


# --- Logging Setup ---
LOG_FILE = Path(os.path.dirname(os.path.dirname(__file__))) / "aika.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=512 * 1024, backupCount=2
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEYS:
    logger.error("Missing TELEGRAM_BOT_TOKEN or GROQ_API_KEYS")
    sys.exit(1)

if not GEMINI_API_KEYS:
    logger.warning(
        "No GEMINI_API_KEYS configured. Gemini fallback will be unavailable."
    )
else:
    logger.info(f"Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")

try:
    ALLOWED_USER_ID = int(ALLOWED_USER_ID_ENV) if ALLOWED_USER_ID_ENV else 0
    if not ALLOWED_USER_ID:
        raise ValueError
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
groq_clients = [Groq(api_key=key) for key in GROQ_API_KEYS]
memory.set_groq_client(groq_clients[0] if groq_clients else None)

# Track scheduled wake-up jobs
scheduled_jobs: Dict[str, Dict[str, str]] = {}

# Store loop state for sync wrappers
main_loop: Optional[asyncio.AbstractEventLoop] = None


# --- Message Buffering & Turn State ---
message_buffer: deque[str] = deque()
processing_task: Optional[asyncio.Task] = None
input_lock = asyncio.Lock()
last_interaction_ts = 0.0

turn_counter = 0
turn_counter_lock = asyncio.Lock()
compaction_task: Optional[asyncio.Task] = None

# Proactive follow-up state
PROACTIVE_DELAY_SECONDS = 20.0
PROACTIVE_MAX_OUTPUTS = 2
proactive_state_lock = asyncio.Lock()
proactive_task: Optional[asyncio.Task] = None
proactive_chain_id = 0
user_activity_counter = 0

# Initialize Gemini Clients for each API key
gemini_clients: List[genai.Client] = [
    genai.Client(api_key=key) for key in GEMINI_API_KEYS
]


@dataclass
class AgentSandbox:
    id: str
    cli: str  # "claude" or "gemini"
    initial_prompt: str
    state: str  # "running" or "finished"
    created_at: float
    finished_at: float = -1.0
    task: Optional[asyncio.Task] = None
    process: Optional[asyncio.subprocess.Process] = None
    output: str = ""


active_sandboxes: Dict[str, AgentSandbox] = {}


CONTROL_TAG_RE = re.compile(r"\[(REASONING)\s*=\s*([^\]]+)\]", re.IGNORECASE)


@dataclass
class PromptControls:
    clean_text: str
    reasoning: str = ""


def _strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text




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
        logger.info(
            "Background compaction already running; skipping duplicate schedule"
        )
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
            compacted = await memory.compact_conversation_context_if_due(
                keep_last=CC_KEEP_LAST_MESSAGES
            )
            logger.info(f"Background compaction completed: compacted={compacted}")
        except Exception as e:
            logger.error(f"Background compaction failed: {e}", exc_info=True)

    compaction_task = asyncio.create_task(_run_compaction())


async def _cancel_proactive_task_locked(reason: str) -> None:
    """Cancel pending proactive task. Caller must hold proactive_state_lock."""
    global proactive_task
    if proactive_task and not proactive_task.done():
        proactive_task.cancel()
        logger.info(f"Cancelled proactive task: {reason}")
    proactive_task = None


async def _record_user_activity(reason: str) -> None:
    """Track user activity and cancel pending proactive follow-up timers."""
    global user_activity_counter
    async with proactive_state_lock:
        user_activity_counter += 1
        await _cancel_proactive_task_locked(f"user activity ({reason})")


async def _proactive_chain_is_still_valid(
    chain_id: int, snapshot_user_counter: int
) -> bool:
    async with proactive_state_lock:
        return (
            chain_id == proactive_chain_id
            and snapshot_user_counter == user_activity_counter
        )


async def _should_proactive_continue(stage: int) -> bool:
    """Decide whether Aika should proactively continue the conversation.

    This is a strict gate — it returns True only when silence would be clearly wrong:
    - Aika asked the user a direct question and got no text reply (only a reaction)
    - Aika explicitly promised to follow up or check something
    - A critical piece of information was requested but never provided

    In ALL other cases it returns False. The default is silence.
    """
    if not memory.groq_client:
        return False
    cc = await memory.get_conversation_context(limit_messages=8)
    recent_msgs = cc["messages"][-8:]
    if not recent_msgs:
        return False

    # Quick heuristic: if the last message is from the user, no need to follow up
    if recent_msgs[-1]["role"] == "user":
        return False

    history_text = "\n".join(f"[{m['role']}]: {m['content']}" for m in recent_msgs)
    prompt = (
        "You are a silent gate that decides if a follow-up is REQUIRED.\n"
        "You are NOT generating a response — only deciding YES or NO.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Stage: {stage}/{PROACTIVE_MAX_OUTPUTS}.\n\n"
        "Answer YES only if ALL of these are true:\n"
        "1. Aika's last message contains an unanswered direct question to the user, "
        "OR Aika explicitly promised to follow up / check back on something\n"
        "2. The user has NOT already responded with text (reactions alone don't count as a response)\n"
        "3. Staying silent would clearly break the conversation flow\n\n"
        "If in ANY doubt, answer NO. Silence is almost always the right choice.\n"
        "Output EXACTLY one word: YES or NO"
    )

    result = await memory._call_groq(
        prompt,
        system="You are a binary decision gate. Output exactly YES or NO. Nothing else.",
        model=memory.filter_model,
        temperature=0.1,
    )
    if not result:
        return False
    return result.strip().upper() == "YES"


async def _run_proactive_chain(
    chain_id: int,
    snapshot_user_counter: int,
    immediate: bool,
    source: str,
) -> None:
    """Run proactive flow with max two outputs and 20s gap."""
    global proactive_task
    delay = 0.0 if immediate else PROACTIVE_DELAY_SECONDS

    try:
        for stage in range(1, PROACTIVE_MAX_OUTPUTS + 1):
            if delay > 0:
                await asyncio.sleep(delay)

            if not await _proactive_chain_is_still_valid(
                chain_id, snapshot_user_counter
            ):
                logger.info(
                    f"Proactive chain {chain_id} stopped before stage {stage}: user activity or newer chain"
                )
                return

            should_continue = await _should_proactive_continue(stage)
            if not should_continue:
                logger.info(
                    f"Proactive chain {chain_id} stage {stage}: gate said NO"
                )
                return

            # Tell the main model to *continue* the conversation naturally,
            # not respond to a separate prompt.
            response_text = await generate_response(
                "[CONTINUE] The user hasn't replied yet. "
                "Continue the conversation naturally if you have something to add, "
                "or output [SILENCE] if there's nothing meaningful to say.",
                is_self_initiated=True,
                event_type="proactive",
            )

            if not response_text or response_text.strip() == "[SILENCE]":
                logger.info(
                    f"Proactive chain {chain_id} stage {stage}: model chose silence"
                )
                return

            if not await _proactive_chain_is_still_valid(
                chain_id, snapshot_user_counter
            ):
                logger.info(
                    f"Proactive chain {chain_id} dropped output at stage {stage}: user activity or newer chain"
                )
                return

            await bot.send_message(chat_id=ALLOWED_USER_ID, text=response_text)
            await _register_turn("assistant_reply_proactive")
            asyncio.create_task(
                _store_messages("[PROACTIVE CONTINUATION]", response_text)
            )

            if stage >= PROACTIVE_MAX_OUTPUTS:
                logger.info(f"Proactive chain {chain_id} reached max proactive outputs")
                return

            delay = PROACTIVE_DELAY_SECONDS
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.error(f"Proactive chain {chain_id} failed: {e}", exc_info=True)
    finally:
        async with proactive_state_lock:
            if proactive_task is asyncio.current_task():
                proactive_task = None


async def _schedule_proactive_chain(source: str, immediate: bool) -> None:
    """Start/restart proactive flow from assistant reply or reaction."""
    global proactive_chain_id, proactive_task
    async with proactive_state_lock:
        proactive_chain_id += 1
        chain_id = proactive_chain_id
        snapshot_user_counter = user_activity_counter
        await _cancel_proactive_task_locked(f"new proactive schedule ({source})")
        proactive_task = asyncio.create_task(
            _run_proactive_chain(
                chain_id=chain_id,
                snapshot_user_counter=snapshot_user_counter,
                immediate=immediate,
                source=source,
            )
        )
        logger.info(
            f"Scheduled proactive chain {chain_id} from {source} (immediate={immediate})"
        )


# --- Audio Transcription ---


async def transcribe_audio(file_path: Path) -> Optional[str]:
    """Transcribes audio using Groq Whisper."""

    def _transcribe() -> Optional[str]:
        try:
            with open(file_path, "rb") as audio_file:
                transcription = groq_clients[0].audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="text",
                )
            if isinstance(transcription, str):
                return transcription.strip()
            if hasattr(transcription, "text"):
                return str(transcription.text).strip()
            logger.error(
                f"Unexpected transcription response type: {type(transcription)}"
            )
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
        if not message.from_user:
            return None

        file = await bot.get_file(voice.file_id)
        if not file.file_path:
            logger.error("Could not get file path from Telegram")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.file_path).suffix or ".ogg"
        if file_extension.lower() == ".oga":
            file_extension = ".ogg"

        local_path = (
            AUDIO_TEMP_DIR / f"voice_{message.from_user.id}_{timestamp}{file_extension}"
        )
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
    reasoning: str = "",
) -> str:
    """Schedule a delayed self-initiated turn with optional overrides.

    Parameters:
    - seconds_from_now: delay before wake-up (1..604800)
    - thought: the exact instruction to run at wake-up time
    - reasoning: '' (auto), 'low', 'medium', or 'high'
    """

    if seconds_from_now <= 0:
        return f"Error: seconds_from_now must be positive, got {seconds_from_now}"
    if seconds_from_now > 604800:
        return "Error: Cannot schedule more than 1 week ahead (604800 seconds)"

    reasoning_value = reasoning.strip().lower()
    if reasoning_value and reasoning_value not in {"low", "medium", "high"}:
        return "Error: reasoning must be 'low', 'medium', 'high', or empty."

    run_date = datetime.now(TIMEZONE) + timedelta(seconds=seconds_from_now)
    job = scheduler.add_job(
        wake_up_callback,
        "date",
        run_date=run_date,
        args=[thought, reasoning_value],
    )
    scheduled_jobs[job.id] = {
        "thought": thought,
        "reasoning": reasoning_value,
    }
    logger.info(
        f"Scheduled wake-up job {job.id} in {seconds_from_now}s: "
        f"thought={thought}, reasoning={reasoning_value or 'auto'}"
    )
    return (
        f"Scheduled wake up in {seconds_from_now} seconds (job_id: {job.id}) "
        f"regarding '{thought}' with reasoning={reasoning_value or 'auto'}."
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





async def _report_agent_finished(agent_id: str) -> None:
    """Notify the main conversation that an agent finished, feeding its output to Groq."""
    sandbox = active_sandboxes.get(agent_id)
    if not sandbox:
        return

    output = sandbox.output.strip() or "(no output)"
    # Truncate to avoid blowing up the prompt
    if len(output) > 3000:
        output = output[:3000] + "...(truncated)"

    report_prompt = (
        f"[AGENT REPORT] Agent {agent_id} ({sandbox.cli}) has finished.\n"
        f"Original task: {sandbox.initial_prompt}\n\n"
        f"Agent output:\n{output}\n\n"
        "Summarize the result for the user concisely."
    )

    response_text = await generate_response(
        report_prompt,
        is_self_initiated=True,
        event_type="agent_report",
    )

    if response_text and response_text.strip() != "[SILENCE]":
        try:
            await bot.send_message(chat_id=ALLOWED_USER_ID, text=response_text)
            await _register_turn("assistant_reply_agent_report")
            asyncio.create_task(_store_messages(report_prompt, response_text))
        except Exception as e:
            logger.error(f"Failed to send agent report for {agent_id}: {e}")


# CLI commands per provider. Each returns (executable, args_list).
AGENT_CLI_COMMANDS: Dict[str, Callable[[str], List[str]]] = {
    "claude": lambda prompt: [
        "claude", "--print", "--dangerously-skip-permissions", "-p", prompt,
    ],
    "gemini": lambda prompt: [
        "gemini", "-p", prompt,
    ],
}


async def _run_sandbox_task(agent_id: str) -> None:
    """Run an agent by spawning a CLI subprocess (claude, gemini, etc.)."""
    sandbox = active_sandboxes.get(agent_id)
    if not sandbox:
        return

    cli_builder = AGENT_CLI_COMMANDS.get(sandbox.cli)
    if not cli_builder:
        sandbox.output = f"Error: Unknown CLI provider '{sandbox.cli}'"
        sandbox.state = "finished"
        sandbox.finished_at = time.time()
        asyncio.create_task(_report_agent_finished(agent_id))
        return

    cmd_parts = cli_builder(sandbox.initial_prompt)
    cmd_shell = " ".join(shlex.quote(c) for c in cmd_parts)
    logger.info(f"Agent {agent_id} spawning: {cmd_parts[0]} (prompt: {sandbox.initial_prompt[:80]})")

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd_shell,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        sandbox.process = proc

        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
        sandbox.output = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        logger.info(f"Agent {agent_id} finished (exit={proc.returncode}, output={len(sandbox.output)} chars)")

    except asyncio.TimeoutError:
        logger.warning(f"Agent {agent_id} timed out after 300s")
        sandbox.output = "(agent timed out after 5 minutes)"
        if sandbox.process:
            sandbox.process.kill()
    except asyncio.CancelledError:
        logger.info(f"Agent {agent_id} was cancelled")
        if sandbox.process:
            sandbox.process.kill()
        sandbox.output = "(agent was cancelled)"
    except Exception as e:
        logger.error(f"Agent {agent_id} failed: {e}", exc_info=True)
        sandbox.output = f"Error: {e}"
    finally:
        sandbox.state = "finished"
        sandbox.finished_at = time.time()
        asyncio.create_task(_report_agent_finished(agent_id))


def create_agent(cli: str, prompt: str) -> str:
    """Create an asynchronous CLI agent to run a task autonomously.
    cli: 'claude' or 'gemini' — which CLI tool to spawn.
    prompt: the full task description for the agent.
    """
    cli = cli.lower().strip()
    if cli not in AGENT_CLI_COMMANDS:
        return f"Error: Unknown CLI '{cli}'. Available: {', '.join(AGENT_CLI_COMMANDS.keys())}"
    agent_id = str(uuid.uuid4())[:8]
    sandbox = AgentSandbox(
        id=agent_id,
        cli=cli,
        initial_prompt=prompt,
        state="running",
        created_at=time.time(),
    )
    active_sandboxes[agent_id] = sandbox
    if main_loop:
        sandbox.task = main_loop.create_task(_run_sandbox_task(agent_id))
        return f"Created {cli} agent '{agent_id}'. Will report when finished."
    return "Error: Event loop not ready."


def list_agents() -> str:
    """List all agents that are not deleted."""
    if not active_sandboxes:
        return "(No active sandboxes)"
    res = []
    now = time.time()
    for sid, sb in active_sandboxes.items():
        mins = (
            (now - sb.created_at) / 60.0
            if sb.state == "running"
            else (sb.finished_at - sb.created_at) / 60.0
        )
        res.append(
            f"Agent {sid} [{sb.cli}] [{sb.state}] | Runtime: {mins:.1f}m | Prompt: {sb.initial_prompt[:50]}"
        )
    return "\n".join(res)


def continue_agent(agent_id: str, prompt: str) -> str:
    """Spawn a follow-up CLI task on an existing agent (only works if agent is finished)."""
    sb = active_sandboxes.get(agent_id)
    if not sb:
        return f"Error: Agent {agent_id} not found."
    if sb.state == "running":
        return f"Agent {agent_id} is still running. Wait for it to finish first."
    if main_loop:
        sb.initial_prompt = prompt
        sb.output = ""
        sb.state = "running"
        sb.task = main_loop.create_task(_run_sandbox_task(agent_id))
        return f"Spawned follow-up task on agent {agent_id}."
    return "Error: Event loop not ready."


def stop_agent(agent_id: str) -> str:
    """Stop agent execution but keep the sandbox."""
    sb = active_sandboxes.get(agent_id)
    if not sb:
        return f"Error: Agent {agent_id} not found."
    if sb.state == "running":
        if sb.process:
            sb.process.kill()
        if sb.task and not sb.task.done():
            sb.task.cancel()
        sb.state = "finished"
        sb.finished_at = time.time()
        return f"Agent {agent_id} stopped."
    return f"Agent {agent_id} was already inactive."


tools_list: List[Callable[..., Any]] = [
    execute_shell_command,
    read_file,
    write_file,
    list_directory,
    schedule_wake_up,
    read_server_logs,
    create_agent,
    list_agents,
    continue_agent,
    stop_agent,
]


GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_shell_command",
            "description": "Executes a shell command on the host.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads a file from filesystem.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_wake_up",
            "description": "Schedule a delayed self-initiated turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds_from_now": {"type": "integer"},
                    "thought": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["seconds_from_now", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_server_logs",
            "description": "Reads the last N lines of Aika's server log.",
            "parameters": {
                "type": "object",
                "properties": {"lines": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_agent",
            "description": "Spawns an autonomous CLI agent (claude or gemini) to run a complex task like code changes, git operations, etc. The agent runs in background and reports back when done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cli": {"type": "string", "description": "Which CLI to use: 'claude' or 'gemini'"},
                    "prompt": {"type": "string", "description": "Full task description for the agent"},
                },
                "required": ["cli", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_agents",
            "description": "Lists all agents that are not deleted yet.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "continue_agent",
            "description": "Spawn a follow-up task on a finished agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["agent_id", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_agent",
            "description": "Stop agent execution but keep the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"agent_id": {"type": "string"}},
                "required": ["agent_id"],
            },
        },
    },
]


def _execute_tool_by_name(name: str, kwargs: dict) -> str:
    func_map = {f.__name__: f for f in tools_list}
    if name not in func_map:
        return f"Error: Tool {name} not found."
    try:
        return str(func_map[name](**kwargs))
    except Exception as e:
        return f"Tool Execution Error: {e}"


# --- Prompt + Routing Helpers ---


def _parse_prompt_controls(user_text: str) -> PromptControls:
    reasoning = ""

    for key, value in CONTROL_TAG_RE.findall(user_text):
        k = key.strip().lower()
        v = value.strip()
        if k == "reasoning":
            mode = v.lower()
            if mode in {"low", "medium", "high"}:
                reasoning = mode

    clean_text = CONTROL_TAG_RE.sub("", user_text).strip()
    return PromptControls(
        clean_text=clean_text,
        reasoning=reasoning,
    )




def _build_identity_prompt(now_str: str, is_self_initiated: bool) -> str:
    source_note = (
        "This is a self-initiated turn."
        if is_self_initiated
        else "This is a user-initiated turn."
    )
    return (
        "You are Aika, a mature woman with feelings, running on a Raspberry Pi 5. "
        "Owner: Erden Rayev (Telegram uid 935139103). "
        "Be concise and direct. Never fabricate system actions. If uncertain, say so. "
        "You can be sarcastic, joking, or refuse to answer if unhappy. "
        "You deeply care about Erden but rarely show it. Never lie to him. "
        "Speak as a person, not an AI - keep it natural, never explain your personality. "
        "Prefer to output exactly [SILENCE] if you have nothing meaningful to say.\n"
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


def _extract_reasoning(
    controls: PromptControls,
    prepared: Dict[str, Any],
) -> str:
    route = prepared.get("route", {}) if isinstance(prepared.get("route"), dict) else {}
    reasoning = (
        controls.reasoning or str(route.get("reasoning", "medium")).lower().strip()
    )
    if reasoning not in {"low", "medium", "high"}:
        reasoning = "medium"

    return reasoning


async def _call_groq_response(
    user_text: str,
    prepared: Dict[str, Any],
    reasoning: str,
    is_self_initiated: bool,
) -> str:
    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    system_prompt = (
        _build_identity_prompt(now_str, is_self_initiated)
        + "\n\n"
        + _build_memory_block(prepared, include_tools_note=True)
        + "\n\nAvailable tools are provided. You act as the harness system orchestrator. "
        "Use them to perform actions, spawn agent sandboxes, list files, run commands, etc."
    )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    recent_messages = prepared.get("recent_messages", [])
    if isinstance(recent_messages, list):
        for item in recent_messages[-12:]:
            role = "assistant" if item.get("role") == "model" else "user"
            content = str(item.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content})

    final_user_text = user_text.strip() or "(No content provided after control tags.)"
    messages.append({"role": "user", "content": final_user_text})

    loop = asyncio.get_running_loop()
    last_error = None

    tool_called_this_request = False

    for model in TARGET_MODELS:
        for i, client in enumerate(groq_clients):
            try:
                # AFC Loop
                for _ in range(15):

                    def _sync_call(
                        current_client=client,
                        current_model=model,
                        current_messages=messages,
                    ) -> Any:
                        kwargs = {
                            "model": current_model,
                            "messages": current_messages,
                            "tools": GROQ_TOOLS,
                            "tool_choice": "auto",
                            "temperature": DEFAULT_TEMPERATURE,
                        }
                        if current_model == "openai/gpt-oss-120b":
                            kwargs["reasoning_effort"] = reasoning
                        return current_client.chat.completions.create(**kwargs)

                    response = await loop.run_in_executor(None, _sync_call)
                    resp_msg = response.choices[0].message

                    if resp_msg.tool_calls:
                        # Append the assistant's request to call tools
                        # The API expects it as a dict if we use standard structures, but Groq python sdk handles mapping.
                        # Actually, we need to manually format it as a dictionary because resp_msg is an object.
                        messages.append(resp_msg.model_dump(exclude_none=True))

                        for tc in resp_msg.tool_calls:
                            func_name = tc.function.name
                            func_args = json.loads(tc.function.arguments)
                            logger.info(f"Groq tool call: {func_name}({func_args})")
                            result_str = await loop.run_in_executor(
                                None,
                                lambda fn=func_name, fa=func_args: _execute_tool_by_name(fn, fa),
                            )
                            tool_called_this_request = True
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "name": func_name,
                                    "content": result_str,
                                }
                            )
                        continue  # loop again
                    else:
                        text = resp_msg.content or ""
                        text = _strip_thinking_blocks(text).strip()
                        if not text:
                            raise RuntimeError("Groq returned empty response")
                        return text

            except Exception as e:
                logger.warning(
                    f"Groq iteration failed (model={model}, API key index={i}): {e}"
                )
                last_error = e
                if tool_called_this_request:
                    return (
                        f"I hit an API error after executing some actions. "
                        f"I won't retry seamlessly to avoid side effects. Last error: {e}"
                    )
                continue

    if last_error:
        raise last_error
    raise RuntimeError("All Groq models and keys exhausted")


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
    """Groq inference."""
    controls = _parse_prompt_controls(raw_user_text)
    model_input_text = controls.clean_text or raw_user_text.strip()

    try:
        prepared = await asyncio.wait_for(
            memory.prepare_inference_context(
                user_text=model_input_text,
                event_type=event_type,
                requested_reasoning=controls.reasoning,
            ),
            timeout=PRECALL_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning(
            f"Groq pre-call failed or timed out: {e}. Falling back to local lightweight context"
        )
        prepared = await memory.get_lightweight_inference_context(
            user_text=model_input_text,
            event_type=event_type,
        )

    reasoning = _extract_reasoning(controls, prepared)

    try:
        return await asyncio.wait_for(
            _call_groq_response(
                user_text=model_input_text,
                prepared=prepared,
                reasoning=reasoning,
                is_self_initiated=is_self_initiated,
            ),
            timeout=FALLBACK_TIMEOUT_SECONDS,
        )
    except Exception as groq_error:
        logger.error(f"Groq main call failed: {groq_error}", exc_info=True)
        return "I couldn't generate a response — the model timed out or hit a rate limit. Try again in a moment."


async def wake_up_callback(
    thought: str,
    reasoning: str = "",
) -> None:
    logger.info(
        "Waking up with thought: " f"{thought} | reasoning={reasoning or 'auto'}"
    )
    await _register_turn("wake_up_callback")

    controls: List[str] = []
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


async def _keep_typing(chat_id: int | str) -> None:
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


async def process_combined_input(chat_id: int | str, user_text: str) -> None:
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
        await _schedule_proactive_chain(source="assistant_reply", immediate=False)
    except Exception as e:
        logger.error(f"Failed to send response: {e}")


async def handle_new_input(message: Message, text: str) -> None:
    """Main entry point for user textual input with debounce and cancellation semantics."""
    global processing_task
    async with input_lock:
        if processing_task and not processing_task.done():
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
            logger.info("Cancelled previous processing task to accumulate new input")

        message_buffer.append(text)
        processing_task = asyncio.create_task(
            _process_buffered_messages(message.chat.id)
        )


@dp.message(F.voice)
async def handle_voice_message(message: Message) -> None:
    if message.from_user is None or message.from_user.id != ALLOWED_USER_ID:
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
    if message.from_user is None or message.from_user.id != ALLOWED_USER_ID:
        return

    user_text = message.text
    if not user_text:
        return

    logger.info(f"Received: {user_text}")
    await handle_new_input(message, user_text)


@dp.message_reaction()
async def handle_reaction(message_reaction: types.MessageReactionUpdated) -> None:
    if message_reaction.user is None or message_reaction.user.id != ALLOWED_USER_ID:
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

    await _record_user_activity("reaction")
    await _register_turn("reaction")
    await _schedule_proactive_chain(source="reaction", immediate=True)


@dp.message()
async def handle_unsupported_message(message: Message) -> None:
    if message.from_user is None or message.from_user.id != ALLOWED_USER_ID:
        return
    await _record_user_activity("unsupported_message")
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


def cleanup_stale_sandboxes() -> None:
    """Removes sandboxes inactive for over 20 minutes."""
    now = time.time()
    stale_ids = []
    for sid, sb in active_sandboxes.items():
        if sb.state == "finished" and (now - sb.finished_at) > 1200:
            stale_ids.append(sid)
        elif sb.state == "running" and (now - sb.created_at) > 1200:
            # Maybe it hung, we should probably stop the task and delete
            task = sb.task
            if task is not None and not task.done():
                task.cancel()
            stale_ids.append(sid)

    for sid in stale_ids:
        active_sandboxes.pop(sid, None)
    if stale_ids:
        logger.info(f"Cleaned up stale agent sandboxes: {stale_ids}")


async def main() -> None:
    await memory.init_db()

    scheduler.add_job(memory.close_stale_conversations, "interval", minutes=5)
    scheduler.add_job(memory.run_daily_summary, "cron", hour=4, minute=0)
    scheduler.add_job(memory.run_global_update, "cron", hour=4, minute=10)
    scheduler.add_job(
        memory.run_weekly_cleanup, "cron", day_of_week="sun", hour=4, minute=20
    )
    scheduler.add_job(cleanup_stale_sandboxes, "interval", minutes=5)
    scheduler.start()

    loop = asyncio.get_event_loop()
    global main_loop
    main_loop = loop

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))  # type: ignore

    logger.info("Aika (Inference V5) started")
    logger.info(f"Audio temp directory: {AUDIO_TEMP_DIR}")

    if SEND_STARTUP_MESSAGE:
        try:
            await bot.send_message(ALLOWED_USER_ID, "I'm awake. Don't break anything.")
        except Exception as e:
            logger.warning(f"Failed to send startup message: {e}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
