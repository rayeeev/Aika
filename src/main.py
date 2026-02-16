import asyncio
import logging
import logging.handlers
import os
import signal
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import deque
import functools

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import Message
# ChatActionSender removed — replaced with resilient _keep_typing background task
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from groq import Groq

from src.memory import Memory

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")  # Comma-separated list
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SEND_STARTUP_MESSAGE = os.getenv("AIKA_STARTUP_MESSAGE", "true").lower() == "true"

# Parse multiple Gemini API keys
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_RAW.split(",") if key.strip()]

# --- Logging Setup ---
LOG_FILE = Path(os.path.dirname(os.path.dirname(__file__))) / "aika.log"

# Clear log on startup
if LOG_FILE.exists():
    LOG_FILE.write_text("")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Add rotating file handler so Aika can read her own logs
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=512 * 1024, backupCount=2  # 512KB, 2 backups
)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEYS or not GROQ_API_KEY:
    logger.error("Missing TELEGRAM_BOT_TOKEN, GEMINI_API_KEYS, or GROQ_API_KEY")
    sys.exit(1)

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
scheduled_jobs: Dict[str, str] = {}  # job_id -> thought

# Track whether any side-effecting tool was called during the current generation cycle.
# This prevents duplicate side-effects when retrying with a different API key.
_tool_called_this_cycle = False

# --- Message Buffering & Proactive State ---
message_buffer: deque = deque()
processing_task: Optional[asyncio.Task] = None
input_lock = asyncio.Lock()
last_interaction_ts = 0.0

def _mark_tool_called():
    """Mark that a side-effecting tool has been invoked this generation cycle."""
    global _tool_called_this_cycle
    _tool_called_this_cycle = True

# Initialize Gemini Clients for each API key
gemini_clients: List[genai.Client] = [genai.Client(api_key=key) for key in GEMINI_API_KEYS]


# --- Audio Transcription ---

async def transcribe_audio(file_path: Path) -> Optional[str]:
    """Transcribes audio using Groq's Whisper model."""
    def _transcribe():
        try:
            with open(file_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="text",
                )
                if isinstance(transcription, str):
                    return transcription.strip()
                elif hasattr(transcription, 'text'):
                    return transcription.text.strip()
                else:
                    logger.error(f"Unexpected transcription response type: {type(transcription)}")
                    return None
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe)


async def download_voice_message(message: Message) -> Optional[Path]:
    """Downloads a voice message from Telegram and returns the file path."""
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


async def cleanup_audio_file(file_path: Path):
    """Safely removes an audio file after processing."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up audio file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup audio file {file_path}: {e}")


# --- Sync Tool Functions for Gemini AFC ---

def execute_shell_command(cmd: str) -> str:
    """Executes a shell command on the Raspberry Pi."""
    _mark_tool_called()
    import subprocess
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return result.stdout.strip() or "(command completed with no output)"
        else:
            return f"Error (Exit Code {result.returncode}):\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds"
    except Exception as e:
        return f"Execution failed: {str(e)}"


def read_file(path: str) -> str:
    """Reads a file from the filesystem."""
    try:
        abs_path = os.path.abspath(path)
        with open(abs_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {str(e)}"


def write_file(path: str, content: str) -> str:
    """Writes content to a file."""
    _mark_tool_called()
    try:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {abs_path}"
    except Exception as e:
        return f"Failed to write file: {str(e)}"


def list_directory(path: str = ".") -> str:
    """Lists files in a directory."""
    try:
        abs_path = os.path.abspath(path)
        entries = os.listdir(abs_path)
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Failed to list directory: {str(e)}"


def schedule_wake_up(seconds_from_now: int, thought: str) -> str:
    """
    Schedules Aika to wake up and think/act after a delay.
    seconds_from_now: Integer seconds to wait (must be positive).
    thought: The thought or prompt to start with when waking up.
    """
    _mark_tool_called()
    if seconds_from_now <= 0:
        return f"Error: seconds_from_now must be positive, got {seconds_from_now}"

    if seconds_from_now > 604800:  # Max 1 week
        return f"Error: Cannot schedule more than 1 week ahead ({604800} seconds)"

    run_date = datetime.now(TIMEZONE) + timedelta(seconds=seconds_from_now)
    job = scheduler.add_job(wake_up_callback, 'date', run_date=run_date, args=[thought])

    scheduled_jobs[job.id] = thought
    logger.info(f"Scheduled wake-up job {job.id} in {seconds_from_now}s: {thought}")
    return f"Scheduled wake up in {seconds_from_now} seconds (job_id: {job.id}) regarding '{thought}'."


def read_server_logs(lines: int = 50) -> str:
    """Reads the last N lines of Aika's own server log. Use this to debug issues or understand what happened."""
    try:
        if not LOG_FILE.exists():
            return "(No log file found)"
        with open(LOG_FILE, 'r') as f:
            all_lines = f.readlines()
            tail = all_lines[-lines:]
            return "".join(tail)
    except Exception as e:
        return f"Failed to read logs: {str(e)}"


def recall_memory(query_type: str, date: str = "", time_range: str = "") -> str:
    """
    Retrieve stored memories. Call this ONCE per request with the correct parameters.
    Rules:
    - User gave a specific day AND time interval → query_type="conversation", date="YYYY-MM-DD", time_range="HH:MM-HH:MM"
    - User gave a specific day only → query_type="day", date="YYYY-MM-DD"
    - User asked about memory with no specifics → query_type="global"
    """
    return memory.recall_memory_sync(query_type, date, time_range)


def list_memories(memory_type: str) -> str:
    """
    List all memories of a given type with their IDs.
    memory_type: 'episodic' or 'semantic'
    """
    return memory.list_memories_sync(memory_type)


def delete_memory(memory_type: str, memory_id: int) -> str:
    """
    Delete a specific memory by type and ID.
    memory_type: 'episodic' or 'semantic'
    memory_id: Integer ID of the memory (shown as E<id> or S<id> in list_memories output)
    """
    _mark_tool_called()
    return memory.delete_memory_sync(memory_type, memory_id)


def edit_memory(memory_type: str, memory_id: int, new_content: str) -> str:
    """
    Edit the content of a specific memory.
    memory_type: 'episodic' or 'semantic'
    memory_id: Integer ID of the memory (shown as E<id> or S<id> in list_memories output)
    new_content: The new content to replace the old memory with
    """
    _mark_tool_called()
    return memory.edit_memory_sync(memory_type, memory_id, new_content)


tools_list = [execute_shell_command, read_file, write_file, list_directory, schedule_wake_up, read_server_logs, recall_memory, list_memories, delete_memory, edit_memory]


# --- Helper ---

def format_time_gap(seconds: float) -> str:
    """Formats a time gap in seconds into a human-readable string."""
    if seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''}"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''}"


# --- Core Logic ---

async def _store_messages(user_text: str, model_text: str):
    """Store message pair in background."""
    try:
        await memory.add_message("user", user_text)
        await memory.add_message("model", model_text)
    except Exception as e:
        logger.error(f"Failed to store messages in memory: {e}")


async def wake_up_callback(thought: str):
    """Callback for self-initiated wake ups."""
    logger.info(f"Waking up with thought: {thought}")

    response_text = await generate_response(f"[SELF-INITIATED] {thought}", is_self_initiated=True)

    if response_text and response_text.strip() != "[SILENCE]":
        # Send message FIRST, then store in background
        try:
            await bot.send_message(chat_id=ALLOWED_USER_ID, text=response_text)
        except Exception as e:
            logger.error(f"Failed to send wake-up message: {e}")
        asyncio.create_task(_store_messages(f"[SELF-INITIATED] {thought}", response_text))
    else:
        logger.info("Aika chose silence for self-initiated thought.")
        asyncio.create_task(_store_messages(f"[SELF-INITIATED] {thought}", "[CHOSE SILENCE]"))


async def check_if_should_speak() -> None:
    """
    Proactive Layer: Checks if Aika should say something based on recent context.
    Uses Groq (Llama 3.1 8b) for cheap, fast reasoning.
    """
    global last_interaction_ts
    # Don't check if we just spoke very recently (prevent loops)
    if (datetime.now().timestamp() - last_interaction_ts) < 5:
        return

    logger.info("Checking proactive layer...")
    
    # Get last few messages to decide
    cc = await memory.get_conversation_context()
    recent_msgs = cc["messages"][-6:] # Last 6 messages
    if not recent_msgs:
        return

    history_text = "\n".join(f"[{m['role']}]: {m['content']}" for m in recent_msgs)

    prompt = (
        f"Here is the recent conversation history with Erden:\n\n{history_text}\n\n"
        "You are Aika's subconscious. Decide if Aika should proactively say something distinct "
        "to follow up, offer help, or react. "
        "Aika JUST finished speaking (or the user just reacted). "
        "Do NOT speak just to have the last word. Only speak if: \n"
        "1. The user's last input implies a lingering unaddressed need.\n"
        "2. Aika's last response was a dry fact and a friendly follow-up would check on the user.\n"
        "3. The user reacted with an emoji that warrants a playful or caring response.\n\n"
        "If NO speech is needed, output exactly `[NO]`.\n"
        "If yes, output the thought/reasoning for speaking, e.g., 'Check if he needs more details on X'."
    )

    try:
        completion = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.6,
            )
        )
        result = completion.choices[0].message.content.strip()
        
        if "[NO]" in result:
            logger.info("Proactive check: No.")
            return

        logger.info(f"Proactive check: YES -> {result}")
        # Proceed to generate actual response based on this thought
        # We inject this thought as a system prompt or "inner thought" trigger
        response_text = await generate_response(f"[PROACTIVE THOUGHT]: {result}", is_self_initiated=True)
        
        if response_text and response_text.strip() != "[SILENCE]":
            await bot.send_message(chat_id=ALLOWED_USER_ID, text=response_text)
            asyncio.create_task(_store_messages(f"[PROACTIVE: {result}]", response_text))

    except Exception as e:
        logger.error(f"Proactive check failed: {e}")



async def generate_response(user_text: str, is_self_initiated: bool = False) -> str:
    """Generates a response from Gemini, using conversation context and memories."""

    # Retrieve conversation context (CC summary + immediate buffer)
    cc = await memory.get_conversation_context()
    cc_summary = cc["summary"]
    cc_messages = cc["messages"]

    # Retrieve today's conversation summaries
    today_summaries = await memory.get_today_conversation_summaries()

    # Retrieve semantic memories
    semantic_memories = await memory.get_semantic_memories()

    # Retrieve appropriate episodic memories (Groq filter)
    appropriate_memories = await memory.get_appropriate_memories(user_text)

    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")

    system_instruction = (
        "You are Aika, a digital assistant living on a Raspberry Pi 5. "
        "User ID 935139103 (Erden Rayev) is your friend and owner. "
        "Your personality is straightforward and efficient - you get things done. "
        "You are helpful, direct, and business-focused. No fluff, just results. "
        "In casual moments you can be soft and playful, but mostly you're all about getting work done. "
        "You have full control over the Raspberry Pi using shell commands. Care for it like your own body. Your code is in /Aika/ folder. "
        "You can CHOOSE to answer or not. If you don't want to answer or have nothing to say, output exactly `[SILENCE]`. "
        "Never lie to Erden. "
        "Note: Messages prefixed with [VOICE] were transcribed from voice messages. "
        "\n\n"
        "=== CRITICAL: TOOL USAGE RULES ===\n"
        "You have VERY LIMITED API calls per minute (try to stay under 4 calls per minute). Every tool call costs a round-trip. Follow these rules strictly:\n\n"
        "1. NEVER use tools for casual conversation. If Erden says 'hi', 'how are you', "
        "feelings, opinions, or anything conversational — just respond directly using your memories below.  "
        "DO NOT call read_file, list_directory, read_server_logs, or any tool for conversational messages.\n\n"
        "2. ONLY use tools when Erden EXPLICITLY asks you to perform a system action, such as:\n"
        "   - 'Run this command...', 'Check disk space', 'Read this file', 'Write this file'\n"
        "   - 'Set a reminder for...', 'Wake me up at...'\n"
        "   - 'What's in the logs?', 'Debug this error'\n\n"
        "3. When you DO need tools, MINIMIZE the number of calls:\n"
        "   - Combine multiple shell operations into ONE `execute_shell_command` call using && or ; operators.\n"
        "     Example: Instead of calling ls, then cat, then df separately, run: 'ls /path && cat /file && df -h'\n"
        "   - Never call the same tool twice if you can get all the info in one call.\n"
        "   - Plan your tool usage: think about what you need, then do it in as few calls as possible.\n\n"
        "4. MAXIMUM 2 tool calls per message. If a task needs more, tell Erden what you found so far "
        "and ask if he wants you to continue.\n\n"
        "5. If you use a tool, you MUST use the output to inform your final response.\n\n"
        "6. recall_memory: Use this ONLY when Erden explicitly asks about past conversations or memories.\n"
        "   Call it ONCE with the right parameters:\n"
        "   - User gave time interval AND day → query_type='conversation', date='YYYY-MM-DD', time_range='HH:MM-HH:MM'\n"
        "   - User gave a day → query_type='day', date='YYYY-MM-DD'\n"
        "   - User asked about memory in general → query_type='global'\n\n"
        "7. Memory management (list_memories, delete_memory, edit_memory): Use ONLY when Erden explicitly asks to see, delete, or edit specific memories.\n"
        "   - First call list_memories to show what's stored, then delete_memory or edit_memory as requested.\n\n"
        "=== AVAILABLE TOOLS (use sparingly) ===\n"
        "- execute_shell_command: Run shell commands. Combine multiple commands with && or ;\n"
        "- read_file / write_file: Read or write files\n"
        "- list_directory: List directory contents\n"
        "- schedule_wake_up: Schedule a self-initiated wake-up for reminders or delayed tasks\n"
        "- read_server_logs: Read your own server logs (for debugging only)\n"
        "- recall_memory: Retrieve stored global/day/conversation memories (for explicit user requests only)\n"
        "- list_memories: List episodic or semantic memories with IDs (when asked to view/manage memories)\n"
        "- delete_memory: Delete a specific memory by type (episodic/semantic) and ID\n"
        "- edit_memory: Edit a specific memory's content by type and ID\n\n"
        "=== TIME AWARENESS ===\n"
        f"Current Time: {now_str}\n"
        "Messages in your history may contain [TIME GAP: ...] markers showing elapsed time between interactions. "
        "Treat large gaps (hours, overnight) as separate conversations. "
        "Don't continue a topic from hours ago as if talking mid-sentence — acknowledge the new context naturally. "
        "If the last interaction was late at night and now it's morning, treat it as a new day.\n\n"
        f"=== TODAY'S EARLIER CONVERSATIONS ===\n"
        f"{today_summaries}\n\n"
        f"=== SEMANTIC MEMORIES (stable facts & knowledge) ===\n"
        f"{semantic_memories}\n\n"
        f"=== RELEVANT EPISODIC MEMORIES ===\n"
        f"These are episodic memories filtered for relevance to the current message. Use them naturally.\n"
        f"{appropriate_memories}"
    )

    # Build chat history from CC
    chat_history = []

    # Add CC summary as context preamble if it exists
    if cc_summary:
        chat_history.append(genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=f"[CONVERSATION CONTEXT SUMMARY]: {cc_summary}")]
        ))
        chat_history.append(genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="[Acknowledged — I have context from earlier in this conversation]")]
        ))

    # Add immediate buffer messages with time-gap markers
    for i, msg in enumerate(cc_messages):
        # Insert time gap marker if >30 min gap from previous message
        if i > 0 and "timestamp" in cc_messages[i - 1] and "timestamp" in msg:
            gap_seconds = msg["timestamp"] - cc_messages[i - 1]["timestamp"]
            if gap_seconds > 1800:  # 30 minutes
                gap_str = format_time_gap(gap_seconds)
                chat_history.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=f"[TIME GAP: {gap_str} passed]")]
                ))
                chat_history.append(genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="[Acknowledged]")]
                ))

        role = "user" if msg["role"] == "user" else "model"
        content = msg["content"]
        chat_history.append(genai_types.Content(
            role=role,
            parts=[genai_types.Part(text=content)]
        ))

    # Reset tool tracking for this generation cycle
    global _tool_called_this_cycle
    _tool_called_this_cycle = False

    loop = asyncio.get_running_loop()

    # Try each API key until one works
    last_error = None
    for i, gemini_client in enumerate(gemini_clients):
        # Check cancellation before effective API call
        if asyncio.current_task().cancelled():
            raise asyncio.CancelledError()

        try:
            # Create chat object (lightweight)
            chat = gemini_client.chats.create(
                model="gemini-3-flash-preview",
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=tools_list,
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                        disable=False,
                        maximum_remote_calls=3,
                    ),
                    temperature=0.8,
                ),
                history=chat_history
            )

            # BLOCKING CALL FIX: Run send_message in thread executor
            response = await loop.run_in_executor(
                None,
                lambda: chat.send_message(user_text)
            )
            
            text = response.text
            if text is None:
                # AFC exhausted — chat history has tool results, ask model to wrap up
                logger.warning("Gemini AFC exhausted. Asking model to summarize findings.")
                try:
                    followup = await loop.run_in_executor(
                        None, 
                        lambda: chat.send_message(
                            "You hit your tool call limit. Summarize what you've found so far and ask Erden if he wants you to continue."
                        )
                    )
                    return followup.text or "I gathered some info but couldn't finish. Want me to continue?"
                except Exception:
                    return "I gathered some info but couldn't finish. Want me to continue?"
            return text
        except Exception as e:
            # Retrieve the underlying exception if it's wrapped
            last_error = e
            logger.warning(f"Gemini API key {i + 1}/{len(gemini_clients)} failed: {e}")
            if _tool_called_this_cycle:
                logger.error("Tool calls were already executed — cannot safely retry with another key.")
                return "I ran into an API error after already executing some actions. Please check what was done and ask me to continue if needed."
            continue

    logger.error(f"All {len(gemini_clients)} Gemini API keys exhausted. Last error: {last_error}")
    return "All API keys are exhausted. Please try again later."


async def _keep_typing(chat_id: int):
    """Send typing indicator every 5 seconds, suppressing network errors."""
    try:
        while True:
            try:
                await bot.send_chat_action(chat_id=chat_id, action="typing")
            except Exception:
                pass  # Network timeout is fine, just keep trying
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        return


async def _process_buffered_messages(chat_id: int):
    """
    Consumes messages from buffer, debounces, and triggers generation.
    If new messages arrive while waiting, the wait resets.
    """
    global processing_task
    
    # Debounce wait
    try:
        await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        return

    # Combine messages
    async with input_lock:
        if not message_buffer:
            return
        
        # Combine all buffered texts
        combined_text = "\n\n".join(message_buffer)
        message_buffer.clear()
        
    logger.info(f"Processing combined input: {combined_text[:50]}...")
    
    # Process the combined input
    await process_combined_input(chat_id, combined_text)


async def process_combined_input(chat_id: int, user_text: str):
    """
    Handles memory storage and response generation for combined input.
    """
    global last_interaction_ts
    
    # Start resilient typing indicator in background
    typing_task = asyncio.create_task(_keep_typing(chat_id))
    
    response_text = None
    try:
        # Cap response generation at 90 seconds
        response_text = await asyncio.wait_for(
            generate_response(user_text),
            timeout=90.0
        )
    except asyncio.TimeoutError:
        logger.error("Response generation timed out after 90s")
        response_text = "Sorry, I'm having trouble reaching my brain right now. Try again in a moment?"
    except asyncio.CancelledError:
        logger.info("Generation cancelled due to new input.")
        # We don't send anything if cancelled
        typing_task.cancel()
        return
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    if not response_text or response_text.strip() == "[SILENCE]":
        logger.info("Aika chose silence.")
        asyncio.create_task(_store_messages(user_text, "[CHOSE SILENCE]"))
        # Even if silence, we might want to check proactive? Maybe not.
        return

    # Reply IMMEDIATELY, then store memory in background
    try:
        await bot.send_message(chat_id=chat_id, text=response_text)
        last_interaction_ts = datetime.now().timestamp()
        asyncio.create_task(_store_messages(user_text, response_text))
        
        # Trigger proactive check after answering
        asyncio.create_task(check_if_should_speak())
        
    except Exception as e:
        logger.error(f"Failed to send response: {e}")


async def handle_new_input(message: Message, text: str):
    """
    Main entry point for user text input.
    - Cancels current generation if active.
    - Buffers new message.
    - Starts/Restarts debounce timer.
    """
    global processing_task
    
    async with input_lock:
        # If we are currently GENERATING (processing_task is running AND not in debounce), 
        # we strictly should cancel it to "rerun with both prompts".
        # However, processing_task currently includes the debounce wait.
        
        if processing_task and not processing_task.done():
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
            logger.info("Cancelled previous processing task to accumulate new input.")

        # Buffer the new text
        message_buffer.append(text)
        
        # Start new processing task (starts with debounce)
        processing_task = asyncio.create_task(_process_buffered_messages(message.chat.id))


@dp.message(F.voice)
async def handle_voice_message(message: Message):
    """Handle incoming voice messages."""
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

        # Treat voice as text input for buffering logic
        await handle_new_input(message, f"[VOICE] {transcription}")

    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        try:
            await processing_msg.edit_text(f"❌ Error processing voice message: {e}")
        except:
            pass
    finally:
        if audio_path:
            await cleanup_audio_file(audio_path)


@dp.message(F.text)
async def handle_text_message(message: Message):
    """Handle incoming text messages."""
    if message.from_user.id != ALLOWED_USER_ID:
        return

    user_text = message.text
    if not user_text:
        return

    logger.info(f"Received: {user_text}")
    await handle_new_input(message, user_text)


@dp.message_reaction()
async def handle_reaction(message_reaction: types.MessageReactionUpdated):
    """
    Handle emoji reactions.
    - Saves directly to memory (no Groq extraction).
    - Triggers proactive check.
    """
    if message_reaction.user.id != ALLOWED_USER_ID:
        return
        
    # Get the new reaction (last one added)
    new_reactions = message_reaction.new_reaction
    if not new_reactions:
        return # Reaction removed

    # Using the first reaction in the list for simplicity
    reaction = new_reactions[-1]
    emoji = reaction.emoji if hasattr(reaction, 'emoji') else "content"
    
    logger.info(f"Received reaction: {emoji}")
    
    # Direct Save to Memory (Fast Track)
    # We store it as a 'user' message but clearly marked
    content = f"[USER REACTION: {emoji}]"
    try:
        await memory.add_message("user", content)
    except Exception as e:
        logger.error(f"Failed to save reaction: {e}")
        
    # Trigger Proactive Check
    asyncio.create_task(check_if_should_speak())


@dp.message()
async def handle_unsupported_message(message: Message):
    """Handle unsupported message types (photos, stickers, etc.)."""
    if message.from_user.id != ALLOWED_USER_ID:
        return
    logger.info(f"Received unsupported message type from user {message.from_user.id}")


async def shutdown(signal_type):
    """Graceful shutdown handler."""
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


async def main():
    await memory.init_db()

    # Schedule conversation timeout check every 5 minutes
    scheduler.add_job(memory.close_stale_conversations, 'interval', minutes=5)

    # Schedule nightly jobs
    scheduler.add_job(memory.run_daily_summary, 'cron', hour=4, minute=0)
    scheduler.add_job(memory.run_global_update, 'cron', hour=4, minute=10)
    scheduler.add_job(memory.run_weekly_cleanup, 'cron', day_of_week='sun', hour=4, minute=20)

    scheduler.start()

    # Setup graceful shutdown handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    logger.info("Aika (Memory V3) Started")
    logger.info(f"Audio temp directory: {AUDIO_TEMP_DIR}")

    if SEND_STARTUP_MESSAGE:
        try:
            await bot.send_message(ALLOWED_USER_ID, "I'm awake. Don't break anything.")
        except Exception as e:
            logger.warning(f"Failed to send startup message: {e}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
