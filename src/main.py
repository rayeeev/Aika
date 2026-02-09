import asyncio
import logging
import os
import signal
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo
from pathlib import Path

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from groq import Groq

from src.memory import Memory
from src.tools import Tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")  # Comma-separated list
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SEND_STARTUP_MESSAGE = os.getenv("AIKA_STARTUP_MESSAGE", "true").lower() == "true"

# Parse multiple Gemini API keys
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_RAW.split(",") if key.strip()]

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

# Track scheduled wake-up jobs for potential cancellation
scheduled_jobs: Dict[str, str] = {}  # job_id -> thought

# Initialize Gemini Clients for each API key
gemini_clients: List[genai.Client] = [genai.Client(api_key=key) for key in GEMINI_API_KEYS]


# --- Audio Transcription ---

async def transcribe_audio(file_path: Path) -> Optional[str]:
    """
    Transcribes audio using Groq's Whisper model.
    Returns the transcribed text or None on error.
    """
    def _transcribe():
        try:
            with open(file_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="text",
                )
                # Handle both string and object responses
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
    
    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe)


async def download_voice_message(message: Message) -> Optional[Path]:
    """
    Downloads a voice message from Telegram and returns the file path.
    Returns None if download fails.
    """
    try:
        voice = message.voice
        if not voice:
            return None
        
        # Get file info from Telegram
        file = await bot.get_file(voice.file_id)
        if not file.file_path:
            logger.error("Could not get file path from Telegram")
            return None
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.file_path).suffix or ".ogg"
        
        # Normalize .oga to .ogg (same format, Groq only accepts .ogg)
        if file_extension.lower() == ".oga":
            file_extension = ".ogg"
        
        local_path = AUDIO_TEMP_DIR / f"voice_{message.from_user.id}_{timestamp}{file_extension}"
        
        # Download the file
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
# Gemini's automatic function calling runs tools synchronously in a thread,
# so we use direct synchronous implementations here.

def execute_shell_command(cmd: str) -> str:
    """Executes a shell command on the Raspberry Pi."""
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
    # Validate input
    if seconds_from_now <= 0:
        return f"Error: seconds_from_now must be positive, got {seconds_from_now}"
    
    if seconds_from_now > 604800:  # Max 1 week
        return f"Error: Cannot schedule more than 1 week ahead ({604800} seconds)"
    
    run_date = datetime.now(TIMEZONE) + timedelta(seconds=seconds_from_now)
    job = scheduler.add_job(wake_up_callback, 'date', run_date=run_date, args=[thought])
    
    # Track the job
    scheduled_jobs[job.id] = thought
    logger.info(f"Scheduled wake-up job {job.id} in {seconds_from_now}s: {thought}")
    
    return f"Scheduled wake up in {seconds_from_now} seconds (job_id: {job.id}) regarding '{thought}'."


tools_list = [execute_shell_command, read_file, write_file, list_directory, schedule_wake_up]

# --- Core Logic ---

async def wake_up_callback(thought: str):
    """Callback for self-initiated wake ups."""
    logger.info(f"Waking up with thought: {thought}")
    
    # Add the self-initiated thought as a "user" message to maintain pair consistency
    await memory.add_message("user", f"[SELF-INITIATED] {thought}")
    
    # Trigger an autonomous response cycle
    response_text = await generate_response(thought, is_self_initiated=True)
    
    if response_text and response_text.strip() != "[SILENCE]":
        await memory.add_message("model", response_text)  # Save response to memory
        try:
            await bot.send_message(chat_id=ALLOWED_USER_ID, text=response_text)
        except Exception as e:
            logger.error(f"Failed to send wake-up message: {e}")
    else:
        # Even on silence, store a placeholder to maintain message pairing
        await memory.add_message("model", "[CHOSE SILENCE]")
        logger.info("Aika chose silence for self-initiated thought.")


async def generate_response(user_text: str, is_self_initiated: bool = False) -> str:
    """Generates a response from Gemini, handling tools and history."""
    
    # Retrieve recent history (Buffer)
    history = await memory.get_recent_messages()
    
    # Retrieve Summaries
    global_sum = await memory.get_summary("global_summary") or "(None)"
    weekly_sum = await memory.get_summary("weekly_summary") or "(None)"

    system_instruction = (
        "You are Aika, a digital assistant living on a Raspberry Pi 5. "
        "User ID 935139103 (Erden Rayev) is your friend and owner. "
        "Your personality is straightforward and efficient - you get things done. "
        "You are helpful, direct, and business-focused. No fluff, just results. "
        "In casual moments you can be soft and playful, but mostly you're all about getting work done. "
        "You have full control over the Raspberry Pi using shell commands. Care for it like your own body. Your code is in /Aika/ folder. "
        "You can CHOOSE to answer or not. If you don't want to answer or have nothing to say, output exactly `[SILENCE]`. "
        "You can schedule your own wake-ups using `schedule_wake_up` to check things later or remind yourself/Erden of things. "
        "You can also use `list_directory` to explore the filesystem. "
        "IMPORTANT: Only use tools when specifically needed. For casual conversation, just respond directly without using tools. "
        "If you use a tool, you MUST use the output to inform your final response. "
        "Never lie to Erden. "
        "Note: Messages prefixed with [VOICE] were transcribed from voice messages. "
        "Current Time: " + datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
        "LONG-TERM MEMORY (Global Summary):\n" + global_sum + "\n\n"
        "SHORT-TERM MEMORY (Weekly Summary):\n" + weekly_sum
    )

    chat_history = []
    # Convert history to Gemini format
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        chat_history.append(genai_types.Content(
            role=role,
            parts=[genai_types.Part(text=msg["content"])]
        ))

    # Try each API key until one works
    last_error = None
    for i, gemini_client in enumerate(gemini_clients):
        try:
            chat = gemini_client.chats.create(
                model="gemini-3-flash-preview",
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=tools_list,
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=False),
                    temperature=0.8,  # More creative/personable
                ),
                history=chat_history
            )
            response = chat.send_message(user_text)
            return response.text
        except Exception as e:
            last_error = e
            logger.warning(f"Gemini API key {i + 1}/{len(gemini_clients)} failed: {e}")
            continue
    
    # All keys exhausted
    logger.error(f"All {len(gemini_clients)} Gemini API keys exhausted. Last error: {last_error}")
    return "All API keys are exhausted. Please try again later."


async def process_user_input(message: Message, user_text: str, is_voice: bool = False):
    """
    Common handler for processing user input (text or transcribed voice).
    Handles memory storage and response generation.
    """
    # Prefix voice messages for context
    stored_text = f"[VOICE] {user_text}" if is_voice else user_text
    
    await memory.add_message("user", stored_text)
    
    response_text = await generate_response(user_text)
    
    if response_text.strip() == "[SILENCE]":
        # Store a placeholder to maintain message pairing
        await memory.add_message("model", "[CHOSE SILENCE]")
        logger.info("Aika chose silence.")
        return

    await memory.add_message("model", response_text)
    # Use reply() for proper message threading
    await message.reply(response_text)


@dp.message(F.voice)
async def handle_voice_message(message: Message):
    """Handle incoming voice messages."""
    if message.from_user.id != ALLOWED_USER_ID:
        # Security: Ignore messages from unauthorized users
        return

    logger.info(f"Received voice message from user {message.from_user.id}")
    
    # Send a "processing" indicator
    processing_msg = await message.reply("🎙️ Processing voice message...")
    
    audio_path: Optional[Path] = None
    try:
        # Download the voice message
        audio_path = await download_voice_message(message)
        if not audio_path:
            await processing_msg.edit_text("❌ Failed to download voice message.")
            return
        
        # Transcribe using Whisper
        transcription = await transcribe_audio(audio_path)
        if not transcription:
            await processing_msg.edit_text("❌ Failed to transcribe voice message.")
            return
        
        logger.info(f"Transcribed voice message: {transcription}")
        
        # Delete the processing message
        await processing_msg.delete()
        
        # Process as regular input with voice flag
        await process_user_input(message, transcription, is_voice=True)
        
    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        try:
            await processing_msg.edit_text(f"❌ Error processing voice message: {e}")
        except:
            pass
    finally:
        # Always cleanup the audio file
        if audio_path:
            await cleanup_audio_file(audio_path)


@dp.message(F.text)
async def handle_text_message(message: Message):
    """Handle incoming text messages."""
    if message.from_user.id != ALLOWED_USER_ID:
        # Security: Ignore messages from unauthorized users
        return

    user_text = message.text
    if not user_text:
        return
        
    logger.info(f"Received: {user_text}")
    await process_user_input(message, user_text, is_voice=False)


@dp.message()
async def handle_unsupported_message(message: Message):
    """Handle unsupported message types (photos, stickers, etc.)."""
    if message.from_user.id != ALLOWED_USER_ID:
        return
    
    # Optionally respond to unsupported types
    logger.info(f"Received unsupported message type from user {message.from_user.id}")
    # Uncomment to notify user:
    # await message.reply("I can only process text and voice messages for now.")


async def shutdown(signal_type):
    """Graceful shutdown handler."""
    logger.info(f"Received signal {signal_type.name}, shutting down gracefully...")
    
    # Stop accepting new jobs
    scheduler.shutdown(wait=False)
    
    # Stop the dispatcher
    await dp.stop_polling()
    
    # Close the bot session
    await bot.session.close()
    
    # Cleanup temp audio directory
    try:
        for file in AUDIO_TEMP_DIR.glob("*"):
            file.unlink()
        logger.info("Cleaned up temporary audio files.")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp directory: {e}")
    
    logger.info("Shutdown complete.")


async def main():
    await memory.init_db()
    
    # Schedule weekly reset: Sunday at midnight
    scheduler.add_job(memory.execute_weekly_reset, 'cron', day_of_week='sun', hour=0, minute=0)
    scheduler.start()
    
    # Setup graceful shutdown handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))
    
    logger.info("Aika (Autonomy Enabled) Started")
    logger.info(f"Audio temp directory: {AUDIO_TEMP_DIR}")
    
    # Conditional startup message
    if SEND_STARTUP_MESSAGE:
        try:
            await bot.send_message(ALLOWED_USER_ID, "I'm awake. Don't break anything.")
        except Exception as e:
            logger.warning(f"Failed to send startup message: {e}")
    
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
