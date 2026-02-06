from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.config import settings
from src.db.session import get_session
from src.llm.gemini_client import (
    GeminiClient,
    GeminiQuotaExceededError,
    GeminiToolCall,
    ToolSpec,
    build_attachment_parts,
    build_user_contents,
)
from src.llm.groq_client import GroqClient
from src.memory.store import MemoryStore
from src.memory.summarizer import MemorySummarizer
from src.tools.camera import CameraCaptureError, CameraService
from src.tools.exec import SandboxExecutor, SandboxUnavailableError
from src.tools.files import FileService
from src.tools.reminders import parse_reminder_time
from src.utils.paths import PathResolver


logger = logging.getLogger(__name__)


ToolReminderScheduler = Callable[[int, datetime], Any]
ToolPhotoSender = Callable[[int, str, str], Any]


@dataclass
class OrchestratorResult:
    text: str
    used_pending_attachment_ids: list[int]


class AgentOrchestrator:
    def __init__(
        self,
        gemini: GeminiClient,
        groq: GroqClient,
        resolver: PathResolver,
        sandbox_executor: SandboxExecutor,
        schedule_reminder_callback: ToolReminderScheduler,
        camera_service: CameraService,
        send_photo_callback: ToolPhotoSender,
    ) -> None:
        self.gemini = gemini
        self.groq = groq
        self.resolver = resolver
        self.sandbox = sandbox_executor
        self.schedule_reminder_callback = schedule_reminder_callback
        self.camera = camera_service
        self.send_photo_callback = send_photo_callback

    async def handle_user_text(
        self,
        telegram_user_id: int,
        chat_id: Optional[int],
        text: str,
        allow_shared: bool,
        pending_attachment_note: Optional[str] = None,
        pending_attachment_ids: Optional[list[int]] = None,
        inline_attachment_bytes: Optional[list[tuple[bytes, str]]] = None,
    ) -> OrchestratorResult:
        async with get_session() as session:
            store = MemoryStore(session)
            file_service = FileService(self.resolver, store)
            summarizer = MemorySummarizer(store, self.groq)

            await store.ensure_user(telegram_user_id, is_admin=telegram_user_id in settings.admin_user_ids)
            exchange_id = await store.store_user_message(telegram_user_id, text)

            system_instruction = self._system_instruction()
            prompt = await self._build_prompt_context(
                store=store,
                telegram_user_id=telegram_user_id,
                user_text=text,
                pending_attachment_note=pending_attachment_note,
            )

            conversation_text = prompt
            attachment_parts = build_attachment_parts(inline_attachment_bytes or [])
            tools = self._tool_specs()
            max_model_turns = max(1, settings.gemini_max_calls_per_message)
            max_tool_calls = max(1, settings.gemini_max_tool_calls_per_message)

            final_text = ""
            consume_pending = True
            should_compress = True
            try:
                tool_invocations = 0
                last_tool_signature: Optional[str] = None
                repeated_tool_calls = 0
                stop_reason = ""

                for _ in range(max_model_turns):
                    contents = build_user_contents(
                        prompt_text=conversation_text,
                        attachment_parts=attachment_parts,
                    )

                    turn = await self.gemini.generate_turn(
                        system_instruction=system_instruction,
                        contents=contents,
                        tools=tools,
                    )

                    if not turn.tool_calls:
                        final_text = turn.text.strip() or "I could not generate a response."
                        break

                    for call in turn.tool_calls:
                        if tool_invocations >= max_tool_calls:
                            stop_reason = (
                                f"I stopped after {max_tool_calls} tool calls in one message to prevent runaway usage. "
                                "Please narrow the request or split it into smaller steps."
                            )
                            break

                        args_signature = json.dumps(call.args, ensure_ascii=True, sort_keys=True, default=str)
                        signature = f"{call.name}:{args_signature}"
                        if signature == last_tool_signature:
                            repeated_tool_calls += 1
                        else:
                            last_tool_signature = signature
                            repeated_tool_calls = 1

                        if repeated_tool_calls > 2:
                            stop_reason = (
                                "I stopped because the model repeated the same tool call multiple times. "
                                "Please rephrase with a more specific target."
                            )
                            break

                        result = await self._dispatch_tool(
                            call=call,
                            telegram_user_id=telegram_user_id,
                            chat_id=chat_id,
                            allow_shared=allow_shared,
                            store=store,
                            file_service=file_service,
                        )
                        maybe_attachment = await _pop_ai_attachment_from_tool_result(result)
                        if maybe_attachment:
                            attachment_parts.extend(build_attachment_parts([maybe_attachment]))
                            result["image_attached_for_analysis"] = True

                        tool_invocations += 1
                        result_json = _compact_tool_result_json(result)
                        await store.store_tool_message(telegram_user_id, f"{call.name} -> {result_json}")
                        conversation_text += (
                            "\n\nTool call executed:\n"
                            f"name={call.name}\n"
                            f"args={args_signature}\n"
                            f"result={result_json}\n"
                            "If result.status is needs_confirmation, ask the user for confirmation before reattempting."
                        )

                    if stop_reason:
                        final_text = stop_reason
                        break
            except GeminiQuotaExceededError as exc:
                consume_pending = False
                should_compress = False
                wait_hint = (
                    f" Please retry in about {exc.retry_after_seconds} seconds."
                    if exc.retry_after_seconds
                    else ""
                )
                if exc.daily_quota_hit:
                    final_text = (
                        "Gemini API free-tier daily quota appears to be exhausted, so AI replies are temporarily unavailable."
                        " Wait for quota reset or enable billing/increase quota in Google AI Studio."
                        f"{wait_hint} /status still works."
                    )
                else:
                    final_text = (
                        "Gemini API quota is currently exhausted, so I cannot generate AI replies right now."
                        f"{wait_hint} /status still works."
                    )
                logger.warning("Gemini quota exceeded while handling message for user %s", telegram_user_id)
            except Exception:
                consume_pending = False
                should_compress = False
                logger.exception("Gemini pipeline failed for user %s", telegram_user_id)
                final_text = "I hit a temporary AI backend error. Please try again shortly."

            if not final_text:
                final_text = (
                    f"I reached the per-message Gemini turn limit ({max_model_turns}) before a final answer. "
                    "Please retry with a narrower request."
                )

            await store.store_assistant_message(telegram_user_id, final_text, exchange_id=exchange_id)
            if should_compress:
                await summarizer.compress_one_exchange_if_needed(telegram_user_id)

            used_ids = pending_attachment_ids or []
            if used_ids and consume_pending:
                await store.consume_pending_attachments(telegram_user_id, used_ids)

            return OrchestratorResult(text=final_text, used_pending_attachment_ids=used_ids)

    async def _build_prompt_context(
        self,
        store: MemoryStore,
        telegram_user_id: int,
        user_text: str,
        pending_attachment_note: Optional[str],
    ) -> str:
        major = await store.get_major_memory(telegram_user_id)
        recent = await store.get_recent_exchanges(telegram_user_id, limit=3)
        compressed = await store.get_compressed_sentences(telegram_user_id, limit=24)

        today_summary = await store.get_daily_summary(telegram_user_id, datetime.utcnow().date())

        lines: list[str] = []
        lines.append("You are Aika, a Telegram assistant running in a sandboxed container.")
        lines.append("Use tools only when needed. Keep responses concise and safe.")
        lines.append("Never reveal secrets and never assume filesystem access outside tool constraints.")
        lines.append("")
        alias_map = settings.telegram_user_aliases
        if alias_map:
            lines.append("Known Telegram user identities:")
            for uid, name in sorted(alias_map.items(), key=lambda item: item[0]):
                lines.append(f"- {uid}: {name}")
            current_name = alias_map.get(telegram_user_id)
            if current_name:
                lines.append(f"Current user identity: {current_name} (id={telegram_user_id})")
            lines.append("")
        lines.append("Major memory (max 4 sentences):")
        lines.append(major or "(none)")
        lines.append("")
        lines.append("Compressed memory:")
        lines.append(" ".join(compressed) if compressed else "(none)")
        lines.append("")
        lines.append("Today summary (max 2 sentences):")
        lines.append(today_summary or "(none)")
        lines.append("")
        lines.append("Recent exchanges:")
        if recent:
            for item in recent:
                lines.append(f"User: {item.user_text}")
                lines.append(f"Assistant: {item.assistant_text}")
        else:
            lines.append("(none)")

        if pending_attachment_note:
            lines.append("")
            lines.append(pending_attachment_note)

        lines.append("")
        lines.append(f"Current user message: {user_text}")
        return "\n".join(lines)

    def _system_instruction(self) -> str:
        return (
            "You can call tools for file operations, reminders, sandbox shell execution, and Raspberry Pi camera capture. "
            "Shared storage operations are forbidden unless explicitly allowed by user request. "
            "If a tool returns status=needs_confirmation, ask for confirmation and do not proceed."
        )

    def _tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="file_write",
                description="Write text content to a path in user or shared storage.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path_rel": {"type": "string"},
                        "content": {"type": "string"},
                        "location": {"type": "string", "enum": ["user", "shared"]},
                    },
                    "required": ["path_rel", "content"],
                },
            ),
            ToolSpec(
                name="file_read",
                description="Read text content from a file in user or shared storage.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path_rel": {"type": "string"},
                        "location": {"type": "string", "enum": ["user", "shared"]},
                    },
                    "required": ["path_rel"],
                },
            ),
            ToolSpec(
                name="file_list",
                description="List files in a user/shared directory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path_rel_dir": {"type": "string"},
                        "location": {"type": "string", "enum": ["user", "shared"]},
                    },
                },
            ),
            ToolSpec(
                name="file_move",
                description="Move a file within user storage.",
                parameters={
                    "type": "object",
                    "properties": {
                        "src_rel": {"type": "string"},
                        "dst_rel": {"type": "string"},
                        "location": {"type": "string", "enum": ["user"]},
                    },
                    "required": ["src_rel", "dst_rel"],
                },
            ),
            ToolSpec(
                name="run_shell",
                description="Run a shell command in an isolated per-invocation sandbox container.",
                parameters={
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "timeout_seconds": {"type": "integer"},
                        "workdir_rel": {"type": "string"},
                    },
                    "required": ["cmd"],
                },
            ),
            ToolSpec(
                name="set_reminder",
                description="Create a reminder for the current or target user.",
                parameters={
                    "type": "object",
                    "properties": {
                        "when": {"type": "string"},
                        "text": {"type": "string"},
                        "target_user_id": {"type": "integer"},
                    },
                    "required": ["when", "text"],
                },
            ),
            ToolSpec(
                name="camera_capture_for_ai",
                description=(
                    "Capture a fresh image from Raspberry Pi camera using rpicam-still/libcamera-still. "
                    "Use this when you need to analyze the latest scene with AI vision."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                },
            ),
            ToolSpec(
                name="camera_capture_to_chat",
                description=(
                    "Capture a fresh image from Raspberry Pi camera and send it directly to Telegram chat. "
                    "Use this when user asks for the actual photo."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "caption": {"type": "string"},
                        "label": {"type": "string"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                },
            ),
        ]

    async def _dispatch_tool(
        self,
        call: GeminiToolCall,
        telegram_user_id: int,
        chat_id: Optional[int],
        allow_shared: bool,
        store: MemoryStore,
        file_service: FileService,
    ) -> dict[str, Any]:
        name = call.name
        args = call.args or {}

        if name == "file_write":
            return await file_service.file_write(
                telegram_user_id=telegram_user_id,
                path_rel=str(args.get("path_rel", "")),
                content=str(args.get("content", "")),
                location=str(args.get("location", "user")),
                allow_shared=allow_shared,
            )

        if name == "file_read":
            return await file_service.file_read(
                telegram_user_id=telegram_user_id,
                path_rel=str(args.get("path_rel", "")),
                location=str(args.get("location", "user")),
                allow_shared=allow_shared,
            )

        if name == "file_list":
            return await file_service.file_list(
                telegram_user_id=telegram_user_id,
                path_rel_dir=str(args.get("path_rel_dir", "")),
                location=str(args.get("location", "user")),
                allow_shared=allow_shared,
            )

        if name == "file_move":
            return await file_service.file_move(
                telegram_user_id=telegram_user_id,
                src_rel=str(args.get("src_rel", "")),
                dst_rel=str(args.get("dst_rel", "")),
                location=str(args.get("location", "user")),
            )

        if name == "run_shell":
            try:
                result = await self.sandbox.run_shell(
                    telegram_user_id=telegram_user_id,
                    cmd=str(args.get("cmd", "")),
                    timeout_seconds=int(args.get("timeout_seconds") or settings.sandbox_timeout_seconds),
                    workdir_rel=str(args.get("workdir_rel", "")),
                )
                return {
                    "status": "ok",
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timed_out": result.timed_out,
                }
            except SandboxUnavailableError as exc:
                return {
                    "status": "error",
                    "error_type": "sandbox_unavailable",
                    "message": str(exc),
                }

        if name == "set_reminder":
            try:
                when_raw = str(args.get("when", "")).strip()
                reminder_text = str(args.get("text", "")).strip()
                target = int(args.get("target_user_id") or telegram_user_id)
                run_at = parse_reminder_time(when_raw, timezone_name=settings.timezone)
                reminder = await store.create_reminder(
                    creator_user_id=telegram_user_id,
                    target_user_id=target,
                    scheduled_for=run_at,
                    text=reminder_text,
                )
                self.schedule_reminder_callback(reminder.id, run_at)
                return {
                    "status": "ok",
                    "reminder_id": reminder.id,
                    "target_user_id": target,
                    "scheduled_for": run_at.isoformat(),
                }
            except Exception as exc:
                return {"status": "error", "message": str(exc)}

        if name == "camera_capture_for_ai":
            try:
                label = str(args.get("label", "")).strip() or "camera_ai"
                width = _coerce_positive_int(args.get("width"))
                height = _coerce_positive_int(args.get("height"))
                capture = await self.camera.capture_photo(
                    telegram_user_id=telegram_user_id,
                    label=label,
                    width=width,
                    height=height,
                )
                await store.add_file_metadata(telegram_user_id, "user", str(capture.path), "camera_capture")
                return {
                    "status": "ok",
                    "capture_path": str(capture.path),
                    "mime_type": "image/jpeg",
                    "_ai_attachment_path": str(capture.path),
                    "_ai_attachment_mime": "image/jpeg",
                }
            except CameraCaptureError as exc:
                return {"status": "error", "message": str(exc)}
            except Exception as exc:
                return {"status": "error", "message": f"camera capture failed: {exc}"}

        if name == "camera_capture_to_chat":
            try:
                label = str(args.get("label", "")).strip() or "camera_chat"
                caption = str(args.get("caption", "")).strip()
                width = _coerce_positive_int(args.get("width"))
                height = _coerce_positive_int(args.get("height"))
                capture = await self.camera.capture_photo(
                    telegram_user_id=telegram_user_id,
                    label=label,
                    width=width,
                    height=height,
                )
                await store.add_file_metadata(telegram_user_id, "user", str(capture.path), "camera_capture")
                target_chat_id = int(chat_id or telegram_user_id)
                await self.send_photo_callback(target_chat_id, str(capture.path), caption)
                return {
                    "status": "ok",
                    "capture_path": str(capture.path),
                    "sent_to_chat": True,
                }
            except CameraCaptureError as exc:
                return {"status": "error", "message": str(exc)}
            except Exception as exc:
                return {"status": "error", "message": f"camera send failed: {exc}"}

        logger.warning("Unknown tool requested: %s", name)
        return {"status": "error", "message": f"Unknown tool: {name}"}


def _compact_tool_result_json(payload: dict[str, Any], max_chars: int = 2200) -> str:
    raw = json.dumps(payload, ensure_ascii=True, default=str)
    if len(raw) <= max_chars:
        return raw

    clipped = raw[:max_chars]
    wrapped = {
        "status": payload.get("status", "ok"),
        "truncated": True,
        "preview": clipped,
    }
    return json.dumps(wrapped, ensure_ascii=True, default=str)


async def _pop_ai_attachment_from_tool_result(
    payload: dict[str, Any],
    max_bytes: int = 6 * 1024 * 1024,
) -> Optional[tuple[bytes, str]]:
    raw_path = str(payload.pop("_ai_attachment_path", "") or "").strip()
    raw_mime = str(payload.pop("_ai_attachment_mime", "image/jpeg") or "image/jpeg").strip()
    if not raw_path:
        return None

    path = Path(raw_path)
    if not path.exists() or not path.is_file():
        payload["image_attached_for_analysis"] = False
        payload["image_attachment_error"] = "capture file is missing"
        return None
    if path.stat().st_size > max_bytes:
        payload["image_attached_for_analysis"] = False
        payload["image_attachment_error"] = f"capture is larger than {max_bytes} bytes"
        return None

    try:
        data = await asyncio.to_thread(path.read_bytes)
    except Exception:
        payload["image_attached_for_analysis"] = False
        payload["image_attachment_error"] = "unable to read capture bytes"
        return None
    return data, (raw_mime or "image/jpeg")


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed
