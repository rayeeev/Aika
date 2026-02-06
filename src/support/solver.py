from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from src.llm.groq_client import GroqClient, SolvePlanCommand
from src.tools.exec import run_root_command


logger = logging.getLogger(__name__)

_CALLBACK_PREFIX = "solve"
_MAX_COMMANDS_PER_SESSION = 8


@dataclass
class SolveExecution:
    index: int
    cmd: str
    reason: str
    approved: bool
    exit_code: Optional[int]
    timed_out: bool
    stdout: str
    stderr: str


@dataclass
class SolveSession:
    session_id: str
    requester_user_id: int
    chat_id: int
    prompt: str
    source: str
    report_path: Optional[str]
    plan_analysis: str
    expected_result: str
    commands: list[SolvePlanCommand]
    next_index: int = 0
    executions: list[SolveExecution] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdminSolveService:
    def __init__(
        self,
        bot: Bot,
        groq: GroqClient,
        admin_user_ids: list[int],
        command_timeout_seconds: int = 120,
    ) -> None:
        self.bot = bot
        self.groq = groq
        self.admin_user_ids = sorted(set(int(x) for x in admin_user_ids))
        self.command_timeout_seconds = max(5, int(command_timeout_seconds))

        self._sessions: dict[str, SolveSession] = {}
        self._lock = asyncio.Lock()

    async def start_manual_solve(self, requester_user_id: int, chat_id: int, prompt: str) -> None:
        if requester_user_id not in self.admin_user_ids:
            await self.bot.send_message(chat_id=chat_id, text="Admin only.")
            return
        await self._start_session(
            requester_user_id=requester_user_id,
            chat_id=chat_id,
            prompt=prompt,
            source="manual",
            report_path=None,
        )

    async def handle_auto_problem(
        self,
        report_path: Path,
        context: str,
        stack_trace: str,
        triage_text: str,
    ) -> None:
        if not self.admin_user_ids:
            return

        target_admin = self.admin_user_ids[0]
        stack_excerpt = _cap_text(stack_trace, 1800)
        triage_excerpt = _cap_text(triage_text, 1800)
        prompt = (
            "Investigate and fix this runtime issue in the current container filesystem.\n\n"
            f"Report: {report_path}\n"
            f"Context: {context}\n\n"
            f"Stack excerpt:\n{stack_excerpt}\n\n"
            f"Triage excerpt:\n{triage_excerpt}\n"
        )

        await self.bot.send_message(
            chat_id=target_admin,
            text=(
                f"Auto-solve requested for report `{report_path.name}`. "
                "A plan will be generated and each command will need your Yes/No approval."
            ),
        )
        await self._start_session(
            requester_user_id=target_admin,
            chat_id=target_admin,
            prompt=prompt,
            source="auto",
            report_path=str(report_path),
        )

    async def handle_callback(self, callback_query: CallbackQuery) -> bool:
        data = (callback_query.data or "").strip()
        parsed = _parse_callback_data(data)
        if parsed is None:
            return False

        decision, session_id, index = parsed
        user = callback_query.from_user
        if user is None:
            await callback_query.answer("Unauthorized", show_alert=True)
            return True
        user_id = int(user.id)

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                await callback_query.answer("Session expired.", show_alert=True)
                return True
            if user_id != session.requester_user_id:
                await callback_query.answer("Only the requesting admin can approve.", show_alert=True)
                return True
            if index != session.next_index:
                await callback_query.answer("This approval is stale.", show_alert=True)
                return True

            command = session.commands[index]
            session.next_index += 1

        if decision == "y":
            await callback_query.answer("Approved. Running command.")
            await self.bot.send_message(
                chat_id=session.chat_id,
                text=f"Running command {index + 1}/{len(session.commands)}:\n{command.cmd}",
            )
            try:
                result = await run_root_command(command.cmd, timeout_seconds=self.command_timeout_seconds)
            except Exception as exc:
                logger.exception("Root command execution failed for solve session %s", session_id)
                result = {
                    "exit_code": 1,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": str(exc),
                }
            execution = SolveExecution(
                index=index,
                cmd=command.cmd,
                reason=command.reason,
                approved=True,
                exit_code=int(result.get("exit_code", 1)),
                timed_out=bool(result.get("timed_out", False)),
                stdout=str(result.get("stdout", "")),
                stderr=str(result.get("stderr", "")),
            )
            await self._append_execution(session_id, execution)
            await self._send_chunked(
                chat_id=session.chat_id,
                text=_format_execution_result(execution),
            )
        else:
            await callback_query.answer("Skipped.")
            execution = SolveExecution(
                index=index,
                cmd=command.cmd,
                reason=command.reason,
                approved=False,
                exit_code=None,
                timed_out=False,
                stdout="",
                stderr="Skipped by admin.",
            )
            await self._append_execution(session_id, execution)
            await self.bot.send_message(
                chat_id=session.chat_id,
                text=f"Skipped command {index + 1}/{len(session.commands)}.",
            )

        await self._continue_or_finish(session_id)
        return True

    async def _start_session(
        self,
        requester_user_id: int,
        chat_id: int,
        prompt: str,
        source: str,
        report_path: Optional[str],
    ) -> None:
        text = prompt.strip()
        if not text:
            await self.bot.send_message(chat_id=chat_id, text="Usage: /solve <prompt>")
            return

        await self.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:
            plan = await self.groq.build_solve_plan(
                prompt=text,
                context=(
                    "Environment: Linux container. "
                    "Commands will run from filesystem root (/). "
                    "Only propose concrete shell commands that can be executed safely."
                ),
            )
        except Exception as exc:
            logger.exception("Failed to build solve plan")
            await self.bot.send_message(chat_id=chat_id, text=f"Solve planning failed: {exc}")
            return

        commands = [cmd for cmd in plan.commands if cmd.cmd.strip()][:_MAX_COMMANDS_PER_SESSION]
        if not commands:
            body = (
                "Solver did not propose runnable shell commands.\n\n"
                f"Analysis:\n{plan.analysis}\n\n"
                f"Expected result:\n{plan.expected_result or '(not provided)'}"
            )
            await self._send_chunked(chat_id=chat_id, text=body)
            return

        session_id = secrets.token_hex(4)
        session = SolveSession(
            session_id=session_id,
            requester_user_id=requester_user_id,
            chat_id=chat_id,
            prompt=text,
            source=source,
            report_path=report_path,
            plan_analysis=plan.analysis,
            expected_result=plan.expected_result,
            commands=commands,
        )

        async with self._lock:
            self._sessions[session_id] = session

        intro_lines = [
            f"Solve session `{session_id}` started ({source}).",
            f"Planned commands: {len(commands)}",
            f"Analysis: {plan.analysis}",
        ]
        if plan.expected_result:
            intro_lines.append(f"Expected result: {plan.expected_result}")
        if report_path:
            intro_lines.append(f"Report: {report_path}")
        await self._send_chunked(chat_id=chat_id, text="\n".join(intro_lines))

        await self._send_approval_prompt(session_id)

    async def _append_execution(self, session_id: str, execution: SolveExecution) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.executions.append(execution)

    async def _continue_or_finish(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            done = session.next_index >= len(session.commands)

        if done:
            await self._finish_session(session_id)
            return
        await self._send_approval_prompt(session_id)

    async def _send_approval_prompt(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            if session.next_index >= len(session.commands):
                return
            index = session.next_index
            command = session.commands[index]

        callback_yes = _build_callback_data("y", session_id, index)
        callback_no = _build_callback_data("n", session_id, index)
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="Yes", callback_data=callback_yes),
                    InlineKeyboardButton(text="No", callback_data=callback_no),
                ]
            ]
        )

        lines = [
            f"Approve command {index + 1}/{len(session.commands)} for session `{session_id}`?",
            f"Reason: {command.reason or '(not provided)'}",
            "",
            command.cmd,
        ]
        await self.bot.send_message(chat_id=session.chat_id, text="\n".join(lines), reply_markup=keyboard)

    async def _finish_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return

        approved_count = sum(1 for item in session.executions if item.approved)
        skipped_count = sum(1 for item in session.executions if not item.approved)
        success_count = sum(
            1
            for item in session.executions
            if item.approved and item.exit_code == 0 and not item.timed_out
        )

        lines = [
            f"Solve session `{session.session_id}` finished.",
            f"Source: {session.source}",
            f"Approved: {approved_count}, Skipped: {skipped_count}, Successful: {success_count}",
            f"Prompt: {session.prompt}",
            f"Plan analysis: {session.plan_analysis}",
            f"Expected result: {session.expected_result or '(not provided)'}",
        ]
        if session.report_path:
            lines.append(f"Report: {session.report_path}")

        lines.append("")
        lines.append("Execution summary:")
        if not session.executions:
            lines.append("(no commands were processed)")
        else:
            for item in session.executions:
                if not item.approved:
                    lines.append(f"{item.index + 1}. SKIPPED: {item.cmd}")
                    continue
                lines.append(
                    f"{item.index + 1}. exit={item.exit_code} timed_out={item.timed_out}: {item.cmd}"
                )

        await self._send_chunked(chat_id=session.chat_id, text="\n".join(lines))

    async def _send_chunked(self, chat_id: int, text: str, chunk_size: int = 3600) -> None:
        body = text.strip() or "(empty)"
        if len(body) <= chunk_size:
            await self.bot.send_message(chat_id=chat_id, text=body)
            return
        for index in range(0, len(body), chunk_size):
            await self.bot.send_message(chat_id=chat_id, text=body[index : index + chunk_size])


def _format_execution_result(execution: SolveExecution) -> str:
    stdout = _cap_text(execution.stdout, 2000)
    stderr = _cap_text(execution.stderr, 2000)
    return (
        f"Command {execution.index + 1} result:\n"
        f"exit={execution.exit_code} timed_out={execution.timed_out}\n\n"
        f"STDOUT:\n{stdout or '(empty)'}\n\n"
        f"STDERR:\n{stderr or '(empty)'}"
    )


def _cap_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>"


def _build_callback_data(decision: str, session_id: str, index: int) -> str:
    return f"{_CALLBACK_PREFIX}:{decision}:{session_id}:{index}"


def _parse_callback_data(raw: str) -> Optional[tuple[str, str, int]]:
    parts = raw.split(":")
    if len(parts) != 4:
        return None
    if parts[0] != _CALLBACK_PREFIX:
        return None
    decision = parts[1]
    if decision not in {"y", "n"}:
        return None
    session_id = parts[2].strip()
    if not session_id:
        return None
    try:
        index = int(parts[3])
    except ValueError:
        return None
    if index < 0:
        return None
    return decision, session_id, index
