from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from groq import AsyncGroq

from src.config import GROQ_FALLBACK_CHAT_MODEL, GROQ_SUMMARY_MODEL, GROQ_TRIAGE_MODEL


logger = logging.getLogger(__name__)


@dataclass
class SolvePlanCommand:
    cmd: str
    reason: str


@dataclass
class SolvePlan:
    analysis: str
    commands: list[SolvePlanCommand]
    expected_result: str
    raw_text: str


class GroqClient:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncGroq(api_key=api_key)

    async def summarize_exchange(self, user_text: str, assistant_text: str) -> str:
        system = (
            "You summarize one user/assistant exchange into exactly one short sentence. "
            "No bullet points, no preamble."
        )
        prompt = f"User: {user_text}\nAssistant: {assistant_text}"
        text = await self._chat(
            model=GROQ_SUMMARY_MODEL,
            system_prompt=system,
            user_prompt=prompt,
            max_tokens=120,
        )
        return _normalize_sentence_count(text, expected=1)

    async def summarize_daily(self, transcript_lines: Iterable[str]) -> str:
        system = (
            "Summarize the conversation in exactly two concise sentences. "
            "No lists, no headings."
        )
        prompt = "\n".join(transcript_lines)
        text = await self._chat(
            model=GROQ_SUMMARY_MODEL,
            system_prompt=system,
            user_prompt=prompt,
            max_tokens=220,
        )
        return _normalize_sentence_count(text, expected=2)

    async def summarize_major(self, daily_summaries: list[str], previous_major_memory: Optional[str]) -> str:
        system = (
            "Summarize into exactly four short sentences preserving key long-term preferences, goals, and ongoing tasks."
        )
        merged = "\n".join(daily_summaries)
        prev = previous_major_memory or "(none)"
        prompt = f"Previous major memory:\n{prev}\n\nDaily summaries:\n{merged}"
        text = await self._chat(
            model=GROQ_SUMMARY_MODEL,
            system_prompt=system,
            user_prompt=prompt,
            max_tokens=280,
        )
        return _normalize_sentence_count(text, expected=4)

    async def triage_error(self, stack_trace: str, snippets: str) -> str:
        system = (
            "You are a production incident triage assistant. Give short root cause analysis and concrete patch steps."
        )
        prompt = (
            "Stack trace:\n"
            f"{stack_trace}\n\n"
            "Relevant snippets:\n"
            f"{snippets}\n\n"
            "Return: root cause, why it happened, and a minimal patch plan."
        )
        return await self._chat(
            model=GROQ_TRIAGE_MODEL,
            system_prompt=system,
            user_prompt=prompt,
            max_tokens=800,
        )

    async def answer_with_fallback(self, prompt_context: str, user_text: str) -> str:
        system = (
            "You are Aika fallback assistant. Gemini is currently unavailable. "
            "Answer the user directly and concisely using provided context only. "
            "If a request requires unavailable tools, say so briefly."
        )
        prompt = (
            "Conversation context:\n"
            f"{prompt_context}\n\n"
            "Current user message:\n"
            f"{user_text}\n\n"
            "Return only the assistant reply text."
        )
        return await self._chat(
            model=GROQ_FALLBACK_CHAT_MODEL,
            system_prompt=system,
            user_prompt=prompt,
            max_tokens=900,
        )

    async def build_solve_plan(self, prompt: str, context: Optional[str] = None) -> SolvePlan:
        system = (
            "You are a cautious Linux incident solver. "
            "Return strict JSON only with keys: analysis (string), expected_result (string), "
            "commands (array of objects with cmd and reason). "
            "Commands must be concrete shell commands and ordered."
        )
        merged_context = (context or "").strip()
        user_prompt = (
            f"Task:\n{prompt.strip()}\n\n"
            f"Context:\n{merged_context if merged_context else '(none)'}\n\n"
            "Output JSON now."
        )
        text = await self._chat(
            model=GROQ_TRIAGE_MODEL,
            system_prompt=system,
            user_prompt=user_prompt,
            max_tokens=1200,
        )
        return _parse_solve_plan(text)

    async def _chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        retries = 4
        delay = 1.0
        for attempt in range(1, retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:
                if attempt == retries:
                    raise
                logger.warning("Groq call failed (attempt %s/%s): %s", attempt, retries, exc)
                await asyncio.sleep(delay)
                delay *= 2
        return ""


def _normalize_sentence_count(text: str, expected: int) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""

    parts = [p.strip() for p in cleaned.replace("!", ".").replace("?", ".").split(".") if p.strip()]
    if not parts:
        return cleaned

    if len(parts) >= expected:
        used = parts[:expected]
    else:
        used = parts + [parts[-1]] * (expected - len(parts))
    return ". ".join(used) + "."


def _parse_solve_plan(raw_text: str) -> SolvePlan:
    text = (raw_text or "").strip()
    data = _safe_parse_json(text)
    if not isinstance(data, dict):
        return SolvePlan(
            analysis=text or "No analysis returned.",
            commands=[],
            expected_result="",
            raw_text=text,
        )

    analysis = str(data.get("analysis", "") or "").strip()
    expected_result = str(data.get("expected_result", "") or "").strip()
    commands_raw = data.get("commands")

    commands: list[SolvePlanCommand] = []
    if isinstance(commands_raw, list):
        for item in commands_raw[:12]:
            if isinstance(item, str):
                cmd = item.strip()
                reason = ""
            elif isinstance(item, dict):
                cmd = str(item.get("cmd", "") or "").strip()
                reason = str(item.get("reason", "") or "").strip()
            else:
                continue

            if not cmd:
                continue
            commands.append(SolvePlanCommand(cmd=cmd, reason=reason))

    return SolvePlan(
        analysis=analysis or "No analysis returned.",
        commands=commands,
        expected_result=expected_result,
        raw_text=text,
    )


def _safe_parse_json(text: str) -> Optional[object]:
    candidate = text.strip()
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except Exception:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        block = fenced_match.group(1).strip()
        try:
            return json.loads(block)
        except Exception:
            pass

    left = candidate.find("{")
    right = candidate.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return None
    body = candidate[left : right + 1]
    try:
        return json.loads(body)
    except Exception:
        return None
