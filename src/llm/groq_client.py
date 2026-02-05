from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Optional

from groq import AsyncGroq

from src.config import GROQ_SUMMARY_MODEL, GROQ_TRIAGE_MODEL


logger = logging.getLogger(__name__)


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
