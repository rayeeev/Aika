from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any, Iterable, Optional, Sequence

from google import genai
from google.genai import types

from src.config import GEMINI_MODEL


logger = logging.getLogger(__name__)


class GeminiQuotaExceededError(RuntimeError):
    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[int] = None,
        details: str = "",
        daily_quota_hit: bool = False,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.details = details
        self.daily_quota_hit = daily_quota_hit


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class GeminiToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class GeminiTurnResult:
    text: str
    tool_calls: list[GeminiToolCall]


@dataclass
class _ApiKeyState:
    api_key: str
    client: genai.Client
    cooldown_until: Optional[datetime] = None


class GeminiClient:
    def __init__(
        self,
        api_keys: Sequence[str] | str,
        model: str = GEMINI_MODEL,
        max_retries: int = 2,
        base_retry_delay_seconds: float = 1.0,
        min_quota_cooldown_seconds: int = 60,
        daily_quota_cooldown_seconds: int = 3600,
    ) -> None:
        if isinstance(api_keys, str):
            keys = [api_keys]
        else:
            keys = list(api_keys)

        keys = [key.strip() for key in keys if key and key.strip()]
        keys = _dedupe_preserve_order(keys)
        if not keys:
            raise ValueError("At least one Gemini API key is required")

        self.model = model
        self.max_retries = max(1, int(max_retries))
        self.base_retry_delay_seconds = max(0.1, float(base_retry_delay_seconds))
        self.min_quota_cooldown_seconds = max(1, int(min_quota_cooldown_seconds))
        self.daily_quota_cooldown_seconds = max(60, int(daily_quota_cooldown_seconds))

        self._states = [_ApiKeyState(api_key=key, client=genai.Client(api_key=key)) for key in keys]
        self._active_state_index = 0
        self._state_lock = asyncio.Lock()

    async def generate_turn(
        self,
        system_instruction: str,
        contents: list[Any],
        tools: Iterable[ToolSpec],
    ) -> GeminiTurnResult:
        tool_defs = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in tools
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=tool_defs)],
            temperature=0.2,
        )

        response = await self._generate_with_retry(contents=contents, config=config)

        calls: list[GeminiToolCall] = []
        text_parts: list[str] = []
        candidates = getattr(response, "candidates", None) or []

        raw_calls = getattr(response, "function_calls", None)
        if raw_calls:
            for fc in raw_calls:
                args = dict(getattr(fc, "args", {}) or {})
                calls.append(GeminiToolCall(name=fc.name, args=args))

        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text_val = getattr(part, "text", None)
                if text_val:
                    text_parts.append(str(text_val).strip())

                if raw_calls:
                    continue

                # Fallback parser for SDK shape variants.
                fn = getattr(part, "function_call", None)
                if fn is None:
                    continue
                calls.append(
                    GeminiToolCall(
                        name=getattr(fn, "name", ""),
                        args=dict(getattr(fn, "args", {}) or {}),
                    )
                )

        text = " ".join(part for part in text_parts if part).strip()
        if not text:
            # Fallback when candidate text parts are absent.
            text = str(getattr(response, "output_text", "") or "").strip()

        return GeminiTurnResult(text=text, tool_calls=calls)

    async def _generate_with_retry(self, contents: list[Any], config: types.GenerateContentConfig) -> Any:
        transient_failures = 0
        total_attempts = 0
        max_attempt_budget = self.max_retries + (len(self._states) * 2)

        while total_attempts < max_attempt_budget:
            total_attempts += 1
            state_index = await self._select_available_state_index()
            state = self._states[state_index]
            key_tag = _mask_api_key(state.api_key)

            try:
                return await asyncio.to_thread(
                    state.client.models.generate_content,
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                error_text = str(exc)

                if _looks_like_quota_error(error_text):
                    retry_after = _extract_retry_after_seconds(error_text)
                    daily_quota_hit = _looks_like_daily_quota_limit(error_text)
                    cooldown = retry_after or self.min_quota_cooldown_seconds
                    if daily_quota_hit:
                        cooldown = max(cooldown, self.daily_quota_cooldown_seconds)

                    await self._apply_cooldown(state_index, cooldown)
                    switched = await self._switch_active_state(exclude_index=state_index)

                    logger.warning(
                        "Gemini key %s quota exhausted (daily=%s). cooldown=%ss switched=%s",
                        key_tag,
                        daily_quota_hit,
                        cooldown,
                        switched,
                    )

                    if switched:
                        continue

                    retry_wait = await self._min_retry_after_seconds()
                    raise GeminiQuotaExceededError(
                        message="All Gemini keys exhausted or cooling down",
                        retry_after_seconds=retry_wait,
                        details=error_text,
                        daily_quota_hit=daily_quota_hit,
                    ) from exc

                if _looks_like_key_auth_error(error_text):
                    await self._apply_cooldown(state_index, self.daily_quota_cooldown_seconds)
                    switched = await self._switch_active_state(exclude_index=state_index)
                    logger.error("Gemini key %s auth/permission error. switched=%s", key_tag, switched)
                    if switched:
                        continue
                    raise RuntimeError("All configured Gemini API keys are unavailable") from exc

                transient_failures += 1
                if transient_failures >= self.max_retries:
                    raise
                logger.warning(
                    "Gemini transient failure key=%s (transient %s/%s): %s",
                    key_tag,
                    transient_failures,
                    self.max_retries,
                    exc,
                )
                wait_seconds = self.base_retry_delay_seconds * (2 ** (transient_failures - 1))
                await asyncio.sleep(wait_seconds)

        raise RuntimeError("Gemini request exhausted retry budget")

    async def _select_available_state_index(self) -> int:
        now = datetime.now(timezone.utc)
        async with self._state_lock:
            self._clear_expired_cooldowns_locked(now)
            total = len(self._states)
            for offset in range(total):
                idx = (self._active_state_index + offset) % total
                if self._states[idx].cooldown_until is None:
                    self._active_state_index = idx
                    return idx

            retry_after = self._min_retry_after_locked(now)

        raise GeminiQuotaExceededError(
            message="All Gemini API keys are cooling down",
            retry_after_seconds=retry_after,
            details="All configured Gemini keys are in cooldown",
        )

    async def _apply_cooldown(self, index: int, seconds: int) -> None:
        now = datetime.now(timezone.utc)
        cooldown_until = now + timedelta(seconds=max(1, seconds))
        async with self._state_lock:
            state = self._states[index]
            if state.cooldown_until is None or state.cooldown_until < cooldown_until:
                state.cooldown_until = cooldown_until
            if self._active_state_index == index:
                self._promote_available_state_locked(now, exclude_index=index)

    async def _switch_active_state(self, exclude_index: Optional[int] = None) -> bool:
        now = datetime.now(timezone.utc)
        async with self._state_lock:
            self._clear_expired_cooldowns_locked(now)
            return self._promote_available_state_locked(now, exclude_index=exclude_index)

    async def _min_retry_after_seconds(self) -> int:
        now = datetime.now(timezone.utc)
        async with self._state_lock:
            self._clear_expired_cooldowns_locked(now)
            return self._min_retry_after_locked(now)

    def _clear_expired_cooldowns_locked(self, now: datetime) -> None:
        for state in self._states:
            if state.cooldown_until is not None and state.cooldown_until <= now:
                state.cooldown_until = None

    def _promote_available_state_locked(self, now: datetime, exclude_index: Optional[int] = None) -> bool:
        total = len(self._states)
        for offset in range(1, total + 1):
            idx = (self._active_state_index + offset) % total
            if exclude_index is not None and idx == exclude_index:
                continue
            state = self._states[idx]
            if state.cooldown_until is None or state.cooldown_until <= now:
                self._active_state_index = idx
                return True
        return False

    def _min_retry_after_locked(self, now: datetime) -> int:
        deltas: list[int] = []
        for state in self._states:
            if state.cooldown_until is None:
                return 1
            delta = ceil((state.cooldown_until - now).total_seconds())
            deltas.append(max(1, delta))
        return min(deltas) if deltas else 1


def build_attachment_parts(attachments: list[tuple[bytes, str]]) -> list[Any]:
    parts: list[Any] = []
    for payload, mime_type in attachments:
        try:
            parts.append(types.Part.from_bytes(data=payload, mime_type=mime_type))
        except Exception:
            # Model/API combination might not support bytes for this account/version.
            logger.warning("Failed to build Gemini attachment part for mime_type=%s", mime_type)
            continue
    return parts


def build_user_contents(prompt_text: str, attachment_parts: Optional[list[Any]] = None) -> list[Any]:
    parts: list[Any] = [types.Part.from_text(text=prompt_text)]
    if attachment_parts:
        parts.extend(attachment_parts)
    return [types.Content(role="user", parts=parts)]


def _looks_like_quota_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return "resource_exhausted" in lowered or "quota exceeded" in lowered


def _looks_like_daily_quota_limit(error_text: str) -> bool:
    return (
        "generaterequestsperdayperprojectpermodel" in error_text.lower()
        or "generate_content_free_tier_requests" in error_text.lower()
    )


def _looks_like_key_auth_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return (
        "api key not valid" in lowered
        or "invalid api key" in lowered
        or "permission_denied" in lowered
        or "forbidden" in lowered
        or "401" in lowered
        or "403" in lowered
    )


def _extract_retry_after_seconds(error_text: str) -> Optional[int]:
    patterns = [
        r"retryDelay['\"]?:\s*['\"](\d+)s['\"]",
        r"Please retry in ([0-9]+(?:\.[0-9]+)?)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return max(1, int(float(match.group(1))))
        except Exception:
            continue
    return None


def _mask_api_key(api_key: str) -> str:
    cleaned = api_key.strip()
    if len(cleaned) <= 4:
        return "***"
    return f"***{cleaned[-4:]}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
