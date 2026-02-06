from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv


GEMINI_MODEL = "gemini-3-flash-preview"
GROQ_SUMMARY_MODEL = "qwen/qwen3-32b"
GROQ_TRIAGE_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_CHAT_MODEL = "moonshotai/kimi-k2-instruct-0905"


@dataclass
class Settings:
    telegram_bot_token: str
    allowed_user_ids: List[int]
    admin_user_ids: List[int]
    gemini_api_key: str
    gemini_api_key_fallbacks: List[str]
    telegram_user_aliases: Dict[int, str]
    groq_api_key: str
    data_dir: Path
    user_dir: Path
    shared_dir: Path
    cache_ttl_hours: int
    sandbox_image: str
    sandbox_timeout_seconds: int
    sandbox_memory: str
    sandbox_cpus: float
    log_level: str
    timezone: str
    sqlite_path: Path
    gemini_max_calls_per_message: int
    gemini_max_tool_calls_per_message: int
    gemini_max_retries: int
    gemini_retry_base_delay_seconds: float
    gemini_quota_cooldown_seconds: int
    gemini_daily_quota_cooldown_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        raw_data_dir = os.getenv("DATA_DIR", "/data").strip()
        use_local_fallback = raw_data_dir == "/data" and not _is_running_in_docker()
        if use_local_fallback:
            data_dir = (Path.cwd() / "data").resolve()
        else:
            data_dir = Path(raw_data_dir).resolve()

        raw_user_dir = os.getenv("USER_DIR", "").strip()
        raw_shared_dir = os.getenv("SHARED_DIR", "").strip()

        if use_local_fallback and raw_user_dir == "/data/users":
            raw_user_dir = ""
        if use_local_fallback and raw_shared_dir == "/data/shared":
            raw_shared_dir = ""

        user_dir = Path(raw_user_dir).resolve() if raw_user_dir else (data_dir / "users").resolve()
        shared_dir = Path(raw_shared_dir).resolve() if raw_shared_dir else (data_dir / "shared").resolve()

        primary_gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        fallback_keys = _parse_str_csv(os.getenv("GEMINI_API_KEY_FALLBACKS", ""))
        single_fallback = os.getenv("GEMINI_API_KEY_FALLBACK", "").strip()
        if single_fallback:
            fallback_keys = [single_fallback] + fallback_keys
        fallback_keys = [k for k in fallback_keys if k and k != primary_gemini_key]
        fallback_keys = _dedupe_preserve_order(fallback_keys)

        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            allowed_user_ids=_parse_int_csv(os.getenv("ALLOWED_USER_IDS", "")),
            admin_user_ids=_parse_int_csv(os.getenv("ADMIN_USER_IDS", "")),
            gemini_api_key=primary_gemini_key,
            gemini_api_key_fallbacks=fallback_keys,
            telegram_user_aliases=_parse_user_aliases(os.getenv("TELEGRAM_USER_ALIASES", "")),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            data_dir=data_dir,
            user_dir=user_dir,
            shared_dir=shared_dir,
            cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "72")),
            sandbox_image=os.getenv("SANDBOX_IMAGE", "agent-sandbox-runner:latest").strip(),
            sandbox_timeout_seconds=int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "30")),
            sandbox_memory=os.getenv("SANDBOX_MEMORY", "512m").strip(),
            sandbox_cpus=float(os.getenv("SANDBOX_CPUS", "0.5")),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            timezone=os.getenv("TIMEZONE", "America/Los_Angeles").strip(),
            sqlite_path=(data_dir / "aika.db").resolve(),
            gemini_max_calls_per_message=max(1, int(os.getenv("GEMINI_MAX_CALLS_PER_MESSAGE", "6"))),
            gemini_max_tool_calls_per_message=max(
                1,
                int(os.getenv("GEMINI_MAX_TOOL_CALLS_PER_MESSAGE", "10")),
            ),
            gemini_max_retries=max(1, int(os.getenv("GEMINI_MAX_RETRIES", "2"))),
            gemini_retry_base_delay_seconds=max(0.1, float(os.getenv("GEMINI_RETRY_BASE_DELAY_SECONDS", "1.0"))),
            gemini_quota_cooldown_seconds=max(1, int(os.getenv("GEMINI_QUOTA_COOLDOWN_SECONDS", "60"))),
            gemini_daily_quota_cooldown_seconds=max(
                60, int(os.getenv("GEMINI_DAILY_QUOTA_COOLDOWN_SECONDS", "3600"))
            ),
        )

    def validate_runtime(self) -> None:
        required = {
            "TELEGRAM_BOT_TOKEN": self.telegram_bot_token,
            "GROQ_API_KEY": self.groq_api_key,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        if not self.all_gemini_api_keys():
            raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY and/or GEMINI_API_KEY_FALLBACK(S).")

    def all_gemini_api_keys(self) -> List[str]:
        return _dedupe_preserve_order([self.gemini_api_key, *self.gemini_api_key_fallbacks])


def _parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_str_csv(value: str) -> List[str]:
    out: List[str] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(token)
    return out


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_user_aliases(value: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    raw = value.strip()
    if not raw:
        return mapping

    separators = [";", ",", "|"]
    parts = [raw]
    for sep in separators:
        if sep in raw:
            parts = [piece.strip() for piece in raw.split(sep)]
            break

    for part in parts:
        if not part:
            continue
        if ":" in part:
            left, right = part.split(":", 1)
        elif "=" in part:
            left, right = part.split("=", 1)
        else:
            continue
        left = left.strip()
        right = right.strip()
        if not left or not right:
            continue
        try:
            mapping[int(left)] = right
        except ValueError:
            continue

    return mapping


def _is_running_in_docker() -> bool:
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists():
            text = cgroup.read_text(encoding="utf-8", errors="ignore")
            return "docker" in text or "containerd" in text
    except Exception:
        return False
    return False


settings = Settings.from_env()
