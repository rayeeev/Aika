from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dateutil import parser as date_parser


_IN_PATTERN = re.compile(
    r"^in\s+(\d+)\s*(m|h|d|mins?|minutes?|hrs?|hours?|days?)$",
    re.IGNORECASE,
)


def parse_reminder_time(raw: str, timezone_name: str) -> datetime:
    value = raw.strip()
    now_local = datetime.now(ZoneInfo(timezone_name))

    in_match = _IN_PATTERN.match(value)
    if in_match:
        amount = int(in_match.group(1))
        unit = in_match.group(2).lower()
        if unit in {"m", "min", "mins", "minute", "minutes"}:
            delta = timedelta(minutes=amount)
        elif unit in {"h", "hr", "hrs", "hour", "hours"}:
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)
        return (now_local + delta).astimezone(timezone.utc)

    if value.lower().startswith("tomorrow"):
        tail = value[len("tomorrow") :].strip() or "9:00am"
        parsed_time = date_parser.parse(tail, fuzzy=True)
        dt = now_local + timedelta(days=1)
        dt = dt.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=0,
            microsecond=0,
        )
        return dt.astimezone(timezone.utc)

    parsed = date_parser.parse(value, fuzzy=True)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))

    as_utc = parsed.astimezone(timezone.utc)
    if as_utc <= datetime.now(timezone.utc):
        raise ValueError("Reminder time must be in the future")
    return as_utc


def schedule_one_off_job(
    scheduler: AsyncIOScheduler,
    reminder_id: int,
    run_at_utc: datetime,
    callback,
) -> None:
    scheduler.add_job(
        callback,
        trigger="date",
        run_date=run_at_utc,
        args=[reminder_id],
        id=f"reminder-{reminder_id}",
        replace_existing=True,
        misfire_grace_time=300,
    )
