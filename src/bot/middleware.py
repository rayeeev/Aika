from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject


class AllowedUserMiddleware(BaseMiddleware):
    def __init__(self, allowed_user_ids: set[int], admin_user_ids: set[int]) -> None:
        self.allowed_user_ids = allowed_user_ids
        self.admin_user_ids = admin_user_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        user = getattr(event, "from_user", None)
        if user is None:
            return await handler(event, data)

        user_id = int(user.id)
        if user_id not in self.allowed_user_ids:
            # Ignore unauthorized messages entirely to avoid compute spend.
            return None

        data["is_admin"] = user_id in self.admin_user_ids
        return await handler(event, data)
