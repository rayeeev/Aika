from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from src.memory.store import MemoryStore
from src.utils.paths import PathIsolationError, PathResolver


class FileService:
    def __init__(self, resolver: PathResolver, store: MemoryStore) -> None:
        self.resolver = resolver
        self.store = store

    async def file_write(
        self,
        telegram_user_id: int,
        path_rel: str,
        content: str,
        location: str = "user",
        allow_shared: bool = False,
    ) -> dict[str, Any]:
        if location == "shared" and not allow_shared:
            return {
                "status": "needs_confirmation",
                "message": "Shared storage write requested; ask user for explicit confirmation.",
            }

        try:
            path = self.resolver.resolve_scoped_path(telegram_user_id, path_rel, location)
        except PathIsolationError as exc:
            return {"status": "error", "message": str(exc)}

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        await self.store.add_file_metadata(telegram_user_id, location, str(path), "text")
        return {"status": "ok", "path": str(path), "bytes": len(content.encode("utf-8"))}

    async def file_read(
        self,
        telegram_user_id: int,
        path_rel: str,
        location: str = "user",
        allow_shared: bool = False,
    ) -> dict[str, Any]:
        if location == "shared" and not allow_shared:
            return {
                "status": "needs_confirmation",
                "message": "Shared storage read requested; ask user for explicit confirmation.",
            }

        try:
            path = self.resolver.resolve_scoped_path(telegram_user_id, path_rel, location)
        except PathIsolationError as exc:
            return {"status": "error", "message": str(exc)}

        if not path.exists() or not path.is_file():
            return {"status": "error", "message": "File does not exist"}

        data = path.read_text(encoding="utf-8", errors="replace")
        return {"status": "ok", "path": str(path), "content": data}

    async def file_list(
        self,
        telegram_user_id: int,
        path_rel_dir: str = "",
        location: str = "user",
        allow_shared: bool = False,
    ) -> dict[str, Any]:
        if location == "shared" and not allow_shared:
            return {
                "status": "needs_confirmation",
                "message": "Shared storage listing requested; ask user for explicit confirmation.",
            }

        try:
            root = self.resolver.resolve_scoped_path(telegram_user_id, path_rel_dir or ".", location)
        except PathIsolationError as exc:
            return {"status": "error", "message": str(exc)}

        if not root.exists() or not root.is_dir():
            return {"status": "error", "message": "Directory does not exist"}

        entries: list[dict[str, Any]] = []
        for item in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            entries.append(
                {
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
        return {"status": "ok", "path": str(root), "entries": entries}

    async def file_move(
        self,
        telegram_user_id: int,
        src_rel: str,
        dst_rel: str,
        location: str = "user",
    ) -> dict[str, Any]:
        if location != "user":
            return {"status": "error", "message": "file_move only supports user scope"}

        try:
            src = self.resolver.resolve_user_path(telegram_user_id, src_rel)
            dst = self.resolver.resolve_user_path(telegram_user_id, dst_rel)
        except PathIsolationError as exc:
            return {"status": "error", "message": str(exc)}

        if not src.exists():
            return {"status": "error", "message": "Source path does not exist"}

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        await self.store.mark_file_deleted(telegram_user_id, str(src))
        await self.store.add_file_metadata(telegram_user_id, "user", str(dst), "moved")
        return {"status": "ok", "src": str(src), "dst": str(dst)}

    async def copy_user_to_shared(
        self,
        telegram_user_id: int,
        src_rel_user: str,
        dst_rel_shared: str,
    ) -> dict[str, Any]:
        try:
            src = self.resolver.resolve_user_path(telegram_user_id, src_rel_user)
            dst = self.resolver.resolve_shared_path(dst_rel_shared)
        except PathIsolationError as exc:
            return {"status": "error", "message": str(exc)}

        if not src.exists():
            return {"status": "error", "message": "Source not found"}

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        await self.store.add_file_metadata(telegram_user_id, "shared", str(dst), "shared_put")
        return {"status": "ok", "dst": str(dst)}

    async def copy_shared_to_user(
        self,
        telegram_user_id: int,
        src_rel_shared: str,
        dst_rel_user: str,
    ) -> dict[str, Any]:
        try:
            src = self.resolver.resolve_shared_path(src_rel_shared)
            dst = self.resolver.resolve_user_path(telegram_user_id, dst_rel_user)
        except PathIsolationError as exc:
            return {"status": "error", "message": str(exc)}

        if not src.exists():
            return {"status": "error", "message": "Shared source not found"}

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

        await self.store.add_file_metadata(telegram_user_id, "user", str(dst), "shared_get")
        return {"status": "ok", "dst": str(dst)}

    async def list_recent_user_files(self, telegram_user_id: int, limit: int = 20) -> list[dict[str, Any]]:
        rows = await self.store.recent_files(telegram_user_id, limit=limit)
        out: list[dict[str, Any]] = []
        for row in rows:
            path = Path(row.file_path)
            exists = path.exists()
            out.append(
                {
                    "scope": row.scope,
                    "path": row.file_path,
                    "kind": row.kind,
                    "status": row.status,
                    "exists": exists,
                }
            )
        return out
