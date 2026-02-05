from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.config import Settings


class PathIsolationError(ValueError):
    pass


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class PathResolver:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.data_dir = cfg.data_dir.resolve()
        self.user_dir = cfg.user_dir.resolve()
        self.shared_dir = cfg.shared_dir.resolve()
        self.admin_dir = (self.data_dir / "admin").resolve()

    def ensure_base_layout(self) -> None:
        try:
            self.user_dir.mkdir(parents=True, exist_ok=True)
            self.shared_dir.mkdir(parents=True, exist_ok=True)
            (self.admin_dir / "error_reports").mkdir(parents=True, exist_ok=True)
            (self.admin_dir / "job_logs").mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to initialize data directories under {self.data_dir}. "
                "Set DATA_DIR, USER_DIR, and SHARED_DIR to writable paths."
            ) from exc

    def ensure_user_layout(self, user_id: int) -> None:
        root = self.user_root(user_id)
        (root / "files").mkdir(parents=True, exist_ok=True)
        (root / "cache").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)

    def user_root(self, user_id: int) -> Path:
        return (self.user_dir / str(user_id)).resolve()

    def resolve_user_path(self, user_id: int, path_rel: str) -> Path:
        return self._resolve_within(self.user_root(user_id), path_rel)

    def resolve_shared_path(self, path_rel: str) -> Path:
        return self._resolve_within(self.shared_dir, path_rel)

    def resolve_scoped_path(self, user_id: int, path_rel: str, location: str) -> Path:
        if location == "user":
            return self.resolve_user_path(user_id, path_rel)
        if location == "shared":
            return self.resolve_shared_path(path_rel)
        raise PathIsolationError(f"Unsupported location: {location}")

    def cache_file_path(self, user_id: int, original_name: str) -> Path:
        self.ensure_user_layout(user_id)
        safe_name = sanitize_filename(original_name or "attachment.bin")
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return (self.user_root(user_id) / "cache" / f"{timestamp}_{safe_name}").resolve()

    def user_disk_usage_bytes(self, user_id: int) -> int:
        root = self.user_root(user_id)
        if not root.exists():
            return 0
        total = 0
        for item in root.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    @staticmethod
    def _resolve_within(base: Path, path_rel: str) -> Path:
        candidate = Path(path_rel)
        if candidate.is_absolute():
            raise PathIsolationError("Absolute paths are forbidden")

        resolved = (base / candidate).resolve()
        try:
            resolved.relative_to(base.resolve())
        except ValueError as exc:
            raise PathIsolationError("Path traversal attempt blocked") from exc

        return resolved


def sanitize_filename(name: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", name).strip("._")
    return cleaned or "file"
