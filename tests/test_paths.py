from pathlib import Path

import pytest

from src.config import Settings
from src.utils.paths import PathIsolationError, PathResolver


def _settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        telegram_bot_token="",
        allowed_user_ids=[],
        admin_user_ids=[],
        gemini_api_key="",
        groq_api_key="",
        data_dir=data_dir,
        user_dir=data_dir / "users",
        shared_dir=data_dir / "shared",
        cache_ttl_hours=72,
        sandbox_image="agent-sandbox-runner:latest",
        sandbox_timeout_seconds=30,
        sandbox_memory="512m",
        sandbox_cpus=0.5,
        log_level="INFO",
        timezone="UTC",
        sqlite_path=data_dir / "aika.db",
    )


def test_user_path_stays_inside_user_root(tmp_path: Path) -> None:
    resolver = PathResolver(_settings(tmp_path))
    resolver.ensure_base_layout()
    resolver.ensure_user_layout(42)

    resolved = resolver.resolve_user_path(42, "files/note.txt")
    assert str(resolved).endswith("/users/42/files/note.txt")


def test_traversal_is_blocked(tmp_path: Path) -> None:
    resolver = PathResolver(_settings(tmp_path))
    resolver.ensure_base_layout()
    resolver.ensure_user_layout(42)

    with pytest.raises(PathIsolationError):
        resolver.resolve_user_path(42, "../43/files/secret.txt")


def test_absolute_path_is_blocked(tmp_path: Path) -> None:
    resolver = PathResolver(_settings(tmp_path))
    resolver.ensure_base_layout()

    with pytest.raises(PathIsolationError):
        resolver.resolve_shared_path("/etc/passwd")


def test_cache_name_sanitized(tmp_path: Path) -> None:
    resolver = PathResolver(_settings(tmp_path))
    resolver.ensure_base_layout()
    path = resolver.cache_file_path(7, "../../evil name?.txt")

    assert "/users/7/cache/" in str(path)
    assert "evil_name_.txt" in path.name
