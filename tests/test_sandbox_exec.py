from pathlib import Path

import pytest

from src.config import Settings
from src.tools.exec import SandboxExecutor
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


def test_container_spec_enforces_isolation(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    resolver = PathResolver(cfg)
    resolver.ensure_base_layout()
    executor = SandboxExecutor(cfg, resolver)

    spec = executor.build_container_spec(
        telegram_user_id=1001,
        cmd="ls -la",
        workdir_rel="files",
        timeout_seconds=10,
    )

    assert spec["image"] == "agent-sandbox-runner:latest"
    assert spec["network_mode"] == "none"
    assert spec["mem_limit"] == "512m"
    assert spec["nano_cpus"] == 500_000_000
    assert spec["pids_limit"] == 128
    assert spec["cap_drop"] == ["ALL"]
    assert spec["security_opt"] == ["no-new-privileges:true"]
    assert spec["working_dir"] == "/work/files"

    mounted = list(spec["volumes"].keys())
    assert len(mounted) == 1
    assert mounted[0].endswith("/users/1001")


def test_workdir_traversal_blocked(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    resolver = PathResolver(cfg)
    resolver.ensure_base_layout()
    executor = SandboxExecutor(cfg, resolver)

    with pytest.raises(PathIsolationError):
        executor.build_container_spec(
            telegram_user_id=1001,
            cmd="pwd",
            workdir_rel="../1002",
            timeout_seconds=10,
        )
