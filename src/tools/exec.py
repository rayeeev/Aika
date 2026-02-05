from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import docker
from docker.errors import DockerException, NotFound

from src.config import Settings
from src.utils.paths import PathResolver


logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class SandboxUnavailableError(RuntimeError):
    pass


class SandboxExecutor:
    def __init__(self, cfg: Settings, resolver: PathResolver) -> None:
        self.cfg = cfg
        self.resolver = resolver
        self._docker_client: Optional[docker.DockerClient] = None

    @property
    def client(self) -> docker.DockerClient:
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
            except Exception as exc:
                raise SandboxUnavailableError(
                    "Sandbox Docker daemon is unavailable. "
                    "Start Docker (or run inside docker-compose with /var/run/docker.sock mounted)."
                ) from exc
        return self._docker_client

    def build_container_spec(
        self,
        telegram_user_id: int,
        cmd: str,
        workdir_rel: str = "",
        timeout_seconds: Optional[int] = None,
    ) -> dict[str, Any]:
        timeout = timeout_seconds or self.cfg.sandbox_timeout_seconds
        user_root = self.resolver.user_root(telegram_user_id)
        self.resolver.ensure_user_layout(telegram_user_id)

        workdir = "/work"
        if workdir_rel.strip():
            safe_path = self.resolver.resolve_user_path(telegram_user_id, workdir_rel)
            rel = safe_path.relative_to(user_root)
            workdir = f"/work/{rel.as_posix()}"

        return {
            "image": self.cfg.sandbox_image,
            "command": ["bash", "-lc", cmd],
            "detach": True,
            "network_mode": "none",
            "mem_limit": self.cfg.sandbox_memory,
            "nano_cpus": int(self.cfg.sandbox_cpus * 1_000_000_000),
            "pids_limit": 128,
            "security_opt": ["no-new-privileges:true"],
            "cap_drop": ["ALL"],
            "working_dir": workdir,
            "volumes": {str(user_root): {"bind": "/work", "mode": "rw"}},
            "environment": {"PYTHONUNBUFFERED": "1"},
            "timeout": timeout,
        }

    async def run_shell(
        self,
        telegram_user_id: int,
        cmd: str,
        timeout_seconds: Optional[int] = None,
        workdir_rel: str = "",
    ) -> SandboxResult:
        spec = self.build_container_spec(
            telegram_user_id=telegram_user_id,
            cmd=cmd,
            workdir_rel=workdir_rel,
            timeout_seconds=timeout_seconds,
        )
        try:
            return await asyncio.to_thread(self._run_sync, spec)
        except SandboxUnavailableError:
            raise
        except (DockerException, FileNotFoundError) as exc:
            raise SandboxUnavailableError(
                "Sandbox Docker daemon is unavailable. "
                "Start Docker (or run inside docker-compose with /var/run/docker.sock mounted)."
            ) from exc

    def _run_sync(self, spec: dict[str, Any]) -> SandboxResult:
        timeout = int(spec.pop("timeout"))
        container = self.client.containers.run(**spec)
        try:
            try:
                wait_result = container.wait(timeout=timeout)
                exit_code = int(wait_result.get("StatusCode", 1))
                timed_out = False
            except Exception:
                timed_out = True
                exit_code = 124
                container.kill()

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            return SandboxResult(
                exit_code=exit_code,
                stdout=_cap(stdout),
                stderr=_cap(stderr),
                timed_out=timed_out,
            )
        finally:
            try:
                container.remove(force=True)
            except NotFound:
                pass


async def run_admin_command(command: str, timeout_seconds: int = 20) -> dict[str, Any]:
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        return {
            "exit_code": process.returncode,
            "stdout": _cap(stdout.decode("utf-8", errors="replace")),
            "stderr": _cap(stderr.decode("utf-8", errors="replace")),
            "timed_out": False,
        }
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return {
            "exit_code": 124,
            "stdout": "",
            "stderr": "Command timed out",
            "timed_out": True,
        }


def _cap(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"
