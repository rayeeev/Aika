from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.paths import PathResolver, sanitize_filename


class CameraCaptureError(RuntimeError):
    pass


@dataclass
class CameraCaptureResult:
    path: Path
    command: str
    stdout: str
    stderr: str


class CameraService:
    def __init__(
        self,
        resolver: PathResolver,
        command: Optional[str] = None,
        capture_timeout_seconds: Optional[int] = None,
        capture_warmup_ms: Optional[int] = None,
    ) -> None:
        self.resolver = resolver
        self.command = (command or os.getenv("CAMERA_STILL_COMMAND", "rpicam-still")).strip() or "rpicam-still"
        self.capture_timeout_seconds = max(
            2,
            int(capture_timeout_seconds or int(os.getenv("CAMERA_CAPTURE_TIMEOUT_SECONDS", "20"))),
        )
        self.capture_warmup_ms = max(
            1,
            int(capture_warmup_ms or int(os.getenv("CAMERA_CAPTURE_WARMUP_MS", "1000"))),
        )

    async def capture_photo(
        self,
        telegram_user_id: int,
        label: str = "camera_capture",
        width: Optional[int] = None,
        height: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> CameraCaptureResult:
        self.resolver.ensure_user_layout(telegram_user_id)
        safe_label = sanitize_filename(label or "camera_capture")
        output_path = self.resolver.cache_file_path(telegram_user_id, f"{safe_label}.jpg")
        binary = _resolve_camera_binary(self.command)

        cmd = [
            binary,
            "-n",
            "--timeout",
            str(self.capture_warmup_ms),
            "-o",
            str(output_path),
        ]
        if width and width > 0:
            cmd.extend(["--width", str(int(width))])
        if height and height > 0:
            cmd.extend(["--height", str(int(height))])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        effective_timeout = max(2, int(timeout_seconds or self.capture_timeout_seconds))
        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(process.communicate(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise CameraCaptureError(f"Camera command timed out after {effective_timeout}s")

        stdout = stdout_raw.decode("utf-8", errors="replace")
        stderr = stderr_raw.decode("utf-8", errors="replace")

        if process.returncode != 0:
            raise CameraCaptureError(
                f"Camera command failed with exit code {process.returncode}. stderr: {stderr.strip() or '(empty)'}"
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise CameraCaptureError("Camera capture command finished but no image file was produced")

        return CameraCaptureResult(
            path=output_path,
            command=" ".join(cmd),
            stdout=stdout,
            stderr=stderr,
        )


def _resolve_camera_binary(preferred: str) -> str:
    if preferred and shutil.which(preferred):
        return preferred
    if preferred != "libcamera-still" and shutil.which("libcamera-still"):
        return "libcamera-still"
    if preferred != "rpicam-still" and shutil.which("rpicam-still"):
        return "rpicam-still"
    raise CameraCaptureError(
        f"Camera binary not found: '{preferred}'. Install rpicam-apps/libcamera tools and expose camera devices."
    )
