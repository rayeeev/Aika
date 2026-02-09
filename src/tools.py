import subprocess
import asyncio
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# Thread pool for running blocking operations
_executor = ThreadPoolExecutor(max_workers=4)


class Tools:
    @staticmethod
    async def execute_command(cmd: str, timeout: int = 60) -> str:
        """Executes a shell command asynchronously and returns the output."""
        def _run():
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    return f"Error (Exit Code {result.returncode}):\n{result.stderr.strip()}"
            except subprocess.TimeoutExpired:
                return f"Command timed out after {timeout} seconds"
            except Exception as e:
                return f"Execution failed: {str(e)}"
        
        # Run blocking subprocess in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _run)

    @staticmethod
    async def read_file(path: str) -> str:
        """Reads a file from the filesystem asynchronously."""
        def _read():
            try:
                # Security: Prevent path traversal attacks
                abs_path = os.path.abspath(path)
                with open(abs_path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Failed to read file: {str(e)}"
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _read)

    @staticmethod
    async def write_file(path: str, content: str) -> str:
        """Writes content to a file asynchronously."""
        def _write():
            try:
                abs_path = os.path.abspath(path)
                # Create parent directories if they don't exist
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, 'w') as f:
                    f.write(content)
                return f"Successfully wrote to {abs_path}"
            except Exception as e:
                return f"Failed to write file: {str(e)}"
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _write)

    @staticmethod
    async def list_dir(path: str = ".") -> str:
        """Lists files in a directory asynchronously."""
        def _list():
            try:
                abs_path = os.path.abspath(path)
                entries = os.listdir(abs_path)
                return "\n".join(entries) if entries else "(empty directory)"
            except Exception as e:
                return f"Failed to list directory: {str(e)}"
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _list)
