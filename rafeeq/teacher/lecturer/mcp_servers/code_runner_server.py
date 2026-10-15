"""
Custom MCP server with ONE tool: run_python(code).

The code_agent uses it to check that a generated example really runs before showing it
to the lecturer.

Safety (read this):
- The code runs in a SEPARATE process, with a timeout, in an empty temp folder.
- The child process gets a minimal environment, so your API keys in .env are NOT visible to it.
- The output is truncated.
- It does NOT block the network or the disk. This is a local-development tool. For any public
  deployment run it inside Docker with `--network none` (Phase 12).

Run the server by itself (it waits for an MCP client on stdin/stdout):
    uv run python rafeeq/teacher/lecturer/mcp_servers/code_runner_server.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("code_runner")

TIMEOUT_SECONDS = 5
MAX_OUTPUT_CHARS = 4000

# Only these variables are passed to the child process. Windows needs SYSTEMROOT to start Python.
_SAFE_ENV_KEYS = ("PATH", "SYSTEMROOT", "TEMP", "TMP")


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n...[output truncated]"


def _clean_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}


@mcp.tool()
def run_python(code: str) -> str:
    """Run a Python snippet with a time limit.

    Returns a JSON string: {"ok": bool, "timed_out": bool, "stdout": str, "stderr": str}.
    ok is true only if the program exited with code 0.
    """
    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "snippet.py"
        script.write_text(code, encoding="utf-8")

        try:
            # -I = isolated mode: ignores PYTHON* variables and the user's site-packages.
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=workdir,
                env=_clean_env(),
                # The server's own stdin carries the MCP messages. The child must never touch it,
                # so it gets an empty stdin (input() inside the snippet fails fast with EOFError).
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_SECONDS,
            )
            result = {
                "ok": proc.returncode == 0,
                "timed_out": False,
                "stdout": _clip(proc.stdout),
                "stderr": _clip(proc.stderr),
            }
        except subprocess.TimeoutExpired:
            result = {
                "ok": False,
                "timed_out": True,
                "stdout": "",
                "stderr": f"Timed out after {TIMEOUT_SECONDS} seconds.",
            }

    return json.dumps(result)


if __name__ == "__main__":
    mcp.run()  # stdio transport