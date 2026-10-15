"""
Custom MCP server with ONE tool: create_presentation(...).

final_agent uses it to turn the approved slides into a .pptx file.

Safety: the tool only writes inside outputs/teacher/. A path that points
anywhere else is rejected, so a model can never make it overwrite other files.

Run the server by itself (it waits for an MCP client on stdin/stdout):
    uv run python rafeeq/teacher/lecturer/mcp_servers/pptx_server.py
"""

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))  # the server runs as a script: make `rafeeq` importable

from rafeeq.teacher.lecturer.mcp_servers.pptx_builder import build_presentation  # noqa: E402
from rafeeq.teacher.lecturer.settings import OUTPUT_DIR  # noqa: E402

mcp = FastMCP("pptx")


@mcp.tool()
def create_presentation(relative_path: str, deck: dict) -> str:
    """Create a PowerPoint file inside outputs/.

    relative_path: e.g. "session-123/python-decorators.pptx" (must end with .pptx).
    deck: {"title", "course_name", "lecturer", "slides": [...], "code_examples": [...], "quiz": [...]}
          (the full shape is documented in pptx_builder.build_presentation).
    Returns a JSON string: {"ok": bool, "path": str, "error": str}.
    """
    target = (OUTPUT_DIR / relative_path).resolve()
    if not target.is_relative_to(OUTPUT_DIR) or target.suffix != ".pptx":
        return json.dumps({"ok": False, "path": "", "error": "path must be a .pptx inside outputs/"})
    try:
        build_presentation(target, deck)
    except Exception as e:
        return json.dumps({"ok": False, "path": "", "error": f"{type(e).__name__}: {e}"})
    return json.dumps({"ok": True, "path": str(target), "error": ""})


if __name__ == "__main__":
    mcp.run()  # stdio transport
