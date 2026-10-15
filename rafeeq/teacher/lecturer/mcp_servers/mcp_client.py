"""
MCP client: the bridge between our graph and the MCP servers.

It knows three servers: code_runner and pptx (our own, stdio) and Tavily (hosted by Tavily,
streamable HTTP).

The graph nodes are normal (sync) functions, but MCP is async. `run_python_code()` hides that,
so code_agent can simply call it like a normal function.
"""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from rafeeq.teacher.lecturer.settings import TAVILY_API_KEY

SERVERS_DIR = Path(__file__).resolve().parent

# Tavily's hosted MCP server. The API key travels in the URL (that is how Tavily designed it),
# so we must never print the URL or an error message that contains it. See _describe().
TAVILY_MCP_URL = os.getenv("TAVILY_MCP_URL", "https://mcp.tavily.com/mcp/")


def _server_config() -> dict:
    config = {
        "code_runner": {
            "transport": "stdio",
            # sys.executable = the python of the current virtual environment
            "command": sys.executable,
            "args": [str(SERVERS_DIR / "code_runner_server.py")],
        },
        "pptx": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(SERVERS_DIR / "pptx_server.py")],
        },
    }
    if TAVILY_API_KEY:
        config["tavily"] = {
            "transport": "streamable_http",
            "url": f"{TAVILY_MCP_URL}?tavilyApiKey={TAVILY_API_KEY}",
        }
    return config


def _describe(error: BaseException) -> str:
    """Readable error text: flattens ExceptionGroup and hides the Tavily API key."""
    if isinstance(error, BaseExceptionGroup):
        text = "; ".join(_describe(e) for e in error.exceptions)
    else:
        text = f"{type(error).__name__}: {error}"
    if TAVILY_API_KEY:
        text = text.replace(TAVILY_API_KEY, "***")
    return text


def run_async(coro):
    """Run a coroutine from sync code, even if an event loop is already running (e.g. FastAPI)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # no loop running: the normal case

    # A loop is already running in this thread: run the coroutine in a helper thread.
    box: dict = {}

    def worker():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001 - re-raised below in the caller's thread
            box["error"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _tool_text(result) -> str:
    """A tool reply is a string or a list of content blocks. Return it as plain text."""
    if isinstance(result, str):
        return result
    parts = []
    for block in result:
        parts.append(block if isinstance(block, str) else block.get("text", ""))
    return "".join(parts)


async def _call_tool(server: str, is_wanted, args: dict) -> str:
    """Connect to one server, find the tool that `is_wanted(name)` accepts, call it, return text."""
    client = MultiServerMCPClient(_server_config())
    tools = await client.get_tools(server_name=server)
    tool = next((t for t in tools if is_wanted(t.name)), None)
    if tool is None:
        raise RuntimeError(f"No matching tool on server '{server}'. Available: {[t.name for t in tools]}")
    return _tool_text(await tool.ainvoke(args))


def run_python_code(code: str) -> dict:
    """Run Python code through the code_runner server.

    Returns {"ok": bool, "timed_out": bool, "stdout": str, "stderr": str}.
    """
    text = run_async(_call_tool("code_runner", lambda name: name == "run_python", {"code": code}))
    return json.loads(text)


def create_presentation(relative_path: str, deck: dict) -> str:
    """Create a .pptx inside outputs/ through the pptx server. Returns the file path.

    Raises RuntimeError if the server reports a problem.
    """
    text = run_async(_call_tool(
        "pptx",
        lambda name: name == "create_presentation",
        {"relative_path": relative_path, "deck": deck},
    ))
    result = json.loads(text)
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result["path"]


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web through Tavily's MCP server. Returns the results as text.

    Raises RuntimeError (with the API key removed from the message) if anything goes wrong.
    """
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set in .env")

    # Tavily names it "tavily-search"; some wrappers use "tavily_search". Accept both,
    # but not "tavily-research" (which also contains the word "search").
    def is_search(name: str) -> bool:
        return name.lower().replace("_", "-") == "tavily-search"

    try:
        return run_async(_call_tool("tavily", is_search, {"query": query, "max_results": max_results}))
    except Exception as e:
        raise RuntimeError(_describe(e)) from None


if __name__ == "__main__":
    print(run_python_code("print('hello from MCP')"))
    print(run_python_code("print(1/0)"))

    if TAVILY_API_KEY:
        print("--- Tavily ---")
        print(web_search("Python decorators tutorial", max_results=2)[:600])
    else:
        print("(TAVILY_API_KEY is not set, skipping the Tavily check)")