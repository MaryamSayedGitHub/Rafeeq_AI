import logging
import re

from rafeeq.teacher.lecturer.nodes.utils import build_chain, format_optional, format_outline, warn_if_count_differs
from rafeeq.teacher.lecturer.llm import get_llm
from rafeeq.teacher.lecturer.mcp_servers.mcp_client import run_python_code
from rafeeq.teacher.lecturer.prompts import CODE_FIX_PROMPT, CODE_PROMPT
from rafeeq.teacher.lecturer.schemas import CodeOutput

log = logging.getLogger(__name__)

MAX_FIX_ATTEMPTS = 2  # how many times we ask the model to repair a crashing example


def _text(content) -> str:
    """A model reply is usually a string, but some providers return a list of blocks."""
    if isinstance(content, str):
        return content
    return "".join(b if isinstance(b, str) else b.get("text", "") for b in content)


def _extract_code(reply: str) -> str:
    """Take the code out of a ```python ... ``` block. If there is no block, use the whole reply."""
    match = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, re.DOTALL)
    return (match.group(1) if match else reply).strip()


def _fix_code(code: str, error: str) -> str:
    chain = (CODE_FIX_PROMPT | get_llm(temperature=0)).with_retry(stop_after_attempt=3)
    reply = chain.invoke({"code": code, "error": error})
    return _extract_code(_text(reply.content))


def verify_example(example: dict) -> dict:
    """Run the example. If it crashes, ask the model to fix it (up to MAX_FIX_ATTEMPTS times).

    Adds three keys to the example:
      verified      True = ran fine | False = still broken after the fixes | None = could not check
      fix_attempts  how many repairs were needed
      output        what the program printed (empty if it never ran successfully)
    """
    title = example.get("title", "example")
    code = example["code"]

    for attempt in range(MAX_FIX_ATTEMPTS + 1):
        try:
            result = run_python_code(code)
        except Exception as e:  # the MCP server itself failed: do not crash the whole graph
            log.warning("could not run '%s': %s: %s", title, type(e).__name__, str(e)[:200])
            return {**example, "code": code, "verified": None, "fix_attempts": attempt, "output": ""}

        if result["ok"]:
            return {**example, "code": code, "verified": True,
                    "fix_attempts": attempt, "output": result["stdout"].strip()}

        error = result["stderr"].strip() or "Unknown error"
        log.info("'%s' failed (try %d): %s", title, attempt + 1, error.splitlines()[-1])

        if attempt < MAX_FIX_ATTEMPTS:
            code = _fix_code(code, error)

    return {**example, "code": code, "verified": False,
            "fix_attempts": MAX_FIX_ATTEMPTS, "output": ""}


def code_agent(state) -> dict:
    """Reads: brief, outline, feedback.   Writes: code_examples (each one verified by running it)."""
    brief = state["brief"]

    num_examples = max(2, min(3, brief["duration_minutes"] // 30))

    chain = build_chain(CODE_PROMPT, CodeOutput, temperature=0.2)

    result = chain.invoke({
        "topic": brief["topic"],
        "outline": format_outline(state.get("outline")),
        "level": brief["student_level"],
        "programming_language": brief["programming_language"],
        "num_examples": num_examples,
        "language": brief["language"],
        "feedback": format_optional(state.get("feedback")),
    })

    warn_if_count_differs("code_agent", len(result.examples), num_examples)

    examples = [e.model_dump() for e in result.examples]

    # The runner executes Python only. For any other language we skip the check.
    if brief["programming_language"].strip().lower() == "python":
        examples = [verify_example(e) for e in examples]
    else:
        examples = [{**e, "verified": None, "fix_attempts": 0, "output": ""} for e in examples]

    return {"code_examples": examples}


if __name__ == "__main__":
    fake_state = {
        "brief": {
            "topic": "Python decorators",
            "student_level": "intermediate",
            "duration_minutes": 60,
            "programming_language": "Python",
            "language": "English",
        },
        "outline": ["Functions are objects", "Closures", "Writing a decorator"],
        "feedback": "",
    }
    out = code_agent(fake_state)

    assert list(out.keys()) == ["code_examples"]
    assert len(out["code_examples"]) > 0
    for e in out["code_examples"]:
        flag = {True: "verified", False: "STILL BROKEN", None: "not checked"}[e["verified"]]
        print(f"- {e['title']}: {flag} (fixes: {e['fix_attempts']})")
        if e["output"]:
            print("    prints:", e["output"].splitlines()[0])
    print("code_agent: OK")