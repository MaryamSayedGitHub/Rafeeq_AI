import logging

from rafeeq.teacher.lecturer.nodes.utils import build_chain, format_optional
from rafeeq.teacher.lecturer.mcp_servers.mcp_client import web_search
from rafeeq.teacher.lecturer.prompts import RESEARCH_PROMPT
from rafeeq.teacher.lecturer.schemas import OutlineOutput

log = logging.getLogger(__name__)

NO_SEARCH = "(web search was not available, use well-established knowledge)"
MAX_SEARCH_CHARS = 6000  # keep the prompt small: search results can be very long


def _search(brief: dict) -> str:
    """Search the web through Tavily (MCP). If it fails, continue without it: never crash the graph."""
    query = (
        f"{brief['topic']} {brief.get('programming_language', '')} "
        f"lecture outline for {brief['student_level']} students"
    ).strip()

    try:
        text = web_search(query, max_results=5)
    except Exception as e:  # the message is already cleaned (no API key) by mcp_client
        log.warning("web search failed, continuing without it: %s", str(e)[:300])
        return NO_SEARCH

    text = text.strip()
    if not text:
        log.warning("web search returned nothing")
        return NO_SEARCH

    log.info("web search OK (%d characters)", len(text))
    return text[:MAX_SEARCH_CHARS]


def research_agent(state) -> dict:
    """Reads: brief.   Writes: outline."""
    brief = state["brief"]

    chain = build_chain(RESEARCH_PROMPT, OutlineOutput, temperature=0.3)

    result = chain.invoke({
        "topic": brief["topic"],
        "level": brief["student_level"],
        "duration": brief["duration_minutes"],
        "objectives": format_optional(brief.get("learning_objectives")),
        "search_results": _search(brief),
    })

    return {"outline": result.sections}


if __name__ == "__main__":
    fake_state = {
        "brief": {
            "topic": "Python decorators",
            "student_level": "intermediate",
            "duration_minutes": 60,
            "programming_language": "Python",
            "learning_objectives": ["Students can write their own decorators."],
        },
    }
    out = research_agent(fake_state)

    assert list(out.keys()) == ["outline"]
    assert 3 <= len(out["outline"]) <= 10, f"got {len(out['outline'])} sections"
    for i, s in enumerate(out["outline"], 1):
        print(f"{i}. {s}")
    print("research_agent: OK")