import logging

from pydantic import BaseModel, Field

from rafeeq.teacher.lecturer.long_term_store import current_store, load_memory
from rafeeq.teacher.lecturer.nodes.utils import build_chain, format_optional
from rafeeq.teacher.lecturer.prompts import ORCHESTRATOR_PROMPT
from rafeeq.teacher.lecturer.state import AGENT_ORDER, AgentName

log = logging.getLogger(__name__)

# Maps what the lecturer ticks in the form ("needs") to an agent name.
NEEDS_TO_AGENT = {
    "slides": "slides_agent",
    "code": "code_agent",
    "quiz": "quiz_agent",
}


class RoutingDecision(BaseModel):
    selected_agents: list[AgentName] = Field(
        description="The specialist agents that must run. Always includes research_agent."
    )
    reasoning: str = Field(
        description="1 or 2 short sentences for the lecturer explaining why these agents were chosen."
    )


def rule_based_agents(needs: list[str]) -> list[str]:
    """The simple version: no LLM, just map the checkboxes to agents."""
    wanted = {NEEDS_TO_AGENT[n] for n in needs if n in NEEDS_TO_AGENT}
    return [a for a in AGENT_ORDER if a == "research_agent" or a in wanted]


def normalize(agents: list[str]) -> list[str]:
    """Safety net: research_agent is always there, no duplicates, fixed order."""
    return [a for a in AGENT_ORDER if a == "research_agent" or a in agents]


def orchestrator(state) -> dict:
    """Reads: brief, feedback, long-term memory.   Writes: selected_agents, reasoning, memory_context."""
    brief = state["brief"]

    # Long-term memory is read once per session (the first time the orchestrator runs).
    memory_context = state.get("memory_context") or load_memory(
        current_store(), brief.get("lecturer_id", "")
    )

    chain = build_chain(ORCHESTRATOR_PROMPT, RoutingDecision, temperature=0)

    try:
        result = chain.invoke({
            "topic": brief["topic"],
            "needs": ", ".join(brief["needs"]),
            "notes": format_optional(brief.get("notes")),
            "memory_context": format_optional(memory_context),
            "feedback": format_optional(state.get("feedback")),
            "language": brief["language"],
        })
        selected = normalize(result.selected_agents)
        reasoning = result.reasoning
    except Exception as e:
        # Log the real reason, otherwise a wrong API key looks the same as a bad JSON.
        log.warning("LLM decision failed, using rules. %s: %s", type(e).__name__, str(e)[:300])
        selected = rule_based_agents(brief["needs"])
        reasoning = "Plan built from the requested outputs."

    return {
        "selected_agents": selected,
        "reasoning": reasoning,
        "memory_context": memory_context,
        "messages": [f"orchestrator: {', '.join(selected)}"],
    }