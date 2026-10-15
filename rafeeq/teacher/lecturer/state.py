from operator import add
from typing import Annotated, Literal, TypedDict


class SessionBrief(TypedDict):
    session_id: str
    lecturer_id: str
    lecturer_name: str           # display name for the slide footer
    room_id: str                 # Rafeeq course room (empty = not linked to a room)
    topic: str
    course_name: str
    student_level: str
    duration_minutes: int
    language: str
    programming_language: str
    needs: list[str]
    num_quiz_questions: int
    learning_objectives: list[str]
    notes: str


class SessionState(TypedDict):
    # 1) Input
    brief: SessionBrief

    # 2) Orchestrator
    selected_agents: list[str]
    reasoning: str
    memory_context: list[str]

    # 3) Agents (each agent stores plain dicts, via model_dump())
    outline: list[str]
    slides: list[dict]
    code_examples: list[dict]
    quiz: list[dict]

    # 4) Draft + revision
    draft: dict
    approval_status: Literal["pending", "approved", "revise"]
    feedback: str                                   # the latest change request
    feedback_history: Annotated[list[str], add]     # every change request of this session
    revision_count: int

    # 5) Final output
    final_output: dict
    messages: Annotated[list[str], add]


AgentName = Literal["research_agent", "slides_agent", "code_agent", "quiz_agent"]
AGENT_ORDER: list[AgentName] = ["research_agent", "slides_agent", "code_agent", "quiz_agent"]


def next_agent(state, current: str) -> str:
    """Return the next selected agent after `current`, or 'draft_agent' if none is left."""
    selected = state["selected_agents"]

    # If current is not one of the 4 agents (e.g. "orchestrator"), start from the beginning.
    start = AGENT_ORDER.index(current) + 1 if current in AGENT_ORDER else 0

    for name in AGENT_ORDER[start:]:
        if name in selected:
            return name
    return "draft_agent"


if __name__ == "__main__":
    s = {"selected_agents": ["research_agent", "quiz_agent"]}
    assert next_agent(s, "orchestrator") == "research_agent"
    assert next_agent(s, "research_agent") == "quiz_agent"   # slides + code skipped
    assert next_agent(s, "quiz_agent") == "draft_agent"
    print("next_agent: OK")