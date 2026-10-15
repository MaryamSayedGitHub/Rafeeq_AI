from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from rafeeq.teacher.lecturer.nodes.code_agent import code_agent
from rafeeq.teacher.lecturer.nodes.draft_agent import draft_agent
from rafeeq.teacher.lecturer.nodes.final_agent import final_agent
from rafeeq.teacher.lecturer.nodes.human_approval import human_approval
from rafeeq.teacher.lecturer.nodes.orchestrator import orchestrator
from rafeeq.teacher.lecturer.nodes.quiz_agent import quiz_agent
from rafeeq.teacher.lecturer.nodes.research_agent import research_agent
from rafeeq.teacher.lecturer.nodes.slides_agent import slides_agent
from rafeeq.teacher.lecturer.state import AGENT_ORDER, SessionState, next_agent

# Every place an agent can hand over to (used by the router and by the diagram).
def targets_after(name: str) -> list[str]:
    """Where `name` can hand over to: only agents later in the order, or draft_agent."""
    start = AGENT_ORDER.index(name) + 1 if name in AGENT_ORDER else 0
    return [*AGENT_ORDER[start:], "draft_agent"]

def route_after(current: str):
    """Return a router function: 'who runs after `current`?'"""
    def router(state):
        return next_agent(state, current)
    return router


def route_after_approval(state) -> str:
    return "final_agent" if state["approval_status"] == "approved" else "orchestrator"


def build_graph(checkpointer=None, store=None):
    builder = StateGraph(SessionState)

    # 1) nodes
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("research_agent", research_agent)
    builder.add_node("slides_agent", slides_agent)
    builder.add_node("code_agent", code_agent)
    builder.add_node("quiz_agent", quiz_agent)
    builder.add_node("draft_agent", draft_agent)
    builder.add_node("human_approval", human_approval)
    builder.add_node("final_agent", final_agent)

    # 2) edges
    builder.add_edge(START, "orchestrator")

    for name in ["orchestrator", *AGENT_ORDER]:
        builder.add_conditional_edges(name, route_after(name), targets_after(name))
    builder.add_edge("draft_agent", "human_approval")
    builder.add_conditional_edges(
        "human_approval", route_after_approval, ["final_agent", "orchestrator"]
    )
    builder.add_edge("final_agent", END)

    # 3) compile (defaults keep scripts and tests working without a database)
    return builder.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store if store is not None else InMemoryStore(),
    )
