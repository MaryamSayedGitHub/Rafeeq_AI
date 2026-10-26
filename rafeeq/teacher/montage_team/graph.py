"""The montage team as a LangGraph graph.

    START -> supervisor -> analyst -> cutter -> transcriber -> chapters -> designer -> sticker -> assembler -> qa -> END
                              (each arrow only goes to the NEXT agent the supervisor planned: the others are skipped)
                                                      assembler  <--- (once) ---  qa     when the file fails the check

The same node functions run through a plain loop when LangGraph is not installed (CI, a laptop without the extra),
so the montage works either way and the tests do not need the library.
"""
import logging
from typing import Any

from rafeeq.teacher.montage_team import agents
from rafeeq.teacher.montage_team.state import ADD_KEYS, AGENT_ORDER

log = logging.getLogger(__name__)
END_NODE = "__end__"            # = langgraph.graph.END


def targets_after(name: str) -> list[str]:
    start = AGENT_ORDER.index(name) + 1 if name in AGENT_ORDER else 0
    return AGENT_ORDER[start:]


def next_agent(state, current: str) -> str:
    """The next planned agent after `current` (supervisor = start of the list). qa is always planned."""
    start = AGENT_ORDER.index(current) + 1 if current in AGENT_ORDER else 0
    return next(a for a in AGENT_ORDER[start:] if a in state["plan"])


def route_after(current: str):
    """A router function: 'who runs after `current`?' (same pattern as the lecturer graph)."""
    def router(state):
        return next_agent(state, current)
    return router


def route_after_qa(state) -> str:
    return END_NODE if state.get("qa_ok") else "assembler_agent"


def build_graph():
    from langgraph.graph import END, START, StateGraph
    from rafeeq.teacher.montage_team.state import MontageState

    g = StateGraph(MontageState)
    g.add_node("supervisor", agents.supervisor)
    for name, fn in agents.NODES.items():
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    for name in ["supervisor", *AGENT_ORDER[:-1]]:
        g.add_conditional_edges(name, route_after(name), targets_after(name))
    g.add_conditional_edges("qa_agent", route_after_qa, ["assembler_agent", END])
    return g.compile()


def _build_or_none():
    try:
        return build_graph()
    except ImportError:
        return None


_graph = None
_graph_tried = False


def _get_graph():
    global _graph, _graph_tried
    if not _graph_tried:
        _graph, _graph_tried = _build_or_none(), True
    return _graph


def _merge(state: dict, update: dict) -> dict:
    out = dict(state)
    for k, v in update.items():
        out[k] = [*out.get(k, []), *v] if k in ADD_KEYS else v
    return out


def _run_plain(state: dict, config: dict) -> dict:
    """The same flow without LangGraph."""
    state = _merge(state, agents.supervisor(state, config))
    name = "supervisor"
    for _ in range(30):                                            # a loop guard; the real flow is at most ~9 steps
        name = route_after_qa(state) if name == "qa_agent" else next_agent(state, name)
        if name == END_NODE:
            return state
        state = _merge(state, agents.NODES[name](state, config))
    raise RuntimeError("montage team did not finish")


def run_team(src, work, opts: dict, *, music_file=None, transcribe=None, propose=None, progress=None,
             use_langgraph: bool = True, sticker_files=None, fetch=None) -> dict:
    """Run the team. Returns {'file','srt','captions','chapters','cut','notes','seconds','agents','plan'}."""
    deps: dict[str, Any] = {"music_file": music_file, "transcribe": transcribe, "propose": propose, "progress": progress,
                       "sticker_files": sticker_files or {}, "fetch": fetch}
    config = {"configurable": {"deps": deps}, "tags": ["rafeeq", "teacher", "montage"], "run_name": "montage-team",
              "metadata": {"montage_title": str(opts.get("title", ""))[:60]}}
    state = {"src": src, "work": work, "opts": opts, "notes": [], "trace": []}
    graph = _get_graph() if use_langgraph else None
    out = graph.invoke(state, config) if graph is not None else _run_plain(state, config)
    return {**out["result"], "agents": out.get("trace", []), "plan": out.get("plan", [])}
