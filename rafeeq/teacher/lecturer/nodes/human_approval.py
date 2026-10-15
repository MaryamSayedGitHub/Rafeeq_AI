from langgraph.types import interrupt


def human_approval(state) -> dict:
    """Reads: draft, revision_count.   Writes: approval_status, feedback, revision_count.

    interrupt() pauses the graph here and saves the state in the checkpointer. The API returns
    the draft to the lecturer. Later, Command(resume={...}) continues from this exact line and
    `decision` receives the value that was sent.

    Expected resume value: {"decision": "approve" | "revise", "feedback": "..."}
    The limit on the number of revisions is enforced by GraphController before resuming.
    """
    decision = interrupt({
        "draft": state["draft"],
        "revision_count": state.get("revision_count", 0),
    })

    if isinstance(decision, dict) and decision.get("decision") == "revise":
        feedback = (decision.get("feedback") or "").strip()
        return {
            "approval_status": "revise",
            "feedback": feedback,
            "feedback_history": [feedback] if feedback else [],
            "revision_count": state.get("revision_count", 0) + 1,
            "messages": ["human_approval: changes requested"],
        }

    return {"approval_status": "approved", "messages": ["human_approval: approved"]}
