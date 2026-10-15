def draft_agent(state) -> dict:
    """Reads: brief, outline, slides, code_examples, quiz.   Writes: draft.

    No LLM here: it only packs everything into one dict. The keys match the LecturerDraft schema in contracts/openapi.yaml.
    """
    brief = state["brief"]

    draft = {
        "topic": brief["topic"],
        "course_name": brief.get("course_name", ""),
        "language": brief["language"],
        "outline": state.get("outline", []),
        "slides": state.get("slides", []),
        "code_examples": state.get("code_examples", []),
        "quiz": state.get("quiz", []),
    }

    return {
        "draft": draft,
        "approval_status": "pending",
        "messages": ["draft_agent: draft is ready"],
    }