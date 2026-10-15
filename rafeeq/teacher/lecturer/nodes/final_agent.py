import logging

from rafeeq.teacher.lecturer.settings import OUTPUT_DIR
from rafeeq.teacher.lecturer.long_term_store import current_store, save_session
from rafeeq.teacher.lecturer import video_discovery
from rafeeq.teacher.lecturer.nodes.export import slugify, write_code_file, write_quiz_files

log = logging.getLogger(__name__)


def _create_pptx(relative_path: str, draft: dict, lecturer: str) -> None:
    """Through the pptx MCP server. If the server cannot start, build the file directly:
    the lecturer already approved the draft, so a tool problem must not lose the slides."""
    deck = {
        "title": draft["topic"],
        "course_name": draft.get("course_name", ""),
        "lecturer": lecturer,
        "slides": draft["slides"],
        "code_examples": draft.get("code_examples", []),
        "quiz": draft.get("quiz", []),
    }
    try:
        from rafeeq.teacher.lecturer.mcp_servers.mcp_client import create_presentation

        create_presentation(relative_path, deck)
    except Exception as e:
        log.warning("pptx MCP server failed, building directly. %s: %s", type(e).__name__, str(e)[:200])
        from rafeeq.teacher.lecturer.mcp_servers.pptx_builder import build_presentation

        build_presentation(OUTPUT_DIR / relative_path, deck)


def final_agent(state) -> dict:
    """Reads: brief, draft, feedback_history.   Writes: final_output.

    Runs only after the lecturer approved. Creates the files in outputs/teacher/<session_id>/ and
    saves what was learned about the lecturer to long-term memory. No LLM is used here.
    """
    brief = state["brief"]
    draft = state["draft"]
    session_id = slugify(brief["session_id"])
    folder = OUTPUT_DIR / session_id
    folder.mkdir(parents=True, exist_ok=True)

    files: list[str] = []

    if draft.get("slides"):
        name = f"{slugify(draft['topic'], 'slides')}.pptx"
        _create_pptx(f"{session_id}/{name}", draft, brief.get("lecturer_name") or brief.get("lecturer_id", ""))
        files.append(name)

    if draft.get("code_examples"):
        files.append(write_code_file(folder, draft["code_examples"], brief["programming_language"]).name)

    if draft.get("quiz"):
        files += [p.name for p in write_quiz_files(folder, draft["quiz"], draft["topic"])]

    memory_saved = save_session(
        current_store(), brief, draft.get("outline", []), state.get("feedback_history", []), files
    )

    # Teacher library: the files of this session become assets (never fails the session).
    try:
        from rafeeq.teacher import assets

        assets.register_session_files(current_store(), brief.get("lecturer_id", ""), session_id, files,
                                      topic=draft.get("topic", ""), room_id=brief.get("room_id", ""))
    except Exception as e:
        log.warning("Could not register session files in the library: %s: %s", type(e).__name__, str(e)[:200])

    # Presentation Studio: the approved slides become an editable deck (never fails the session).
    try:
        from rafeeq.teacher import decks

        decks.create_from_draft(current_store(), brief.get("lecturer_id", ""), session_id, draft, brief)
    except Exception as e:
        log.warning("Could not create the editable deck: %s: %s", type(e).__name__, str(e)[:200])

    # Video discovery (Lab 4): index each slide's speaker notes as a timed segment, so a
    # student question can later jump to the exact slide that explains it. Never blocks or
    # fails the session -- see video_discovery.index_session's own try/except.
    video_discovery.index_session(
        brief["lecturer_id"], session_id, draft.get("topic", ""), draft.get("slides", [])
    )

    return {
        "final_output": {"folder": session_id, "files": files, "memory_saved": memory_saved},
        "messages": [f"final_agent: created {len(files)} file(s)"],
    }
