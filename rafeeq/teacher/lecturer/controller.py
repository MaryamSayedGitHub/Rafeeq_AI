"""Wraps the compiled graph so the API layer never touches LangGraph details.

    start(brief)                       -> run until the human-approval pause, return the draft
    decide(session_id, decision, ...)  -> resume: create the files, or build a revised draft
    get_state(session_id)              -> where a session is now
"""

import logging
import uuid

from langgraph.types import Command

from rafeeq.teacher.lecturer.settings import MAX_REVISIONS
from rafeeq.teacher.lecturer.tracing import run_config

log = logging.getLogger(__name__)


class SessionNotFound(Exception):
    pass


class SessionNotWaiting(Exception):
    """The session is not paused at the approval step (already finished, or never got there)."""


class RevisionLimitReached(Exception):
    pass


class GraphController:
    def __init__(self, graph):
        self.graph = graph

    # ------------------------------------------------------------ helpers
    def _snapshot(self, session_id: str):
        snapshot = self.graph.get_state(run_config(session_id))
        if not snapshot.values:
            raise SessionNotFound(session_id)
        return snapshot

    @staticmethod
    def _waiting(snapshot) -> bool:
        return "human_approval" in snapshot.next

    def _view(self, session_id: str, snapshot) -> dict:
        """One response shape for every endpoint: the LecturerSession schema in contracts/openapi.yaml."""
        values = snapshot.values
        final = values.get("final_output") or {}
        if self._waiting(snapshot):
            status = "awaiting_approval"
        elif final:
            status = "done"
        else:
            status = "incomplete"
        revision_count = values.get("revision_count", 0)
        return {
            "session_id": session_id,
            "room_id": values.get("brief", {}).get("room_id", ""),
            "topic": values.get("brief", {}).get("topic", ""),
            "status": status,
            "selected_agents": values.get("selected_agents", []),
            "reasoning": values.get("reasoning", ""),
            "memory": values.get("memory_context", []),
            "draft": values.get("draft", {}),
            "revision_count": revision_count,
            "revisions_left": max(0, MAX_REVISIONS - revision_count),
            "files": [
                {"name": name, "url": f"/teacher/sessions/{session_id}/files/{name}"}
                for name in final.get("files", [])
            ],
        }

    # ------------------------------------------------------------- actions
    def start(self, brief: dict) -> dict:
        session_id = uuid.uuid4().hex
        initial_state = {
            "brief": {**brief, "session_id": session_id},
            "memory_context": [],
            "feedback": "",
            "feedback_history": [],
            "revision_count": 0,
            "approval_status": "pending",
            "messages": [],
        }
        config = run_config(session_id, brief.get("lecturer_id", ""))
        log.info("session %s: started (topic: %s)", session_id, brief.get("topic"))
        self.graph.invoke(initial_state, config)  # returns when the graph pauses for approval
        return self._view(session_id, self._snapshot(session_id))

    def decide(self, session_id: str, decision: str, feedback: str = "") -> dict:
        snapshot = self._snapshot(session_id)
        if not self._waiting(snapshot):
            raise SessionNotWaiting(session_id)
        if decision == "revise" and snapshot.values.get("revision_count", 0) >= MAX_REVISIONS:
            raise RevisionLimitReached(session_id)

        lecturer_id = snapshot.values.get("brief", {}).get("lecturer_id", "")
        log.info("session %s: lecturer decision = %s", session_id, decision)
        self.graph.invoke(
            Command(resume={"decision": decision, "feedback": feedback}),
            run_config(session_id, lecturer_id),
        )
        return self._view(session_id, self._snapshot(session_id))

    def get_state(self, session_id: str) -> dict:
        return self._view(session_id, self._snapshot(session_id))

    def output_folder(self, session_id: str) -> tuple[str, list[str]]:
        """(folder name inside outputs/, file names) of a finished session."""
        final = self._snapshot(session_id).values.get("final_output") or {}
        return final.get("folder", ""), final.get("files", [])
