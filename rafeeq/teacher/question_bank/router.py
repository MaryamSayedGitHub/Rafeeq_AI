"""M6 Question Bank API endpoints.

All routes are under /teacher/rooms/{room_id}/question-bank and require
a teacher Bearer token. A teacher only sees and changes the questions
they own, inside the room named in the URL.

Endpoints
---------
POST   /teacher/rooms/{room_id}/question-bank/generate
    Generate questions from the room's uploaded material.

GET    /teacher/rooms/{room_id}/question-bank
    List the teacher's bank items (optionally filtered by topic or approved status).

GET    /teacher/rooms/{room_id}/question-bank/search?q=...
    Ask in your own words: "give me 5 medium questions about photosynthesis".

POST   /teacher/rooms/{room_id}/question-bank/search/voice
    The same, but the request is an audio recording.

POST   /teacher/rooms/{room_id}/question-bank/backfill
    Give older questions a grading guide and a difficulty.

GET    /teacher/rooms/{room_id}/question-bank/{item_id}
    Retrieve a single bank item.

PUT    /teacher/rooms/{room_id}/question-bank/{item_id}
    Update editable fields (stem, answer, rubric, options, ...).

POST   /teacher/rooms/{room_id}/question-bank/{item_id}/grading
    Write a fresh grading guide (correct / partial / wrong) for one question.

PATCH  /teacher/rooms/{room_id}/question-bank/{item_id}/approve
    Approve or un-approve a question for inclusion in student quizzes.

DELETE /teacher/rooms/{room_id}/question-bank/{item_id}
    Remove a question from the bank.
"""
from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile

from rafeeq.teacher.question_bank.contracts import (
    ApproveRequest,
    BackfillResponse,
    BankItem,
    BankItemList,
    GenerateRequest,
    GenerateResponse,
    SearchResponse,
    UpdateRequest,
)
from rafeeq.teacher.question_bank.nlu import result_message
from rafeeq.teacher.question_bank.service import InvalidItem, get_service
from rafeeq.teacher.question_bank.stt import MAX_AUDIO_BYTES, SttError, SttUnavailable, transcribe

log = logging.getLogger(__name__)

router = APIRouter(prefix="/teacher/rooms/{room_id}/question-bank", tags=["M6 question-bank"])

Auth = Annotated[str | None, Header()]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require_teacher(authorization: str | None) -> str:
    """Returns teacher_id or raises HTTP 401/403 (same pattern as teacher/routes.py)."""
    if not authorization:
        from rafeeq.platform.access import anonymous_allowed

        if anonymous_allowed():
            return "demo-teacher"
        raise HTTPException(401, "Log in first")

    from rafeeq.platform import service as platform

    token = platform.token_from_header(authorization)
    user = platform.get_store().user_for_token(token) if token else None
    if user is None:
        raise HTTPException(401, "Invalid or expired token")
    if user.get("role") != "teacher":
        raise HTTPException(403, "Teachers only")
    return user["user_id"]


def _svc():
    svc = get_service()
    if svc is None:
        raise HTTPException(503, "Question bank is not initialised (DATABASE_URL missing?)")
    return svc


def _llm():
    """Reuse the same LLM the quiz module uses (M4_LLM_PROVIDER / LLM_*)."""
    from rafeeq.quiz.deps import get_llm

    try:
        return get_llm()
    except Exception as exc:
        log.warning("question_bank router: LLM not available", exc_info=True)
        raise HTTPException(503, "The question generator (LLM) is not configured") from exc


def _llm_or_none():
    """The LLM if there is one. Searching still works without it (simple rules read the request)."""
    try:
        return _llm()
    except HTTPException:
        return None


def _chunks(room_id: str) -> list[dict]:
    """Return the room's material chunks (M2), or an empty list."""
    try:
        from rafeeq.knowledge import integration as ki

        if not ki.enabled():
            return []
        from rafeeq.knowledge.service import get_service as get_ks

        return get_ks().room_chunks(room_id) or []
    except Exception:
        log.warning("question_bank router: could not load chunks for room=%s", room_id, exc_info=True)
        return []


def _to_bank_item(entry) -> BankItem:
    return BankItem(
        item=entry.item, teacher_id=entry.teacher_id, approved=entry.approved,
        grading=entry.grading, difficulty=entry.difficulty,
    )


def _owned(svc, room_id: str, item_id: str, teacher_id: str):
    """The item, only if it is in this room AND belongs to this teacher.

    Another room's item and a missing item both give 404 (do not leak that it exists);
    another teacher's item in this room gives 403.
    """
    result = svc.get_item(item_id, room_id=room_id)
    if result is None:
        raise HTTPException(404, f"Item {item_id!r} not found")
    if result.teacher_id != teacher_id:
        raise HTTPException(403, "You can only change your own questions")
    return result


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@router.post("/generate", response_model=GenerateResponse, status_code=201)
def generate(
    room_id: str,
    body: GenerateRequest,
    authorization: Auth = None,
):
    """Generate questions from the room's uploaded material and add them to the bank."""
    teacher_id = _require_teacher(authorization)
    svc = _svc()
    chunks = _chunks(room_id)

    if not chunks:
        raise HTTPException(
            422,
            "No indexed material found for this room. "
            "Upload and wait for the files to finish indexing before generating questions.",
        )

    items = svc.generate(
        room_id=room_id,
        teacher_id=teacher_id,
        chunks=chunks,
        llm=_llm(),
        n=body.n,
        kinds=body.kinds,
        topic_id=body.topic_id,
    )

    return GenerateResponse(generated=len(items), items=[_to_bank_item(entry) for entry in items])


@router.get("", response_model=BankItemList)
def list_items(
    room_id: str,
    authorization: Auth = None,
    topic_id: Optional[str] = Query(default=None),
    approved_only: bool = Query(default=False),
):
    """List the teacher's questions in this room (newest first)."""
    teacher_id = _require_teacher(authorization)
    svc = _svc()

    entries = svc.list_items(
        room_id,
        teacher_id=teacher_id,
        topic_id=topic_id,
        approved_only=approved_only,
    )
    items = [_to_bank_item(entry) for entry in entries]
    return BankItemList(items=items, total=len(items))


def _answer_request(svc, room_id: str, teacher_id: str, text: str, transcript: Optional[str] = None) -> SearchResponse:
    """Understand `text`, find the questions, and say what was understood."""
    result = svc.nlu_search(room_id, text, teacher_id=teacher_id, llm=_llm_or_none())
    items = [_to_bank_item(entry) for entry in result.entries]
    return SearchResponse(
        query=text,
        items=items,
        total=len(items),
        understood=result.intent,
        requested=result.intent.n,
        message=result_message(text, result.intent.n, len(items), len(result.related_ids), list(result.missing)),
        related_ids=sorted(result.related_ids),
        missing_levels=list(result.missing),
        transcript=transcript,
    )


@router.get("/search", response_model=SearchResponse)
def search(
    room_id: str,
    q: str = Query(..., min_length=1, description="What you want, in your own words (English or Arabic)"),
    authorization: Auth = None,
):
    """Ask for questions in your own words, e.g. "give me 5 medium questions about photosynthesis".

    The answer says what was understood (`understood`), how many were asked for (`requested`),
    and how many were found (`total`)."""
    teacher_id = _require_teacher(authorization)
    return _answer_request(_svc(), room_id, teacher_id, q.strip())


@router.post("/search/voice", response_model=SearchResponse)
def search_by_voice(
    room_id: str,
    audio: UploadFile = File(..., description="A recording of the request (webm, mp3, wav, m4a, ogg)"),
    language: Optional[str] = Form(default=None, description='"ar" or "en"; leave empty to detect'),
    authorization: Auth = None,
):
    """The same as /search, but the request is spoken. `transcript` in the answer is what we heard."""
    teacher_id = _require_teacher(authorization)
    svc = _svc()

    data = audio.file.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(422, "The recording is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, f"The recording is too big (maximum {MAX_AUDIO_BYTES // (1024 * 1024)} MB)")
    try:
        text = transcribe(data, audio.filename or "audio.webm", audio.content_type, language)
    except SttUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except SttError as exc:
        raise HTTPException(502, str(exc)) from exc
    text = (text or "").strip()
    if not text:
        raise HTTPException(422, "No speech was heard in the recording")

    return _answer_request(svc, room_id, teacher_id, text, transcript=text)


@router.post("/backfill", response_model=BackfillResponse)
def backfill(
    room_id: str,
    authorization: Auth = None,
):
    """Give your older questions in this room a grading guide and a difficulty (those that have none)."""
    teacher_id = _require_teacher(authorization)
    return BackfillResponse(updated=_svc().backfill(room_id, teacher_id, _llm_or_none()))


@router.get("/{item_id}", response_model=BankItem)
def get_item(
    room_id: str,
    item_id: str,
    authorization: Auth = None,
):
    """Get a single question by its item_id."""
    teacher_id = _require_teacher(authorization)
    return _to_bank_item(_owned(_svc(), room_id, item_id, teacher_id))


@router.put("/{item_id}", response_model=BankItem)
def update_item(
    room_id: str,
    item_id: str,
    body: UpdateRequest,
    authorization: Auth = None,
):
    """Edit an item's stem, answer, rubric, options, or right_choice."""
    teacher_id = _require_teacher(authorization)
    svc = _svc()
    _owned(svc, room_id, item_id, teacher_id)

    try:
        result = svc.update_item(item_id, **body.model_dump(exclude_none=True))
    except InvalidItem as exc:
        raise HTTPException(422, f"That edit makes the question invalid: {exc}") from exc
    if result is None:
        raise HTTPException(404, f"Item {item_id!r} not found")
    return _to_bank_item(result)


@router.post("/{item_id}/grading", response_model=BankItem)
def regenerate_grading(
    room_id: str,
    item_id: str,
    authorization: Auth = None,
):
    """Write a new grading guide (correct / partial / wrong) for one question.

    Use it after editing a question's answer or rubric, or for questions that have no guide yet.
    The teacher can also write the guide by hand with PUT (field `grading`)."""
    teacher_id = _require_teacher(authorization)
    svc = _svc()
    _owned(svc, room_id, item_id, teacher_id)

    result = svc.regenerate_grading(item_id, _llm())
    if result is None:
        raise HTTPException(404, f"Item {item_id!r} not found")
    return _to_bank_item(result)


@router.patch("/{item_id}/approve", response_model=BankItem)
def approve_item(
    room_id: str,
    item_id: str,
    body: ApproveRequest,
    authorization: Auth = None,
):
    """Approve or un-approve a question. Approved questions appear in student quizzes."""
    teacher_id = _require_teacher(authorization)
    svc = _svc()
    _owned(svc, room_id, item_id, teacher_id)

    try:
        result = svc.approve_item(item_id, body.approved)
    except InvalidItem as exc:
        raise HTTPException(422, f"This question cannot be approved: {exc}") from exc
    if result is None:
        raise HTTPException(404, f"Item {item_id!r} not found")
    return _to_bank_item(result)


@router.delete("/{item_id}", status_code=204)
def delete_item(
    room_id: str,
    item_id: str,
    authorization: Auth = None,
):
    """Delete a question from the bank."""
    teacher_id = _require_teacher(authorization)
    svc = _svc()
    _owned(svc, room_id, item_id, teacher_id)

    if not svc.delete_item(item_id):
        raise HTTPException(404, f"Item {item_id!r} not found")