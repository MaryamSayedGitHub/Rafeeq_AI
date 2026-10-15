"""M2 routes from contracts/openapi.yaml: POST /search, GET /rooms/{room_id}/topics, GET /sources/page.

Thin HTTP layer; the work is in rafeeq/knowledge/service.py. Every route needs
`Authorization: Bearer <token>` and the caller must belong to the room (teacher or member),
exactly like M1's room routes.
"""
from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from rafeeq.knowledge import service as knowledge
from rafeeq.platform import service as platform

router = APIRouter(tags=["M2 knowledge"])


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room_id: str
    query: str
    top_k: int = Field(default=knowledge.DEFAULT_TOP_K, ge=1, le=knowledge.MAX_TOP_K)


def _member(authorization: str | None, room_id: str) -> None:
    token = platform.token_from_header(authorization) if authorization else None
    if token is None:
        raise HTTPException(status_code=401, detail="Use: Authorization: Bearer <token>")
    store = platform.get_store()
    user = store.user_for_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    try:
        store.room_for(user, room_id)
    except platform.PlatformError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from None


def _run(fn, *args):
    try:
        return fn(*args)
    except knowledge.KnowledgeError as e:
        raise HTTPException(status_code=e.status, detail=str(e)) from None


@router.post("/search")
def search_material(request: SearchRequest, authorization: str | None = Header(default=None)):
    _member(authorization, request.room_id)
    return _run(knowledge.get_service().search, request.room_id, request.query, request.top_k)


@router.get("/rooms/{room_id}/topics")
def list_topics(room_id: str, authorization: str | None = Header(default=None)):
    _member(authorization, room_id)
    return _run(knowledge.get_service().list_topics, room_id)


@router.get("/sources/page")
def get_source_page(room_id: str = Query(...), file_name: str = Query(...), page: int = Query(..., ge=1),
                    authorization: str | None = Header(default=None)):
    _member(authorization, room_id)
    return _run(knowledge.get_service().source_page, room_id, file_name, page)
