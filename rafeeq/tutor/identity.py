"""Who is asking, for the pieces of M3 that M1's Copilot calls without a user (connect.tutor_answer).

M1's hook signature is tutor_answer(room_id, question): no student. This ASGI middleware reads the request's Authorization / X-User-Id once and keeps
the student in a ContextVar for the length of that request, so the question log can say "the same student asked twice" without ever exporting who.
"""
from __future__ import annotations

from contextvars import ContextVar

from .events import current_user_id

REQUEST_USER: ContextVar[str | None] = ContextVar("request_user", default=None)


class IdentityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        token = REQUEST_USER.set(current_user_id(headers))
        try:
            await self.app(scope, receive, send)
        finally:
            REQUEST_USER.reset(token)
