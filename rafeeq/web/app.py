"""The web UI as a Starlette app, mounted inside the FastAPI app (see rafeeq/web/__init__.py)."""
from __future__ import annotations

import os
import secrets
import warnings
from typing import Any

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from rafeeq.web import views_auth as auth
from rafeeq.web import views_rooms as rooms
from rafeeq.web import views_student as student
from rafeeq.web import views_studios as studios
from rafeeq.web import views_teacher as teacher
from rafeeq.web import views_tutor as tutor
from rafeeq.web.api import Api
from rafeeq.web.common import HERE, Redirect, is_htmx, notice, render, tr


def _routes() -> list:
    R = Route
    room = "/rooms/{room_id}"
    return [
        R("/", auth.home),
        R("/login", auth.login_page, methods=["GET"]),
        R("/login", auth.login_submit, methods=["POST"]),
        R("/register", auth.register_page, methods=["GET"]),
        R("/register", auth.register_submit, methods=["POST"]),
        R("/logout", auth.logout, methods=["POST"]),
        R("/lang/{code}", auth.set_language, methods=["GET"]),
        R("/p/{token}", teacher.parent_report_shared, methods=["GET"]),  # parent report share link, no login
        R("/rooms", rooms.rooms_index, methods=["GET"]),
        R("/rooms/join", rooms.rooms_join, methods=["POST"]),
        R("/rooms/create", rooms.rooms_create, methods=["POST"]),
        R(room, rooms.room_home, methods=["GET"]),
        # student
        R(room + "/study", tutor.study_page, methods=["GET"]),
        R(room + "/my-parent", teacher.my_parent_page, methods=["GET"]),
        R(room + "/my-parent/save", teacher.my_parent_save, methods=["POST"]),
        R(room + "/study/message", tutor.remembering(student.study_message), methods=["POST"]),
        R(room + "/study/photo", student.study_photo, methods=["POST"]),
        R(room + "/study/agent", tutor.remembering(tutor.agent_action, {"research"}), methods=["POST"]),
        R(room + "/study/quiz-result", tutor.quiz_result, methods=["POST"]),
        R(room + "/study/chats/{cid}/delete", tutor.chat_delete, methods=["POST"]),
        R(room + "/study/memory/forget", tutor.memory_delete, methods=["POST"]),
        R(room + "/study/memory/{nid:int}/delete", tutor.memory_delete, methods=["POST"]),
        R(room + "/study-files", tutor.files_page, methods=["GET"]),
        R(room + "/study-files/{fid:int}", tutor.file_page, methods=["GET"]),
        R(room + "/study-files/{fid:int}/download", tutor.file_download, methods=["GET"]),
        R(room + "/study-files/{fid:int}/delete", tutor.file_delete, methods=["POST"]),
        R(room + "/practice", student.practice_page, methods=["GET"]),
        R(room + "/practice/quiz", student.practice_quiz, methods=["POST"]),
        R(room + "/practice/submit", student.practice_submit, methods=["POST"]),
        R(room + "/practice/{quiz_id}/{kind}.pdf", student.practice_pdf, methods=["GET"]),
        R(room + "/practice/{quiz_id}/grade", student.practice_grade, methods=["POST"]),
        R(room + "/progress", student.progress_page, methods=["GET"]),
        R(room + "/progress/quizzes/{quiz_id}", student.quiz_review_page, methods=["GET"]),
        # shared
        R(room + "/materials", student.materials_page, methods=["GET"]),
        R(room + "/materials/table", student.materials_table, methods=["GET"]),
        R(room + "/materials/upload", student.materials_upload, methods=["POST"]),
        R(room + "/materials/{file_id}/delete", student.materials_delete, methods=["POST"]),
        R(room + "/materials/{file_id}/download", student.materials_download, methods=["GET"]),
        R(room + "/rename", student.room_rename, methods=["POST"]),
        R(room + "/source", student.source_page, methods=["GET"]),
        # teacher
        R(room + "/overview", teacher.overview_page, methods=["GET"]),
        R(room + "/settings", teacher.room_settings_save, methods=["POST"]),
        R(room + "/students", teacher.students_page, methods=["GET"]),
        R(room + "/students/draft", teacher.students_draft, methods=["POST"]),
        R(room + "/students/{user_id}/report", teacher.parent_report_page, methods=["GET"]),
        R(room + "/students/{user_id}/parent-contact", teacher.parent_contact_save, methods=["POST"]),
        R(room + "/students/{user_id}/send-email", teacher.parent_report_send_email, methods=["POST"]),
        R(room + "/parent-batch", teacher.parent_batch_page, methods=["GET"]),
        R(room + "/parent-batch/template.xlsx", teacher.parent_batch_template, methods=["GET"]),
        R(room + "/parent-batch/import", teacher.parent_batch_import, methods=["POST"]),
        R(room + "/parent-batch/prepare", teacher.parent_batch_prepare, methods=["POST"]),
        R(room + "/parent-batch/approve-all", teacher.parent_batch_approve_all, methods=["POST"]),
        R(room + "/parent-batch/send-emails", teacher.parent_batch_send_emails, methods=["POST"]),
        R(room + "/parent-batch/{user_id}/edit", teacher.parent_batch_edit, methods=["POST"]),
        R(room + "/parent-batch/{user_id}/status", teacher.parent_batch_status, methods=["POST"]),
        R(room + "/parent-batch/{user_id}/wa-done", teacher.parent_batch_wa_done, methods=["POST"]),
        R(room + "/inbox", teacher.inbox_page, methods=["GET"]),
        R(room + "/inbox/{question_id}/draft", teacher.inbox_draft, methods=["POST"]),
        R(room + "/inbox/{question_id}/answer", teacher.inbox_answer, methods=["POST"]),
        R(room + "/insights", teacher.insights_page, methods=["GET"]),
        R(room + "/studio", teacher.studio_page, methods=["GET"]),
        R(room + "/studio/sessions", teacher.studio_generate, methods=["POST"]),
        R(room + "/studio/sessions/{session_id}", teacher.session_page, methods=["GET"]),
        R(room + "/studio/sessions/{session_id}/decision", teacher.session_decision, methods=["POST"]),
        R(room + "/studio/sessions/{session_id}/files/{name:path}", teacher.session_file, methods=["GET"]),
        # Lesson Studio entry + the other studios, each on its own page
        R("/studio", studios.studio_entry, methods=["GET"]),
        R("/video", studios.video_page, methods=["GET"]),
        R("/video/search", studios.video_search, methods=["POST"]),
        R("/video/searches/{search_id}", studios.video_history_open, methods=["GET"]),
        R("/video/searches/{search_id}/delete", studios.video_history_delete, methods=["POST"]),
        R("/video/moments", studios.video_moment_save, methods=["POST"]),
        R("/video/cover", studios.video_cover_save, methods=["POST"]),
        R("/presentations", studios.presentations_page, methods=["GET"]),
        R("/presentations/open", studios.presentation_open, methods=["GET"]),
        R("/presentations/themes", studios.theme_extract, methods=["POST"]),
        R("/presentations/{deck_id}", studios.deck_page, methods=["GET"]),
        R("/presentations/{deck_id}", studios.deck_save, methods=["POST"]),
        R("/presentations/{deck_id}/export", studios.deck_export, methods=["POST"]),
        R("/presentations/{deck_id}/slides/{index:int}/image", studios.deck_slide_image, methods=["POST"]),
        R("/presentations/{deck_id}/slides/{index:int}/visual", studios.deck_slide_visual, methods=["POST"]),
        R("/presentations/{deck_id}/visuals/{task_id}", studios.deck_visual_status, methods=["GET"]),
        R("/media", studios.media_page, methods=["GET"]),
        R("/media/upload", studios.media_upload, methods=["POST"]),
        R("/media/speech", studios.media_speech, methods=["POST"]),
        R("/story", studios.story_page, methods=["GET"]),
        R("/story", studios.story_create, methods=["POST"]),
        R("/story/{asset_id}", studios.story_status, methods=["GET"]),
        R("/media/video/generate", studios.media_video_generate, methods=["POST"]),
        R("/media/video/generate/{task_id}", studios.media_video_status, methods=["GET"]),
        R("/media/{asset_id}/edit", studios.media_edit, methods=["POST"]),
        R("/media/montage", studios.montage_start, methods=["POST"]),
        R("/media/{asset_id}/editor", studios.montage_editor_page, methods=["GET"]),
        R("/media/{asset_id}/editor/{action}", studios.montage_editor_api, methods=["POST"]),
        R("/media/{asset_id}/montage", studios.montage_start, methods=["POST"]),
        R("/montage/{asset_id}", studios.montage_status, methods=["GET"]),
        R("/montage/{asset_id}/captions.srt", studios.montage_captions, methods=["GET"]),
        R("/learning", studios.learning_page, methods=["GET"]),
        R("/learning/courses", studios.course_create, methods=["POST"]),
        R("/learning/{course_id}", studios.course_page, methods=["GET"]),
        R("/learning/{course_id}/preview", studios.course_preview, methods=["GET"]),
        R("/learning/{course_id}/update", studios.course_update, methods=["POST"]),
        R("/learning/{course_id}/delete", studios.course_delete, methods=["POST"]),
        R("/learning/{course_id}/publish", studios.course_publish, methods=["POST"]),
        R("/learning/{course_id}/unpublish", studios.course_unpublish, methods=["POST"]),
        R("/learning/{course_id}/lessons", studios.lesson_add, methods=["POST"]),
        R("/learning/{course_id}/lessons/{lesson_id}/update", studios.lesson_update, methods=["POST"]),
        R("/learning/{course_id}/lessons/{lesson_id}/delete", studios.lesson_delete, methods=["POST"]),
        R("/learning/{course_id}/lessons/{lesson_id}/move", studios.lesson_move, methods=["POST"]),
        R("/learning/{course_id}/lessons/{lesson_id}/assets", studios.lesson_assets, methods=["POST"]),
        R(room + "/learn", studios.learn_page, methods=["GET"]),
        R(room + "/learn/{course_id}", studios.learn_course, methods=["GET"]),
        R(room + "/learn/{course_id}/progress", studios.learn_progress, methods=["POST"]),
        R(room + "/courses/{course_id}/assets/{asset_id}/file", studios.learn_file, methods=["GET"]),
        R("/library", teacher.library_page, methods=["GET"]),
        R("/assets/{asset_id}/file", teacher.asset_file, methods=["GET"]),
        R("/profile", teacher.profile_page, methods=["GET"]),
        R("/profile", teacher.profile_save, methods=["POST"]),
        R("/profile/avatar", teacher.profile_avatar, methods=["GET"]),
        R("/profile/avatar", teacher.profile_avatar_save, methods=["POST"]),
        Mount("/static", app=StaticFiles(directory=str(HERE / "static")), name="static"),
    ]


async def _on_redirect(request: Request, exc: Redirect) -> Response:
    if is_htmx(request):
        return Response(status_code=200, headers={"HX-Redirect": exc.url})
    return RedirectResponse(exc.url, status_code=303)


async def _on_http_error(request: Request, exc: HTTPException) -> Response:
    if exc.status_code == 403 and exc.detail == "csrf":
        if is_htmx(request):
            return notice(request, tr(request, "err.csrf"), "warning")
        return render(request, "error.html", status=403, code=403, message=tr(request, "err.csrf"))
    messages = {403: "err.forbidden", 404: "err.notfound"}
    message = tr(request, messages[exc.status_code]) if exc.status_code in messages else (
        exc.detail if isinstance(exc.detail, str) and exc.status_code == 502 else tr(request, "err.generic"))
    if is_htmx(request):
        return notice(request, message, "warning")
    return render(request, "error.html", status=exc.status_code, code=exc.status_code, message=message)


class SecurityHeaders:
    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [(b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"strict-origin-when-cross-origin"),
                            (b"x-frame-options", b"DENY")]
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


def create_web_app(api_app: Any = None, api_base: str | None = None, secret: str | None = None,
                   demo_accounts: bool | None = None) -> Starlette:
    secret = secret or os.environ.get("RAFEEQ_WEB_SECRET")
    if not secret:
        warnings.warn("RAFEEQ_WEB_SECRET is not set: using a random key, sessions end on restart.")
        secret = secrets.token_urlsafe(32)
    app = Starlette(
        routes=_routes(),
        middleware=[
            Middleware(SecurityHeaders),
            Middleware(SessionMiddleware, secret_key=secret, session_cookie="rafeeq_session", same_site="lax",
                       https_only=os.environ.get("RAFEEQ_HTTPS_ONLY", "0") == "1", max_age=14 * 24 * 3600),
        ],
        exception_handlers={Redirect: _on_redirect, HTTPException: _on_http_error},
    )
    app.state.api = Api(app=api_app, base_url=api_base)
    if demo_accounts is None:
        demo_accounts = os.environ.get("RAFEEQ_DEMO_ACCOUNTS", "1") != "0"
    app.state.demo_accounts = demo_accounts
    return app


__all__ = ["create_web_app"]
