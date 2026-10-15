"""A stand-in for the Rafeeq API, used by the web UI tests.

M1 routes run the real rafeeq.platform.service and rafeeq.copilot code. M2-M5 routes
return the contract samples. Every request body the UI sends is validated against the
request schema in contracts/openapi.yaml; violations are recorded in `violations`, so a
test fails if the UI ever sends something the contract does not allow.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from jsonschema import Draft202012Validator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from rafeeq.copilot.supervisor import handle_message
from rafeeq.platform import service
from rafeeq.samples import load_sample
from rafeeq.snap.reader import NotAnImage, ReaderUnavailable, read_snap

SPEC = yaml.safe_load((Path(__file__).resolve().parent.parent / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))


def _validator(name):
    return Draft202012Validator({"$ref": f"#/components/schemas/{name}", "components": SPEC["components"]})


class _DictStore:
    """Just enough of a LangGraph store (put / search) for the inbox routes."""

    class _Item:
        def __init__(self, key, value):
            self.key, self.value = key, value

    def __init__(self):
        self.data = {}

    def put(self, namespace, key, value):
        self.data.setdefault(tuple(namespace), {})[key] = dict(value)

    def search(self, namespace, limit=10, offset=0, **kw):
        items = [self._Item(k, dict(v)) for k, v in self.data.get(tuple(namespace), {}).items()]
        return items[offset:offset + limit]


class ContractDouble:
    def __init__(self, disabled: set[str] | None = None):
        self.disabled = disabled or set()   # route keys that answer 404 (not implemented yet)
        self.violations: list[str] = []
        self.video_saved = False            # True: the status route reports the clip as saved in the library
        self.calls: list[tuple[str, str, object]] = []
        self.class_store = _DictStore()     # M5's store for the inbox routes
        self.app = Starlette(routes=self._routes())

    # -- helpers
    async def _body(self, request: Request, schema: str):
        body = await request.json()
        self.calls.append((request.method, request.url.path, body))
        for e in _validator(schema).iter_errors(body):
            self.violations.append(f"{request.method} {request.url.path} -> {schema}: {e.message}")
        return body

    def _user(self, request: Request):
        token = service.token_from_header(request.headers.get("authorization"))
        return service.get_store().user_for_token(token) if token else None

    def _guard(self, key: str):
        return JSONResponse({"detail": "Not Found"}, status_code=404) if key in self.disabled else None

    @staticmethod
    def _err(e: service.PlatformError):
        return JSONResponse({"detail": e.message}, status_code=e.status)

    def _routes(self):
        d = self

        async def login(request):
            body = await d._body(request, "LoginRequest")
            try:
                return JSONResponse(service.get_store().login(body["email"], body["password"]))
            except service.PlatformError as e:
                return d._err(e)

        async def register(request):
            b = await d._body(request, "RegisterRequest")
            try:
                return JSONResponse(service.get_store().register(b["name"], b["email"], b["password"], b["role"], b.get("join_code")))
            except service.PlatformError as e:
                return d._err(e)

        async def rooms(request):
            user = d._user(request)
            if user is None:
                return JSONResponse({"detail": "Log in first"}, status_code=401)
            if request.method == "GET":
                return JSONResponse(service.get_store().list_rooms(user))
            b = await d._body(request, "CreateRoomRequest")
            try:
                return JSONResponse(service.get_store().create_room(user, b["name"]))
            except service.PlatformError as e:
                return d._err(e)

        async def join(request):
            b = await d._body(request, "JoinRoomRequest")
            try:
                return JSONResponse(service.get_store().join_room(d._user(request), b["code"]))
            except service.PlatformError as e:
                return d._err(e)

        async def join_code(request):
            try:
                return JSONResponse(service.get_store().join_code(d._user(request), request.path_params["room_id"]))
            except service.PlatformError as e:
                return d._err(e)

        async def files(request):
            rid = request.path_params["room_id"]
            try:
                if request.method == "GET":
                    return JSONResponse(service.get_store().list_files(d._user(request), rid))
                form = await request.form()
                up = form["file"]
                return JSONResponse(service.get_store().add_file(d._user(request), rid, up.filename, await up.read()), status_code=202)
            except service.PlatformError as e:
                return d._err(e)

        async def file_delete(request):
            try:
                service.get_store().delete_file(d._user(request), request.path_params["room_id"], request.path_params["file_id"])
                return Response(status_code=204)
            except service.PlatformError as e:
                return d._err(e)

        async def parent_report(request):
            from rafeeq.teacher import parent_report as pr

            user = d._user(request)
            room = service.get_store().db["rooms"].get(request.path_params["room_id"])
            if user is None or room is None or room["teacher_id"] != user["user_id"]:
                return JSONResponse({"detail": "Only this room's teacher"}, status_code=403)
            out = pr.build(None, room["room_id"], request.path_params["user_id"])
            for e in _validator("ParentReport").iter_errors(out):
                d.violations.append(f"parent report -> ParentReport: {e.message}")
            return JSONResponse({**out, "share_token": pr.make_share_token(room["room_id"], request.path_params["user_id"])})

        async def shared_report(request):
            from rafeeq.teacher import parent_report as pr

            parsed = pr.read_share_token(request.path_params["token"])
            if parsed is None:
                return JSONResponse({"detail": "expired"}, status_code=404)
            return JSONResponse(pr.build(None, *parsed))

        async def inbox_list(request):
            from rafeeq.teacher import inbox

            user = d._user(request)
            room = service.get_store().db["rooms"].get(request.path_params["room_id"])
            if user is None or room is None:
                return JSONResponse({"detail": "no"}, status_code=403)
            mine = None if room["teacher_id"] == user["user_id"] else user["user_id"]
            out = {"questions": inbox.list_questions(room["room_id"], user_id=mine, store=d.class_store)}
            for e in _validator("InboxQuestions").iter_errors(out):
                d.violations.append(f"inbox -> InboxQuestions: {e.message}")
            return JSONResponse(out)

        async def inbox_answer(request):
            from rafeeq.teacher import inbox

            body = await d._body(request, "InboxAnswerRequest")
            user = d._user(request)
            room = service.get_store().db["rooms"].get(request.path_params["room_id"])
            if user is None or room is None or room["teacher_id"] != user["user_id"]:
                return JSONResponse({"detail": "Only the teacher"}, status_code=403)
            saved = inbox.answer(room["room_id"], request.path_params["question_id"], body["answer"], store=d.class_store)
            return JSONResponse({**saved, "student_name": ""}) if saved else JSONResponse({"detail": "nf"}, status_code=404)

        async def students(request):
            try:
                return JSONResponse(service.get_store().list_students(d._user(request), request.path_params["room_id"]))
            except service.PlatformError as e:
                return d._err(e)

        async def snap(request):
            form = await request.form()
            try:
                return JSONResponse(read_snap(await form["photo"].read()))
            except NotAnImage:
                return JSONResponse({"detail": "The upload is not an image"}, status_code=422)
            except ReaderUnavailable as e:
                return JSONResponse({"detail": str(e)}, status_code=503)

        async def copilot(request):
            b = await d._body(request, "CopilotRequest")
            user = d._user(request)
            try:
                service.get_store().room_for(user, b["room_id"])
            except service.PlatformError as e:
                return d._err(e)
            if user["role"] == "student":
                service.get_store().record_question(user["user_id"], b["room_id"])
            return JSONResponse(handle_message(b["room_id"], b["text"], b.get("file_id"), b.get("mode", "auto"), user["user_id"]))

        def sample_get(key, sample, required_query=()):
            async def handler(request):
                if (g := d._guard(key)):
                    return g
                d.calls.append((request.method, request.url.path, dict(request.query_params)))
                for q in required_query:
                    if q not in request.query_params:
                        d.violations.append(f"GET {request.url.path}: missing query parameter {q}")
                return JSONResponse(load_sample(sample))
            return handler

        def sample_post(key, schema, sample):
            async def handler(request):
                if (g := d._guard(key)):
                    return g
                await d._body(request, schema)
                return JSONResponse(load_sample(sample))
            return handler

        async def profile(request):
            if (g := d._guard("profile")):
                return g
            if request.method == "PUT":
                body = await d._body(request, "TeacherProfileUpdate")
                return JSONResponse({**load_sample("teacher_profile"), **{k: v for k, v in body.items() if k != "teacher_id"}})
            if "teacher_id" not in request.query_params:
                d.violations.append("GET /teacher/profile: missing teacher_id")
            return JSONResponse(load_sample("teacher_profile"))

        async def video_generate(request):
            if (g := d._guard("videos")):
                return g
            form = await request.form()
            d.calls.append((request.method, request.url.path, {k: v for k, v in form.items() if k != "image"}))
            if len(str(form.get("prompt", ""))) < 3:
                d.violations.append("POST /teacher/videos/generate: prompt missing")
            return JSONResponse({**load_sample("video_job"), "used_avatar": form.get("image") is None})

        async def video_status(request):
            if (g := d._guard("videos")):
                return g
            data = load_sample("video_job_status")
            if d.video_saved:
                data = {**data, "asset_id": "a_77aa10cd", "file_url": "/teacher/assets/a_77aa10cd/file"}
            return JSONResponse(data)

        async def assets(request):
            if (g := d._guard("assets")):
                return g
            if request.method == "GET":
                d.calls.append((request.method, request.url.path, dict(request.query_params)))
                return JSONResponse(load_sample("assets"))
            await d._body(request, "AssetCreate")
            return JSONResponse(load_sample("asset"), status_code=201)

        async def asset_file(request):
            if (g := d._guard("assets")):
                return g
            return Response(b"MP4-BYTES", media_type="video/mp4")

        async def avatar(request):
            if (g := d._guard("profile")):
                return g
            if request.method == "POST":
                form = await request.form()
                if not form.get("teacher_id") or form.get("image") is None:
                    d.violations.append("POST /teacher/avatar: teacher_id and image are required")
                return JSONResponse(load_sample("avatar_saved"))
            return Response(b"\x89PNG-AVATAR", media_type="image/png")

        async def session_file(request):
            if (g := d._guard("sessions")):
                return g
            return Response(b"PPTX-BYTES", media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

        return [
            Route("/auth/login", login, methods=["POST"]),
            Route("/auth/register", register, methods=["POST"]),
            Route("/rooms", rooms, methods=["GET", "POST"]),
            Route("/rooms/join", join, methods=["POST"]),
            Route("/rooms/{room_id}/join-code", join_code, methods=["GET"]),
            Route("/rooms/{room_id}/files", files, methods=["GET", "POST"]),
            Route("/rooms/{room_id}/files/{file_id}", file_delete, methods=["DELETE"]),
            Route("/rooms/{room_id}/students", students, methods=["GET"]),
            Route("/snap", snap, methods=["POST"]),
            Route("/copilot/message", copilot, methods=["POST"]),
            Route("/rooms/{room_id}/topics", sample_get("topics", "room_topics"), methods=["GET"]),
            Route("/sources/page", sample_get("sources", "source_page", ("room_id", "file_name", "page")), methods=["GET"]),
            Route("/quiz", sample_post("quiz", "MakeQuizRequest", "make_quiz"), methods=["POST"]),
            Route("/quiz/{quiz_id}/submit", sample_post("quiz", "SubmitQuizRequest", "submit_quiz"), methods=["POST"]),
            Route("/next-step", sample_get("next_step", "next_step", ("room_id",)), methods=["GET"]),
            Route("/profile/summary", sample_get("summary", "student_summary", ("room_id",)), methods=["GET"]),
            Route("/teacher/rooms/{room_id}/insights", sample_get("insights", "insights"), methods=["GET"]),
            Route("/teacher/rooms/{room_id}/report", sample_get("report", "class_report"), methods=["GET"]),
            Route("/teacher/rooms/{room_id}/students/{user_id}/parent-report", parent_report, methods=["GET"]),
            Route("/parent-report/{token}", shared_report, methods=["GET"]),
            Route("/rooms/{room_id}/inbox", inbox_list, methods=["GET"]),
            Route("/rooms/{room_id}/inbox/{question_id}/answer", inbox_answer, methods=["POST"]),
            Route("/teacher/assistant", sample_post("assistant", "TeacherAssistantRequest", "teacher_assistant"), methods=["POST"]),
            Route("/teacher/profile", profile, methods=["GET", "PUT"]),
            Route("/teacher/sessions", sample_get("sessions", "teacher_sessions", ("teacher_id",)), methods=["GET"]),
            Route("/teacher/sessions/{sid}", sample_get("sessions", "lecturer_session"), methods=["GET"]),
            Route("/teacher/sessions/{sid}/files/{name}", session_file, methods=["GET"]),
            Route("/teacher/videos/search", sample_get("videos", "video_search", ("q",)), methods=["GET"]),
            Route("/teacher/videos/generate", video_generate, methods=["POST"]),
            Route("/teacher/videos/generate/{task_id}", video_status, methods=["GET"]),
            Route("/teacher/assets", assets, methods=["GET", "POST"]),
            Route("/teacher/assets/{asset_id}/file", asset_file, methods=["GET"]),
            Route("/teacher/avatar", avatar, methods=["POST"]),
            Route("/teacher/avatar/{teacher_id}", avatar, methods=["GET"]),
        ] + [
            Route("/teacher/sessions", sample_post("sessions", "LecturerSessionRequest", "lecturer_session"), methods=["POST"]),
            Route("/teacher/sessions/{sid}/decision", sample_post("sessions", "LecturerDecision", "lecturer_session"), methods=["POST"]),
        ]
