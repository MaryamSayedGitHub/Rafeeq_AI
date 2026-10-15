"""M1 platform logic: accounts, rooms, join codes, room files and activity.

Plain Python with no FastAPI imports, so it can be tested directly. router.py is the thin
HTTP layer on top of it.

Storage is one JSON file (default: storage/platform/platform.json). That is enough for the
demo and keeps the real student journey working: register -> no rooms yet -> join with
the teacher's code -> see the room. Swap this class for a database later; the method
names are the interface router.py depends on.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------- errors


class PlatformError(Exception):
    status = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class Invalid(PlatformError):
    status = 400


class AuthError(PlatformError):
    status = 401


class Forbidden(PlatformError):
    status = 403


class NotFound(PlatformError):
    status = 404


class Conflict(PlatformError):
    status = 409


# ---------------------------------------------------------------- helpers

_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I, easy to read aloud
_PBKDF2_ROUNDS = 120_000
TOKEN_DAYS = 30  # a login lasts a month; the demo accounts' fixed tokens never expire
DEMO_PASSWORD = "rafeeq-demo"
DEMO_ACCOUNTS = [
    # user_id, name, email, role, fixed token (keeps the old DEMO_USER token working)
    ("u_sara", "سارة أحمد", "sara@rafeeq.demo", "student", "demo-token-sara-123"),
    ("u_mona", "د. منى", "mona@rafeeq.demo", "teacher", "demo-token-mona-456"),
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ROUNDS).hex()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


def _new_join_code() -> str:
    raw = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))
    return f"{raw[:3]}-{raw[3:]}"


def normalize_code(code: str) -> str:
    """Accept 'abc4kq', 'ABC 4KQ' or 'ABC-4KQ' as the same code."""
    raw = re.sub(r"[^A-Za-z0-9]", "", code or "").upper()
    return f"{raw[:3]}-{raw[3:]}" if len(raw) == 6 else raw


def token_from_header(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


# ---------------------------------------------------------------- upload hooks
# M2 registers its indexer here:  service.on_upload(lambda room_id, file_id, path: ...)
# and reports progress with:      service.get_store().set_file_status(room_id, file_id, "indexing" | "ready" | "failed")
_UPLOAD_HOOKS: list = []


def on_upload(fn) -> None:
    _UPLOAD_HOOKS.append(fn)


def clear_upload_hooks() -> None:
    _UPLOAD_HOOKS.clear()


# ---------------------------------------------------------------- delete hooks
# M2 removes a deleted file from its index:  service.on_delete(lambda room_id, file_id: ...)
_DELETE_HOOKS: list = []


def on_delete(fn) -> None:
    _DELETE_HOOKS.append(fn)


def clear_delete_hooks() -> None:
    _DELETE_HOOKS.clear()


# ---------------------------------------------------------------- store


class PlatformStore:
    def __init__(self, data_dir: Path | str, seed_demo: bool = True):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "platform.json"
        self._lock = threading.Lock()
        self.db = self._load()
        if seed_demo:
            self._seed_demo_accounts()

    # -- persistence
    def _load(self) -> dict:
        empty = {"users": {}, "tokens": {}, "token_times": {}, "rooms": {}, "members": {}, "files": {}, "questions": []}
        if not self.path.exists():
            return empty
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for key, value in empty.items():
            data.setdefault(key, value)
        return data

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.db, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def _seed_demo_accounts(self) -> None:
        changed = False
        for user_id, name, email, role, token in DEMO_ACCOUNTS:
            if user_id not in self.db["users"]:
                salt = secrets.token_hex(8)
                self.db["users"][user_id] = {
                    "user_id": user_id, "name": name, "email": email, "role": role,
                    "salt": salt, "password_hash": _hash_password(DEMO_PASSWORD, salt), "demo": True,
                }
                self.db["tokens"][token] = user_id
                changed = True
        if changed:
            self._save()

    # -- accounts
    def _login_payload(self, user: dict, token: str) -> dict:
        return {"token": token, "user_id": user["user_id"], "name": user["name"], "role": user["role"]}

    def _user_by_email(self, email: str) -> dict | None:
        email = email.strip().lower()
        return next((u for u in self.db["users"].values() if u["email"] == email), None)

    def register(self, name: str, email: str, password: str, role: str, join_code: str | None = None) -> dict:
        name, email = (name or "").strip(), (email or "").strip().lower()
        if not name:
            raise Invalid("Name is required")
        if "@" not in email or len(email) < 5:
            raise Invalid("Enter a valid email address")
        if len(password or "") < 6:
            raise Invalid("Password must be at least 6 characters")
        if role not in ("student", "teacher"):
            raise Invalid("Role must be student or teacher")
        with self._lock:
            if self._user_by_email(email):
                raise Conflict("An account with this email already exists")
            room = None
            if join_code and role == "student":
                room = self._room_by_code(join_code)
                if room is None:
                    raise Invalid("Join code not found")
            salt = secrets.token_hex(8)
            user = {"user_id": _new_id("u"), "name": name[:80], "email": email, "role": role,
                    "salt": salt, "password_hash": _hash_password(password, salt)}
            self.db["users"][user["user_id"]] = user
            token = secrets.token_urlsafe(24)
            self.db["tokens"][token] = user["user_id"]
            self.db["token_times"][token] = _now().isoformat()
            if room is not None:
                self._add_member(room["room_id"], user["user_id"])
            self._save()
            return self._login_payload(user, token)

    def login(self, email: str, password: str) -> dict:
        user = self._user_by_email(email or "")
        if user is None or not secrets.compare_digest(user["password_hash"], _hash_password(password or "", user["salt"])):
            raise AuthError("Wrong email or password")
        with self._lock:
            token = secrets.token_urlsafe(24)
            self.db["tokens"][token] = user["user_id"]
            self.db["token_times"][token] = _now().isoformat()
            self._save()
        return self._login_payload(user, token)

    def user_for_token(self, token: str | None) -> dict | None:
        user_id = self.db["tokens"].get(token or "")
        issued = self.db["token_times"].get(token or "")
        if user_id and issued and issued < (_now() - timedelta(days=TOKEN_DAYS)).isoformat():
            return None  # expired: log in again (tokens without a time are the demo accounts' fixed ones)
        return self.db["users"].get(user_id) if user_id else None

    # -- rooms
    def _room_by_code(self, code: str) -> dict | None:
        code = normalize_code(code)
        return next((r for r in self.db["rooms"].values() if r["join_code"] == code), None)

    def _add_member(self, room_id: str, user_id: str) -> None:
        members = self.db["members"].setdefault(room_id, [])
        if user_id not in members:
            members.append(user_id)

    def _room_payload(self, room: dict) -> dict:
        ready = sum(1 for f in self.db["files"].get(room["room_id"], []) if f["status"] == "ready")
        return {"room_id": room["room_id"], "name": room["name"], "kind": "course", "files_ready": ready}

    def _room(self, room_id: str) -> dict:
        room = self.db["rooms"].get(room_id)
        if room is None:
            raise NotFound("Room not found")
        return room

    def room_for(self, user: dict, room_id: str) -> dict:
        """The room if this user teaches it or is a member, else 403/404."""
        room = self._room(room_id)
        if room["teacher_id"] == user["user_id"] or user["user_id"] in self.db["members"].get(room_id, []):
            return room
        raise Forbidden("You are not in this room")

    def _owned_room(self, user: dict, room_id: str) -> dict:
        room = self._room(room_id)
        if room["teacher_id"] != user["user_id"]:
            raise Forbidden("Only the room's teacher can do this")
        return room

    def create_room(self, user: dict, name: str) -> dict:
        if user["role"] != "teacher":
            raise Forbidden("Only teachers can create rooms")
        name = (name or "").strip()
        if not name:
            raise Invalid("Room name is required")
        with self._lock:
            code = _new_join_code()
            while self._room_by_code(code):
                code = _new_join_code()
            room = {"room_id": _new_id("r"), "name": name[:100], "teacher_id": user["user_id"],
                    "join_code": code, "created_at": _now().isoformat()}
            self.db["rooms"][room["room_id"]] = room
            self._save()
            return {"room": self._room_payload(room), "join_code": code}

    def rename_room(self, user: dict, room_id: str, name: str) -> dict:
        """Only the room's teacher. Students see the new name on their next page load (rooms are read, never copied)."""
        room = self._owned_room(user, room_id)
        name = (name or "").strip()
        if not name:
            raise Invalid("Room name is required")
        with self._lock:
            room["name"] = name[:100]
            self._save()
            return self._room_payload(room)

    def join_room(self, user: dict, code: str) -> dict:
        if user["role"] != "student":
            raise Forbidden("Only students join rooms with a code")
        with self._lock:
            room = self._room_by_code(code)
            if room is None:
                raise NotFound("Join code not found")
            self._add_member(room["room_id"], user["user_id"])
            self._save()
            return self._room_payload(room)

    def list_rooms(self, user: dict) -> dict:
        if user["role"] == "teacher":
            rooms = [r for r in self.db["rooms"].values() if r["teacher_id"] == user["user_id"]]
        else:
            rooms = [r for rid, r in self.db["rooms"].items() if user["user_id"] in self.db["members"].get(rid, [])]
        rooms.sort(key=lambda r: r["created_at"])
        return {"rooms": [self._room_payload(r) for r in rooms]}

    def join_code(self, user: dict, room_id: str) -> dict:
        return {"join_code": self._owned_room(user, room_id)["join_code"]}

    # -- files
    def add_file(self, user: dict, room_id: str, filename: str, data: bytes) -> dict:
        self._owned_room(user, room_id)
        safe = Path(filename or "file").name.strip() or "file"
        if not data:
            raise Invalid("The file is empty")
        with self._lock:
            file_id = _new_id("f")
            folder = self.dir / "rooms" / room_id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{file_id}__{safe}"
            path.write_bytes(data)
            # Status stays "uploaded" until M2's indexer reports progress. Never mark it ready here.
            record = {"file_id": file_id, "name": safe[:200], "source": "teacher", "status": "uploaded",
                      "path": str(path), "uploaded_at": _now().isoformat()}
            self.db["files"].setdefault(room_id, []).append(record)
            self._save()
        for hook in list(_UPLOAD_HOOKS):
            try:
                hook(room_id, file_id, path)
            except Exception as e:  # an indexer failure must not lose the upload
                self.set_file_status(room_id, file_id, "failed")
                record["error"] = str(e)[:200]
        return {"file_id": file_id, "status": record["status"]}

    def set_file_status(self, room_id: str, file_id: str, status: str) -> None:
        """For M2's indexer to report progress (uploaded -> indexing -> ready / failed)."""
        if status not in ("uploaded", "indexing", "ready", "failed"):
            raise Invalid("Unknown file status")
        with self._lock:
            for f in self.db["files"].get(room_id, []):
                if f["file_id"] == file_id:
                    f["status"] = status
                    self._save()
                    return
        raise NotFound("File not found")

    def delete_file(self, user: dict, room_id: str, file_id: str) -> None:
        """Only the room's teacher. Removes the record and the stored file, then tells the delete
        hooks (M2's index). A hook failure is logged in the record list, never undoes the delete."""
        self._owned_room(user, room_id)
        with self._lock:
            files = self.db["files"].get(room_id, [])
            record = next((f for f in files if f["file_id"] == file_id), None)
            if record is None:
                raise NotFound("File not found")
            files.remove(record)
            self._save()
        try:
            Path(record["path"]).unlink(missing_ok=True)
        except OSError:
            pass  # the record is gone; a locked file on Windows can be cleaned up later
        for hook in list(_DELETE_HOOKS):
            try:
                hook(room_id, file_id)
            except Exception:  # the file is deleted for the user either way
                pass

    def file_path(self, room_id: str, file_id: str) -> Path:
        for f in self.db["files"].get(room_id, []):
            if f["file_id"] == file_id:
                return Path(f["path"])
        raise NotFound("File not found")

    def read_file(self, user: dict, room_id: str, file_id: str) -> tuple[str, bytes]:
        """(name, bytes) of one material, for the room's teacher or any member (a student downloads the teacher's material)."""
        self.room_for(user, room_id)
        record = next((f for f in self.db["files"].get(room_id, []) if f["file_id"] == file_id), None)
        if record is None:
            raise NotFound("File not found")
        try:
            return record["name"], Path(record["path"]).read_bytes()
        except OSError:
            raise NotFound("The file is no longer stored") from None

    def list_files(self, user: dict, room_id: str) -> dict:
        self.room_for(user, room_id)
        files = self.db["files"].get(room_id, [])
        return {"files": [{k: f[k] for k in ("file_id", "name", "source", "status")} for f in files]}

    # -- students and activity
    def record_question(self, user_id: str, room_id: str) -> None:
        with self._lock:
            self.db["questions"].append({"user_id": user_id, "room_id": room_id, "at": _now().isoformat()})
            cutoff = (_now() - timedelta(days=30)).isoformat()
            self.db["questions"] = [q for q in self.db["questions"] if q["at"] >= cutoff]
            self._save()

    def list_students(self, user: dict, room_id: str) -> dict:
        self._owned_room(user, room_id)
        week_ago = (_now() - timedelta(days=7)).isoformat()
        counts: dict[str, int] = {}
        for q in self.db["questions"]:
            if q["room_id"] == room_id and q["at"] >= week_ago:
                counts[q["user_id"]] = counts.get(q["user_id"], 0) + 1
        students = []
        for uid in self.db["members"].get(room_id, []):
            u = self.db["users"].get(uid)
            if u:
                # mastery and status belong to M4. M1 does not invent them: null until M4 fills them.
                students.append({"user_id": uid, "name": u["name"], "questions_this_week": counts.get(uid, 0),
                                 "mastery": None, "status": None})
        return {"students": students}


# ---------------------------------------------------------------- default instance

_store: PlatformStore | None = None
_store_lock = threading.Lock()


def get_store() -> PlatformStore:
    global _store
    with _store_lock:
        if _store is None:
            data_dir = Path(os.environ.get("RAFEEQ_DATA_DIR", "storage")) / "platform"
            _store = PlatformStore(data_dir, seed_demo=os.environ.get("RAFEEQ_DEMO_ACCOUNTS", "1") != "0")
        return _store


def set_store(store: PlatformStore | None) -> None:
    """Tests use this to point M1 at a temporary folder."""
    global _store
    with _store_lock:
        _store = store
