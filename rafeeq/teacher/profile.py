"""The teacher profile: who the teacher is and the defaults the session form starts from.

Stored in the same LangGraph Store as the lecturer assistant's long-term memory, under
("lecturer", <teacher_id>, "profile"). So one teacher_id links the profile, what the assistant
remembers about the teacher, and every finished session.

Until M1's login exists, teacher_id is chosen once on the profile form (from the name) and
kept in a cookie. When login lands, teacher_id becomes the logged-in user's id.
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from rafeeq.teacher.lecturer.long_term_store import get_defaults, get_profile, list_sessions, put_profile
from rafeeq.teacher.lecturer.settings import AVATAR_DIR

Level = Literal["beginner", "intermediate", "advanced"]

LEVEL_AR = {"beginner": "مبتدئ", "intermediate": "متوسط", "advanced": "متقدم"}


def make_teacher_id(name: str) -> str:
    """'Dr. Sara Ahmed' -> 'dr-sara-ahmed'. Arabic letters are kept. Empty -> 'teacher'."""
    words = re.findall(r"[\w]+", name.lower(), flags=re.UNICODE)
    return "-".join(words)[:60] or "teacher"


class TeacherDefaults(BaseModel):
    language: str = Field(default="Arabic", max_length=40)
    student_level: Level = "intermediate"
    programming_language: str = Field(default="Python", max_length=40)
    duration_minutes: int = Field(default=60, ge=15, le=240)


class TeacherProfileUpdate(BaseModel):
    """What the teacher can edit (TeacherProfileUpdate in contracts/openapi.yaml)."""

    teacher_id: str = Field(default="", max_length=60)  # empty on first save: made from the name
    name: str = Field(min_length=2, max_length=120)
    title: str = Field(default="", max_length=40)        # e.g. "د." or "م."
    subjects: list[str] = Field(default_factory=list)
    bio: str = Field(default="", max_length=600)
    avatar_filename: str = Field(default="", max_length=120)  # saved under settings.AVATAR_DIR
    defaults: TeacherDefaults = Field(default_factory=TeacherDefaults)

    @field_validator("subjects", mode="before")
    @classmethod
    def subjects_as_list(cls, v):
        """The form sends one text box: 'Python, AI' or one per line."""
        if isinstance(v, str):
            v = re.split(r"[,\n،]", v)
        return [s.strip() for s in (v or []) if s and s.strip()][:12]


class TeacherProfile(TeacherProfileUpdate):
    """What the page and the API return (TeacherProfile in contracts/openapi.yaml)."""

    teacher_id: str
    display_name: str          # title + name, printed on every generated slide
    sessions_count: int = 0
    remembered: list[str] = Field(default_factory=list)  # what the assistant learned, shown openly


def display_name(title: str, name: str) -> str:
    return f"{title.strip()} {name.strip()}".strip()


def save_avatar(teacher_id: str, data: bytes, original_filename: str) -> str:
    """Saves a teacher's character image under settings.AVATAR_DIR as
    <teacher_id>.<ext>, replacing any previous one. Returns the saved filename,
    which is what Lab 5's video generation reuses as the default first-frame
    image when the teacher doesn't upload a different one for that request."""
    if not teacher_id or any(c in teacher_id for c in "/\\*?[]") or teacher_id.startswith("."):
        raise ValueError("invalid teacher_id")
    ext = Path(original_filename).suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    for old_file in AVATAR_DIR.glob(f"{teacher_id}.*"):
        old_file.unlink(missing_ok=True)
    path = AVATAR_DIR / f"{teacher_id}{ext}"
    path.write_bytes(data)
    return path.name


def load_profile(store, teacher_id: str, memory_lines: list[str] | None = None) -> TeacherProfile | None:
    """The full profile, or None if this teacher never saved one."""
    saved = get_profile(store, teacher_id)
    if not saved:
        return None
    data = TeacherProfileUpdate(**{**saved, "teacher_id": teacher_id})
    # The last approved session's choices win over the form defaults: that is what the
    # teacher actually used most recently.
    learned = {k: v for k, v in get_defaults(store, teacher_id).items() if v}
    try:
        defaults = TeacherDefaults(**{**data.defaults.model_dump(), **learned})
    except ValueError:
        defaults = data.defaults
    return TeacherProfile(
        **data.model_dump(exclude={"defaults"}),
        defaults=defaults,
        display_name=display_name(data.title, data.name),
        sessions_count=len(list_sessions(store, teacher_id)),
        remembered=memory_lines or [],
    )


def save_profile(store, update: TeacherProfileUpdate) -> str:
    """Save and return the teacher_id (new teachers get one made from their name).

    A new teacher never takes over an existing profile: 'sara' is taken -> 'sara-2'.
    """
    teacher_id = update.teacher_id
    if not teacher_id:
        base = make_teacher_id(update.name)
        teacher_id, n = base, 1
        while get_profile(store, teacher_id):
            n += 1
            teacher_id = f"{base}-{n}"
    data = update.model_dump(exclude={"teacher_id"})
    if not data["avatar_filename"]:  # managed by the avatar upload; keep the saved one
        data["avatar_filename"] = (get_profile(store, teacher_id) or {}).get("avatar_filename", "")
    put_profile(store, teacher_id, data)
    return teacher_id
