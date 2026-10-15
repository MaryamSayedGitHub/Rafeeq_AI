"""Presentation decks: the editable project behind a lesson's slides.

A deck is created from an approved Lesson Studio session (same slides, code examples and quiz) and
then edited in the Presentation Studio: slide text, a theme, a picture per slide from the library.
Exporting builds a new .pptx and saves it in the library as the NEXT VERSION of the previous file
(parent_asset_id), so nothing is overwritten.

Records are in the LangGraph store next to the profile and the assets:
    ("lecturer", <teacher_id>, "decks")      key = deck_id ("d_" + 8 hex)
"""

import re
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rafeeq.teacher import assets, themes
from rafeeq.teacher.lecturer.long_term_store import lecturer_key

MAX_SLIDES = 40
MAX_BULLETS = 8
DEFAULT_THEME_ID = "preset:rafeeq"


class DeckError(ValueError):
    """Bad deck input."""


def _ns(teacher_id: str) -> tuple:
    return ("lecturer", lecturer_key(teacher_id), "decks")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(value, limit: int) -> str:
    return str(value or "").strip()[:limit]


def clean_slide(raw: dict) -> dict:
    """One slide in the shape the deck builder expects (plus the picture reference)."""
    if not isinstance(raw, dict):
        raise DeckError("A slide must be an object")
    title = _text(raw.get("title"), 200)
    if not title:
        raise DeckError("Every slide needs a title")
    bullets = [_text(b, 300) for b in (raw.get("bullets") or []) if _text(b, 300)][:MAX_BULLETS]
    steps = [_text(s, 80) for s in (raw.get("diagram_steps") or []) if _text(s, 80)][:5]
    return {"title": title, "bullets": bullets, "notes": _text(raw.get("notes"), 2000), "diagram_steps": steps,
            "image_asset_id": _text(raw.get("image_asset_id"), 40)}


# ------------------------------------------------------------------ read
def _all(store, teacher_id: str) -> list[dict]:
    if store is None or not teacher_id:
        return []
    return [{"deck_id": i.key, **i.value} for i in store.search(_ns(teacher_id), limit=500)]


def get_deck(store, teacher_id: str, deck_id: str) -> dict | None:
    if store is None or not teacher_id or not deck_id:
        return None
    item = store.get(_ns(teacher_id), deck_id)
    return {"deck_id": deck_id, **item.value} if item else None


def list_decks(store, teacher_id: str, room_id: str = "") -> list[dict]:
    found = [d for d in _all(store, teacher_id) if not room_id or d.get("room_id") == room_id]
    return sorted(found, key=lambda d: (d.get("updated_at", ""), d["deck_id"]), reverse=True)


def find_by_session(store, teacher_id: str, session_id: str) -> dict | None:
    return next((d for d in _all(store, teacher_id) if session_id and d.get("session_id") == session_id), None)


# ------------------------------------------------------------------ write
def _session_pptx_asset(store, teacher_id: str, session_id: str) -> str:
    if not session_id:
        return ""
    found = assets.list_assets(store, teacher_id, kind="pptx", session_id=session_id)
    return found[-1]["asset_id"] if found else ""  # the oldest one is the original from the session


def create_deck(store, teacher_id: str, *, title: str, slides: list[dict], course_name: str = "", lecturer: str = "",
                room_id: str = "", session_id: str = "", code_examples: list | None = None, quiz: list | None = None,
                theme_id: str = DEFAULT_THEME_ID) -> dict:
    """Create a deck. For a session that already has one, that deck is returned (safe to repeat)."""
    if store is None or not teacher_id:
        raise DeckError("A teacher is required")
    existing = find_by_session(store, teacher_id, session_id)
    if existing:
        return existing
    cleaned = [clean_slide(s) for s in slides][:MAX_SLIDES]
    if not cleaned:
        raise DeckError("A deck needs at least one slide")
    deck_id = f"d_{secrets.token_hex(4)}"
    now = _now()
    record = {
        "title": _text(title, 200) or "Untitled", "course_name": _text(course_name, 200), "lecturer": _text(lecturer, 120),
        "room_id": room_id, "session_id": session_id, "theme_id": theme_id, "slides": cleaned,
        "code_examples": list(code_examples or []), "quiz": list(quiz or []),
        "pptx_asset_id": _session_pptx_asset(store, teacher_id, session_id), "exports": 0,
        "created_at": now, "updated_at": now,
    }
    store.put(_ns(teacher_id), deck_id, record)
    return {"deck_id": deck_id, **record}


def create_from_draft(store, teacher_id: str, session_id: str, draft: dict, brief: dict) -> dict | None:
    """Called when a session is approved. No slides in the draft = no deck."""
    if not draft.get("slides"):
        return None
    return create_deck(
        store, teacher_id, title=draft.get("topic") or brief.get("topic", ""), slides=draft["slides"],
        course_name=draft.get("course_name") or brief.get("course_name", ""),
        lecturer=brief.get("lecturer_name") or brief.get("lecturer_id", ""), room_id=brief.get("room_id", ""),
        session_id=session_id, code_examples=draft.get("code_examples"), quiz=draft.get("quiz"))


def resolve_theme(store, teacher_id: str, theme_id: str) -> dict:
    """The theme dict for a preset ('preset:teal') or a library theme asset id. Unknown = the default."""
    if theme_id.startswith("preset:") and theme_id[7:] in themes.PRESETS:
        return themes.normalize(themes.PRESETS[theme_id[7:]])
    asset = assets.get_asset(store, teacher_id, theme_id) if theme_id else None
    if asset and asset["kind"] == "theme":
        try:
            return themes.normalize(asset.get("settings") or {})
        except themes.ThemeError:
            pass
    return themes.normalize(themes.PRESETS["rafeeq"])


def update_deck(store, teacher_id: str, deck_id: str, *, title: str | None = None, slides: list | None = None,
                theme_id: str | None = None) -> dict | None:
    deck = get_deck(store, teacher_id, deck_id)
    if deck is None:
        return None
    if title is not None:
        deck["title"] = _text(title, 200) or deck["title"]
    if slides is not None:
        cleaned = [clean_slide(s) for s in slides][:MAX_SLIDES]
        if not cleaned:
            raise DeckError("A deck needs at least one slide")
        for slide in cleaned:  # a picture must be one of this teacher's images
            aid = slide["image_asset_id"]
            if aid:
                picture = assets.get_asset(store, teacher_id, aid)
                if picture is None or picture["kind"] != "image":
                    raise DeckError("A slide picture must be one of your images")
        deck["slides"] = cleaned
    if theme_id is not None:
        if not (theme_id.startswith("preset:") and theme_id[7:] in themes.PRESETS):
            asset = assets.get_asset(store, teacher_id, theme_id)
            if asset is None or asset["kind"] != "theme":
                raise DeckError("Unknown theme")
        deck["theme_id"] = theme_id
    deck["updated_at"] = _now()
    deck.pop("deck_id")
    store.put(_ns(teacher_id), deck_id, deck)
    return {"deck_id": deck_id, **deck}


def set_slide_image(store, teacher_id: str, deck_id: str, index: int, asset_id: str) -> dict | None:
    """Attach a library image to one slide ('' removes it)."""
    deck = get_deck(store, teacher_id, deck_id)
    if deck is None:
        return None
    if not 0 <= index < len(deck["slides"]):
        raise DeckError("No such slide")
    slides = [dict(s) for s in deck["slides"]]
    slides[index]["image_asset_id"] = asset_id
    return update_deck(store, teacher_id, deck_id, slides=slides)


def delete_deck(store, teacher_id: str, deck_id: str) -> bool:
    if get_deck(store, teacher_id, deck_id) is None:
        return False
    store.delete(_ns(teacher_id), deck_id)
    return True


# ------------------------------------------------------------------ export
def _slug(text: str) -> str:
    return re.sub(r"[^\w]+", "-", text.lower(), flags=re.UNICODE).strip("-")[:50] or "deck"


def export_deck(store, teacher_id: str, deck_id: str) -> dict | None:
    """Build the .pptx with the deck's theme and pictures and save it in the library as a new version."""
    from rafeeq.teacher.lecturer.mcp_servers.pptx_builder import build_presentation

    deck = get_deck(store, teacher_id, deck_id)
    if deck is None:
        return None
    theme = resolve_theme(store, teacher_id, deck.get("theme_id", DEFAULT_THEME_ID))
    slides = []
    for slide in deck["slides"]:
        item = {k: slide.get(k) for k in ("title", "bullets", "notes", "diagram_steps")}
        picture = assets.get_asset(store, teacher_id, slide["image_asset_id"]) if slide.get("image_asset_id") else None
        path = assets.asset_file(picture) if picture else None
        if path:
            item["image_path"] = str(path)
        slides.append(item)
    payload = {"title": deck["title"], "course_name": deck.get("course_name", ""), "lecturer": deck.get("lecturer", ""),
               "slides": slides, "code_examples": deck.get("code_examples", []), "quiz": deck.get("quiz", []),
               "theme": theme}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "deck.pptx"
        build_presentation(out, payload)
        data = out.read_bytes()

    parent = assets.get_asset(store, teacher_id, deck["pptx_asset_id"]) if deck.get("pptx_asset_id") else None
    record = assets.register_asset(
        store, teacher_id, kind="pptx", title=deck["title"], origin="edited", status="pending",
        room_id=deck.get("room_id", ""), session_id=deck.get("session_id", ""),
        parent_asset_id=parent["asset_id"] if parent else "", version=parent["version"] + 1 if parent else 1,
        settings_used={"deck_id": deck_id, "theme": theme["name"], "slides": len(slides)})
    root, ref = assets.save_file(teacher_id, record["asset_id"], data, f"{_slug(deck['title'])}-v{record['version']}.pptx")
    record = assets.update_asset(store, teacher_id, record["asset_id"], status="ready", storage_root=root, storage_ref=ref)

    deck.pop("deck_id")
    deck.update(pptx_asset_id=record["asset_id"], exports=deck.get("exports", 0) + 1, updated_at=_now())
    store.put(_ns(teacher_id), deck_id, deck)
    return record
