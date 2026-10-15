"""Safe load / create / save of the persisted vector store.

The store itself (``FAISSVectorStore``) knows how to ``save()`` and ``load()``
two files, ``index.faiss`` and ``metadata.json``. This module adds the
operational rules the ingestion and query scripts need on top of that, WITHOUT
changing the store:

- **Missing vs. broken.** No files at all means "no store yet" and a new empty
  store may be created. ONE file missing, unreadable/corrupt files, files that
  disagree with each other, or a path that is not a directory are errors. In
  those cases a fresh empty store is never created, because saving it would
  silently overwrite whatever is left of the real one.
- **Safe save.** ``store.save()`` replaces ``index.faiss`` first and
  ``metadata.json`` second, so a failure in between would leave a store that
  cannot be loaded. ``save_store_safely`` backs up the two existing files,
  saves, reloads the result from disk to prove it is intact, and restores the
  backups if anything fails. Success is only reported after the reload check.
- **Duplicate pre-check.** ``document_exists`` lets a caller detect an already
  ingested document BEFORE spending minutes embedding it. The store's own
  ``(room_id, document_id, chunk_id)`` duplicate rule remains the final
  authority.

Nothing here imports FAISS or NumPy at import time: the store class is
injectable (``store_cls=``), so the rules can be tested without either.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

# Must match rafeeq.knowledge.vector_store.faiss_store (a test guards against drift).
INDEX_FILE_NAME = "index.faiss"
METADATA_FILE_NAME = "metadata.json"
BACKUP_SUFFIX = ".bak"
TEMP_SUFFIX = ".tmp"

# Where the scripts keep the store by default (relative to the project root).
DEFAULT_STORE_SUBDIR = Path("data") / "vector_store"
DEFAULT_RAW_PDF_SUBPATH = Path("data") / "raw" / "hands_on_llm.pdf"
DEFAULT_ROOM_ID = "llm_course"


class PersistenceError(RuntimeError):
    """The persisted store is missing pieces, corrupt, incompatible, or could not be saved."""


class StoreNotFoundError(PersistenceError):
    """There is no persisted store in the directory (neither file exists)."""


class DocumentAlreadyIngestedError(ValueError):
    """The document is already present in the room (the store is append-only)."""


def _default_store_cls() -> Any:
    from rafeeq.knowledge.vector_store import FAISSVectorStore  # lazy: needs faiss + numpy

    return FAISSVectorStore


def _paths(directory: str | Path) -> tuple[Path, Path, Path]:
    folder = Path(directory)
    return folder, folder / INDEX_FILE_NAME, folder / METADATA_FILE_NAME


def find_leftover_files(directory: str | Path) -> list[Path]:
    """Backup / temp files that only exist if a save was interrupted."""
    folder = Path(directory)
    found = []
    for name in (INDEX_FILE_NAME, METADATA_FILE_NAME):
        for suffix in (BACKUP_SUFFIX, TEMP_SUFFIX):
            path = folder / (name + suffix)
            if path.exists():
                found.append(path)
    return found


def _leftover_hint(directory: Path) -> str:
    leftovers = find_leftover_files(directory)
    if not leftovers:
        return ""
    names = ", ".join(p.name for p in leftovers)
    return (
        f" Files from an interrupted save exist ({names}); if a '{BACKUP_SUFFIX}' file is present, "
        f"copying it over the matching file (without the '{BACKUP_SUFFIX}' suffix) restores the previous store."
    )


# ------------------------------------------------------------------ loading

def load_existing_store(directory: str | Path, *, store_cls: Any = None) -> Any:
    """Load the store saved in ``directory``; never creates one.

    Raises:
        StoreNotFoundError: neither file exists.
        PersistenceError: one file is missing, the path is not a directory, or
            the files cannot be loaded (corrupt / out of sync / wrong format).
    """
    folder, index_path, meta_path = _paths(directory)
    if folder.exists() and not folder.is_dir():
        raise PersistenceError(f"{folder} exists but is not a directory")

    have_index, have_meta = index_path.exists(), meta_path.exists()
    if not have_index and not have_meta:
        raise StoreNotFoundError(f"no vector store found in {folder} ({INDEX_FILE_NAME} and {METADATA_FILE_NAME} are both missing)")
    if have_index != have_meta:
        present, missing = (index_path, meta_path) if have_index else (meta_path, index_path)
        raise PersistenceError(
            f"the vector store in {folder} is incomplete: {present.name} exists but {missing.name} is missing. "
            f"Nothing was changed and no empty store was created, because saving one would overwrite "
            f"what is left. Restore {missing.name} from a backup, or move this directory away and re-ingest."
            + _leftover_hint(folder)
        )

    store_cls = store_cls or _default_store_cls()
    try:
        return store_cls.load(folder)
    except Exception as exc:  # re-raised below with context; never swallowed
        raise PersistenceError(
            f"the vector store in {folder} could not be loaded and looks corrupt or out of sync: "
            f"{type(exc).__name__}: {exc}. Nothing was changed and no empty store was created. "
            f"Restore both files from a backup, or move this directory away and re-ingest."
            + _leftover_hint(folder)
        ) from exc


def load_or_create_store(
    directory: str | Path,
    dimension: int | Callable[[], int],
    *,
    store_cls: Any = None,
) -> tuple[Any, bool]:
    """Return ``(store, created)``.

    An existing complete store is loaded (``created=False``). A new empty store
    (``created=True``) is returned ONLY when the directory has no store files at
    all. ``dimension`` may be a callable so the embedding model is only loaded
    when a new store is actually needed.
    """
    store_cls = store_cls or _default_store_cls()
    try:
        return load_existing_store(directory, store_cls=store_cls), False
    except StoreNotFoundError:
        size = dimension() if callable(dimension) else dimension
        return store_cls(size), True


def ensure_dimension(store: Any, expected: int, directory: str | Path) -> None:
    """The store and the embedder must agree, or scores/inserts are meaningless."""
    if store.dimension != expected:
        raise PersistenceError(
            f"the store in {directory} has dimension {store.dimension} but the embedding model produces "
            f"{expected}. It was built with a different model; use another --store-dir or re-ingest everything."
        )


# ---------------------------------------------------------------- inspection

def document_exists(store: Any, room_id: str, document_id: str) -> bool:
    """True if the room already holds any chunk of ``document_id`` (public API only)."""
    if len(store) == 0:
        return False
    probe = [1.0] + [0.0] * (store.dimension - 1)  # any unit vector: we only care whether a hit exists
    return bool(store.search(probe, top_k=1, room_id=room_id, document_id=document_id))


def store_inventory(directory: str | Path) -> dict[str, dict[str, int]]:
    """``{room_id: {document_id: chunk_count}}`` read from the saved ``metadata.json``."""
    _, _, meta_path = _paths(directory)
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PersistenceError(f"cannot read {meta_path}: {exc}") from exc
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise PersistenceError(f"{meta_path} has no 'records' list")

    rooms: dict[str, dict[str, int]] = {}
    for record in records:
        if not isinstance(record, dict) or "room_id" not in record:
            raise PersistenceError(f"{meta_path} contains a record without a room_id")
        document = record.get("document_id") or record.get("file_name") or "?"
        docs = rooms.setdefault(record["room_id"], {})
        docs[document] = docs.get(document, 0) + 1
    return {room: dict(sorted(docs.items())) for room, docs in sorted(rooms.items())}


# -------------------------------------------------------------------- saving

def _rollback(
    folder: Path,
    targets: list[Path],
    existed: list[bool],
    backups: list[tuple[Path, Path]],
    folder_existed: bool,
) -> str:
    """Undo a failed save. Returns a sentence describing the outcome."""
    try:
        for backup, original in backups:
            shutil.copy2(backup, original)  # copy (not move): the backup survives until restore is proven
        for target, was_there in zip(targets, existed):
            if not was_there:
                target.unlink(missing_ok=True)  # only files this save created
        for name in (INDEX_FILE_NAME, METADATA_FILE_NAME):
            (folder / (name + TEMP_SUFFIX)).unlink(missing_ok=True)
        for backup, _ in backups:
            backup.unlink(missing_ok=True)
        if not folder_existed:
            try:
                folder.rmdir()
            except OSError:
                pass
    except OSError as exc:
        names = ", ".join(str(b) for b, _ in backups) or "none"
        raise PersistenceError(
            f"saving failed AND restoring the previous files failed ({exc}). "
            f"Backups of the previous store: {names}. Copy them over the matching files to recover."
        ) from exc
    return "The previous store files were restored." if backups else "No store was written."


def save_store_safely(store: Any, directory: str | Path, *, store_cls: Any = None) -> None:
    """Save ``store`` to ``directory`` and prove it can be loaded back.

    On any failure (save error, or the reloaded store does not match) the
    previous files are restored and ``PersistenceError`` is raised. On success
    the directory holds exactly the files of ``store`` and no backups.
    """
    store_cls = store_cls or _default_store_cls()
    folder, index_path, meta_path = _paths(directory)
    targets = [index_path, meta_path]
    existed = [p.exists() for p in targets]
    folder_existed = folder.exists()
    if folder_existed and not folder.is_dir():
        raise PersistenceError(f"{folder} exists but is not a directory")
    if any(existed) and not all(existed):
        raise PersistenceError(
            f"refusing to save: the store in {folder} is incomplete (only one of {INDEX_FILE_NAME} / "
            f"{METADATA_FILE_NAME} exists). Nothing was changed."
        )

    backups: list[tuple[Path, Path]] = []
    try:
        if all(existed):
            for target in targets:
                backup = target.with_name(target.name + BACKUP_SUFFIX)
                shutil.copy2(target, backup)
                backups.append((backup, target))
        store.save(folder)
        reloaded = store_cls.load(folder)
        if len(reloaded) != len(store):
            raise PersistenceError(
                f"verification failed: {len(store)} vectors were in memory but {len(reloaded)} were read back from {folder}"
            )
    except BaseException as exc:
        outcome = _rollback(folder, targets, existed, backups, folder_existed)
        if not isinstance(exc, Exception):
            raise  # KeyboardInterrupt etc.: files restored, interruption continues
        raise PersistenceError(f"saving the vector store to {folder} failed: {type(exc).__name__}: {exc}. {outcome}") from exc

    for backup, _ in backups:
        backup.unlink(missing_ok=True)