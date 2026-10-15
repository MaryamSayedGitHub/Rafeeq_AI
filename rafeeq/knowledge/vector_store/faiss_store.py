"""FAISS vector store: persistence + similarity search + metadata mapping.

Responsibilities (and nothing more)
-----------------------------------
This module receives embeddings that ALREADY exist. It never parses PDFs,
chunks text, imports the embedder, or loads any model.

Index choice: ``faiss.IndexFlatIP`` (exact inner-product search)
----------------------------------------------------------------
- The BGE-M3 embedder returns L2-normalised vectors, so inner product ==
  cosine similarity. ``score`` in results is therefore a cosine similarity
  in [-1, 1] (higher = more similar), NOT a FAISS distance.
- For a course-sized corpus (thousands to a few hundred thousand chunks) an
  exact flat index is fast enough, has no training step, never misses a
  neighbour, and is trivial to persist. IVF/HNSW/PQ are deliberately not used.
- Because the score only means "cosine" for unit vectors, ``add()`` and
  ``search()`` REJECT vectors that are not normalised instead of silently
  returning misleading scores.

FAISS ID -> metadata mapping
----------------------------
The store is append-only. FAISS assigns ids 0..n-1 in insertion order and
``self._records[i]`` is the metadata of FAISS id ``i``. Both are only ever
extended together, inside ``add()``, after all validation has passed, and are
persisted together (``index.faiss`` + ``metadata.json``), so the mapping is
stable across add / save / load / search. There is no delete or update; if
that is added later the mapping must be revisited (e.g. IndexIDMap).

Filtering
---------
FAISS knows nothing about rooms or documents. Filters (``room_id``,
``document_id``, ``document_type``) are applied here, after the FAISS search,
over a candidate pool that is grown until enough matching results are found or
the whole index has been examined. A filtered search can therefore never
return a chunk from another room, and never returns fewer results than exist.

Chunk fields
------------
Required (exactly what ``chunk_pages`` produces): ``chunk_id``, ``room_id``,
``file_name``, ``page``, ``block_type``, ``text``. Optional: ``document_id``
(defaults to ``file_name`` when absent) and ``document_type``. Any extra
JSON-serialisable fields are preserved untouched.

Logical identity of a chunk is ``(room_id, document_id, chunk_id)``; adding
the same identity twice is rejected (append-only, no silent duplicates).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import faiss
import numpy as np

FORMAT_VERSION = 1
INDEX_FILE_NAME = "index.faiss"
METADATA_FILE_NAME = "metadata.json"
NORM_TOLERANCE = 1e-3  # float32 model output is normalised to ~1e-6; this is generous

REQUIRED_CHUNK_FIELDS = ("chunk_id", "room_id", "file_name", "page", "block_type", "text")


class DuplicateChunkError(ValueError):
    """The same (room_id, document_id, chunk_id) was added twice."""


class VectorStoreLoadError(Exception):
    """Persisted vector store is missing pieces, corrupt, or inconsistent."""


@dataclass(frozen=True)
class SearchResult:
    """One search hit.

    ``score`` is cosine similarity (inner product of unit vectors).
    ``metadata`` holds every stored field except ``text``
    (room_id, document_id, file_name, page, chunk_id, block_type, ...).
    """

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- validation

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_non_empty_str(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")


def _prepare_record(chunk: Mapping[str, Any], position: int) -> dict[str, Any]:
    """Validate one chunk and return the record that will be stored."""
    if not isinstance(chunk, Mapping):
        raise TypeError(f"chunks[{position}] must be a dict, got {type(chunk).__name__}")
    missing = [name for name in REQUIRED_CHUNK_FIELDS if name not in chunk]
    if missing:
        raise ValueError(f"chunks[{position}] is missing required field(s): {', '.join(missing)}")

    record = copy.deepcopy(dict(chunk))
    for name in ("chunk_id", "room_id", "file_name", "block_type", "text"):
        _require_non_empty_str(record[name], f"chunks[{position}]['{name}']")
    if not _is_int(record["page"]) or record["page"] < 1:
        raise ValueError(f"chunks[{position}]['page'] must be an integer >= 1, got {record['page']!r}")

    document_id = record.get("document_id")
    if document_id is None:
        document_id = record["file_name"]  # documented default
    _require_non_empty_str(document_id, f"chunks[{position}]['document_id']")
    record["document_id"] = document_id

    if record.get("document_type") is not None:
        _require_non_empty_str(record["document_type"], f"chunks[{position}]['document_type']")
    return record


def _identity(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (record["room_id"], record["document_id"], record["chunk_id"])


def _check_normalised(vectors: np.ndarray, label: str) -> None:
    if not np.isfinite(vectors).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=-1)
    if not np.allclose(norms, 1.0, atol=NORM_TOLERANCE):
        raise ValueError(
            f"{label} must be L2-normalised (unit length) so inner product equals cosine "
            f"similarity; found norms between {float(norms.min()):.4f} and {float(norms.max()):.4f}"
        )


def _as_float32(array: Any, label: str) -> np.ndarray:
    try:
        return np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric array: {exc}") from exc


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------- store

class FAISSVectorStore:
    """Exact vector store with room / document filtering. Chunks are added in bulk and removed
    one whole document at a time (when a teacher deletes a file)."""

    def __init__(self, dimension: int) -> None:
        if not _is_int(dimension) or dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {dimension!r}")
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._records: list[dict[str, Any]] = []  # _records[i] <-> FAISS id i
        self._keys: set[tuple[str, str, str]] = set()

    # ----- basic properties

    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self) -> int:
        return self._index.ntotal

    # ----- add

    def add(self, embeddings: np.ndarray, chunks: Sequence[Mapping[str, Any]]) -> int:
        """Append ``embeddings[i]`` with ``chunks[i]``; return the number added.

        All-or-nothing: if anything is invalid (shape, dimension, length
        mismatch, bad metadata, non-normalised vectors, duplicates) nothing
        is added and an explicit exception is raised.
        """
        vectors = _as_float32(embeddings, "embeddings")
        if vectors.ndim != 2:
            raise ValueError(
                f"embeddings must be a 2D array (n, dimension), got {vectors.ndim}D with shape "
                f"{vectors.shape}; use embeddings.reshape(1, -1) for a single vector"
            )
        if isinstance(chunks, (str, bytes, Mapping)):
            raise TypeError("chunks must be a list of chunk dicts")
        chunks = list(chunks)
        if vectors.shape[0] == 0 or not chunks:
            raise ValueError("embeddings and chunks must not be empty")
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"number of embeddings ({vectors.shape[0]}) does not match number of chunks "
                f"({len(chunks)}); refusing to truncate"
            )
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"embedding dimension {vectors.shape[1]} does not match store dimension {self._dimension}"
            )
        _check_normalised(vectors, "embeddings")

        records = [_prepare_record(chunk, i) for i, chunk in enumerate(chunks)]

        new_keys: set[tuple[str, str, str]] = set()
        for i, record in enumerate(records):
            key = _identity(record)
            if key in self._keys or key in new_keys:
                raise DuplicateChunkError(
                    f"duplicate chunk (room_id={key[0]!r}, document_id={key[1]!r}, "
                    f"chunk_id={key[2]!r}) at chunks[{i}]; the store is append-only"
                )
            new_keys.add(key)

        # Everything is valid: mutate index and mapping together.
        self._index.add(vectors)
        self._records.extend(records)
        self._keys |= new_keys
        return len(records)

    # ----- remove

    def remove_document(self, room_id: str, document_id: str) -> int:
        """Remove every chunk of one document in one room; return how many were removed.

        IndexFlatIP keeps vectors in insertion order, so the store is rebuilt from the kept
        vectors. That is exact and fast enough for course-sized stores. Unknown document: 0.
        """
        _require_non_empty_str(room_id, "room_id")
        _require_non_empty_str(document_id, "document_id")
        keep = [i for i, r in enumerate(self._records)
                if not (r["room_id"] == room_id and r["document_id"] == document_id)]
        removed = len(self._records) - len(keep)
        if removed == 0:
            return 0
        index = faiss.IndexFlatIP(self._dimension)
        if keep:
            vectors = np.vstack([self._index.reconstruct(i) for i in keep]).astype("float32")
            index.add(vectors)
        records = [self._records[i] for i in keep]
        self._index, self._records = index, records
        self._keys = {_identity(r) for r in records}
        self._row_cache = None
        return removed

    # ----- search

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        room_id: str | None = None,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> list[SearchResult]:
        """Return up to ``top_k`` most similar chunks, best first.

        Filters are ANDed; ``None`` means "no filter". An empty store, or a
        filter matching nothing (e.g. an unknown room), returns ``[]``.
        """
        if not _is_int(top_k) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        for label, value in (
            ("room_id", room_id), ("document_id", document_id), ("document_type", document_type),
        ):
            if value is not None:
                _require_non_empty_str(value, label)

        query = _as_float32(query_embedding, "query_embedding")
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1:
            raise ValueError(f"query_embedding must be a 1D vector, got shape {query.shape}")
        if query.shape[0] != self._dimension:
            raise ValueError(
                f"query dimension {query.shape[0]} does not match store dimension {self._dimension}"
            )
        _check_normalised(query, "query_embedding")

        total = self._index.ntotal
        if total == 0:
            return []

        wanted = {"room_id": room_id, "document_id": document_id, "document_type": document_type}
        wanted = {name: value for name, value in wanted.items() if value is not None}

        # Grow the candidate pool until enough matches are found or the whole
        # index has been examined. FAISS returns hits sorted by score, so the
        # first top_k matches inside the pool are the true top_k matches.
        pool = min(total, max(top_k * 10, top_k)) if wanted else min(total, top_k)
        query_matrix = query.reshape(1, -1)
        while True:
            scores, ids = self._index.search(query_matrix, pool)
            hits = [
                (float(score), int(idx))
                for score, idx in zip(scores[0], ids[0])
                if idx >= 0 and self._matches(self._records[int(idx)], wanted)
            ]
            if len(hits) >= top_k or pool >= total:
                break
            pool = min(total, pool * 2)

        return [self._to_result(score, idx) for score, idx in hits[:top_k]]

    def room_chunks(self, room_id: str) -> list[dict[str, Any]]:
        """Every stored chunk of ``room_id`` (all fields including ``text``), in insertion order.

        Read-only helper for lexical indexes: the returned dicts are copies, an unknown room gives ``[]``.
        """
        _require_non_empty_str(room_id, "room_id")
        return [copy.deepcopy(record) for record in self._records if record["room_id"] == room_id]

    def similarities(
        self,
        query_embedding: np.ndarray,
        room_id: str,
        keys: Sequence[tuple[str, str]],
    ) -> dict[tuple[str, str], float]:
        """Exact cosine similarity between the query and chunks of ``room_id``.

        ``keys`` are ``(document_id, chunk_id)`` pairs. The result maps each key that exists in the room to the
        same value ``search()`` would report for it (inner product of unit vectors); unknown keys are omitted.
        Used by hybrid retrieval to keep the true cosine for candidates the dense search did not return.
        """
        _require_non_empty_str(room_id, "room_id")
        query = _as_float32(query_embedding, "query_embedding")
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1 or query.shape[0] != self._dimension:
            raise ValueError(f"query_embedding must be a vector of dimension {self._dimension}, got shape {query.shape}")
        _check_normalised(query, "query_embedding")
        rows = self._rows_by_identity()
        out: dict[tuple[str, str], float] = {}
        for document_id, chunk_id in keys:
            row = rows.get((room_id, document_id, chunk_id))
            if row is not None:
                out[(document_id, chunk_id)] = float(np.dot(self._index.reconstruct(row), query))
        return out

    def _rows_by_identity(self) -> dict[tuple[str, str, str], int]:
        cached = getattr(self, "_row_cache", None)
        if cached is None or cached[0] != len(self._records):
            cached = (len(self._records), {_identity(record): i for i, record in enumerate(self._records)})
            self._row_cache = cached
        return cached[1]

    @staticmethod
    def _matches(record: Mapping[str, Any], wanted: Mapping[str, str]) -> bool:
        return all(record.get(name) == value for name, value in wanted.items())

    def _to_result(self, score: float, idx: int) -> SearchResult:
        record = self._records[idx]
        metadata = copy.deepcopy({k: v for k, v in record.items() if k != "text"})
        return SearchResult(text=record["text"], score=score, metadata=metadata)

    # ----- persistence

    def save(self, directory: str | os.PathLike[str]) -> None:
        """Write ``index.faiss`` and ``metadata.json`` into ``directory``.

        The folder is created if needed. Each file is written to a temporary
        name and then renamed, so a crash cannot leave a half-written file.
        Extra chunk fields must be JSON-serialisable.
        """
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": FORMAT_VERSION,
            "index_type": "IndexFlatIP",
            "metric": "inner_product (cosine for normalised vectors)",
            "dimension": self._dimension,
            "count": len(self._records),
            "records": self._records,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)  # fails before any file is touched

        index_path = folder / INDEX_FILE_NAME
        tmp_index = folder / (INDEX_FILE_NAME + ".tmp")
        faiss.write_index(self._index, str(tmp_index))
        os.replace(tmp_index, index_path)
        _atomic_write_text(folder / METADATA_FILE_NAME, text)

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> "FAISSVectorStore":
        """Rebuild a store from ``save()`` output. Never re-embeds anything."""
        folder = Path(directory)
        index_path = folder / INDEX_FILE_NAME
        meta_path = folder / METADATA_FILE_NAME
        for path in (index_path, meta_path):
            if not path.is_file():
                raise FileNotFoundError(f"vector store file not found: {path}")

        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VectorStoreLoadError(f"{meta_path} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise VectorStoreLoadError(f"{meta_path} must contain a JSON object")
        if payload.get("format_version") != FORMAT_VERSION:
            raise VectorStoreLoadError(
                f"unsupported metadata format_version {payload.get('format_version')!r} "
                f"(expected {FORMAT_VERSION})"
            )
        dimension = payload.get("dimension")
        records = payload.get("records")
        if not _is_int(dimension) or dimension <= 0:
            raise VectorStoreLoadError(f"{meta_path} has an invalid 'dimension': {dimension!r}")
        if not isinstance(records, list):
            raise VectorStoreLoadError(f"{meta_path} is missing the 'records' list")

        try:
            index = faiss.read_index(str(index_path))
        except RuntimeError as exc:
            raise VectorStoreLoadError(f"could not read FAISS index {index_path}: {exc}") from exc
        if index.d != dimension:
            raise VectorStoreLoadError(
                f"index dimension {index.d} does not match metadata dimension {dimension}"
            )
        if index.ntotal != len(records):
            raise VectorStoreLoadError(
                f"index holds {index.ntotal} vectors but metadata has {len(records)} records; "
                "the two files are out of sync"
            )

        store = cls(dimension)
        store._index = index
        try:
            prepared = [_prepare_record(record, i) for i, record in enumerate(records)]
        except (TypeError, ValueError) as exc:
            raise VectorStoreLoadError(f"invalid record in {meta_path}: {exc}") from exc
        keys = {_identity(record) for record in prepared}
        if len(keys) != len(prepared):
            raise VectorStoreLoadError(f"{meta_path} contains duplicate chunk identities")
        store._records = prepared
        store._keys = keys
        return store