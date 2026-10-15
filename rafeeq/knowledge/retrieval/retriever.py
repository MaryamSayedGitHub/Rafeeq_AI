"""Room-isolated semantic retrieval.

    query -> embedder.embed_query -> vector_store.search(room_id=...) -> top k

The retriever only composes the embedder and the vector store; it adds no
ranking logic of its own, never invents results (everything returned comes
from the store), and refuses to return a chunk from another room.

One optional behaviour exists, off by default: ``exclude_non_content=True`` skips
index-like pages (see ``content_filter``) and fills the freed slots with the next
best chunks. It only removes candidates; scores and the order of the rest are untouched.

Arabic and English queries both work because the injected embedder
(BGE-M3) is multilingual; this module is language-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol

from .content_filter import filter_hits

DEFAULT_K = 6
_OVERFETCH = 3          # first candidate pool = max(k * _OVERFETCH, k + _MIN_EXTRA) when filtering
_MIN_EXTRA = 10
_MAX_POOL_DOUBLINGS = 5


class QueryEmbedderLike(Protocol):
    def embed_query(self, text: str) -> Any: ...


class SearchableStoreLike(Protocol):
    def search(self, query_embedding: Any, top_k: int = 5, room_id: str | None = None, **filters: Any) -> list[Any]: ...


class Retriever:
    """``retrieve(room_id, query, k=6)`` over an injected store and embedder."""

    def __init__(
        self,
        vector_store: SearchableStoreLike,
        embedder: QueryEmbedderLike,
        *,
        exclude_non_content: bool = False,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._exclude_non_content = bool(exclude_non_content)

    @property
    def exclude_non_content(self) -> bool:
        return self._exclude_non_content

    def _search(self, query_vector: Any, room_id: str, k: int) -> list[Any]:
        if not self._exclude_non_content:
            return self._store.search(query_vector, top_k=k, room_id=room_id)
        pool = max(k * _OVERFETCH, k + _MIN_EXTRA)
        for _ in range(_MAX_POOL_DOUBLINGS + 1):
            raw = self._store.search(query_vector, top_k=pool, room_id=room_id)
            kept = filter_hits(raw)
            if len(kept) >= k or len(raw) < pool:  # enough, or the room has nothing more to give
                return kept[:k]
            pool *= 2
        return kept[:k]

    def retrieve(self, room_id: str, query: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
        """Top ``k`` chunks of ``room_id`` for ``query``, best first.

        Each item has exactly: ``chunk_id, text, file_name, page, topic_id, score``
        (``score`` = cosine similarity). An unknown room or empty store gives ``[]``.
        """
        if not isinstance(room_id, str) or not room_id.strip():
            raise ValueError("room_id must be a non-empty string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        query_vector = self._embedder.embed_query(query)
        hits = self._search(query_vector, room_id, k)

        results: list[dict[str, Any]] = []
        for hit in hits:
            meta = hit.metadata
            if meta.get("room_id") != room_id:  # the store must never do this; fail loudly if it does
                raise RuntimeError(
                    f"room isolation violated: asked for {room_id!r}, store returned {meta.get('room_id')!r}"
                )
            results.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "text": hit.text,
                    "file_name": meta["file_name"],
                    "page": meta["page"],
                    "topic_id": meta.get("topic_id"),  # None only for chunks indexed without tagging
                    "score": float(hit.score),
                }
            )
        return results


# --------------------------------------------- module-level team contract
# M3/M4/M5 call ``retrieve(room_id, query, k=6)``. The app configures the
# retriever once at start-up; there is no hidden default and no model is
# loaded implicitly.

_default_retriever: Retriever | None = None


def set_default_retriever(retriever: Retriever | None) -> None:
    global _default_retriever
    _default_retriever = retriever


def retrieve(room_id: str, query: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    if _default_retriever is None:
        raise RuntimeError("no retriever configured: call set_default_retriever(Retriever(store, embedder)) first")
    return _default_retriever.retrieve(room_id, query, k)