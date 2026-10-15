"""Hybrid retrieval: dense (cosine) + in-project BM25, combined with Reciprocal Rank Fusion.

    query -> embed_query -> dense candidates  (the existing Retriever path, cosine order)
          -> BM25 over the room's chunks      (lexical order)
          -> RRF:  fusion(d) = 1/(k + rank_dense(d)) + 1/(k + rank_lexical(d))      k = 60, equal weights
          -> top k by fusion score

Design decisions
----------------
* ``Retriever`` (dense) is untouched and stays the default; this is a separate class with the same
  ``retrieve(room_id, query, k=6)`` signature, so it can be passed anywhere a retriever is accepted.
* ``score`` in every result is the TRUE cosine similarity between the query and that chunk, exactly what
  dense search reports. Candidates found only by BM25 get their exact cosine from the store. The ranking
  uses ``fusion_score``, a separate number; it is never written into ``score`` and it is not a probability.
  Anything that gates on ``score`` (the Answerer's ``min_score``) therefore keeps its meaning.
* A document missing from one list simply gets no contribution from it. With no lexical match at all
  (e.g. an Arabic question over an English book) the fused order equals the dense order.
* Both candidate lists have depth ``candidate_pool`` (default 50). RRF constant k = 60, weights equal.
  These are fixed design values, not tuned on the benchmark.
* ``exclude_non_content=True`` removes index-like pages from BOTH lists before ranks are assigned
  (same detector as the dense retriever), so an index page can never earn a fusion score.
* Room isolation: dense hits come from a room-filtered store search, the lexical index is built only from
  ``store.room_chunks(room_id)``, and every dense hit is re-checked against the requested room.
* Ties in fusion score are broken by higher cosine, then by chunk_id, so results are deterministic.
* The BM25 index of a room is built on first use and rebuilt when the store has grown.
"""

from __future__ import annotations

from typing import Any, Hashable, Sequence

from .bm25 import BM25Index, tokenize
from .content_filter import is_non_content_page
from .retriever import DEFAULT_K, Retriever

RRF_K = 60
DEFAULT_CANDIDATE_POOL = 50


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> dict[Hashable, float]:
    """RRF score of every item over several ranked lists (best first), each list weighted equally.

    ``score(item) = sum over lists of 1 / (k + rank)``, rank starting at 1. An item that appears twice in
    one list counts once, at its best rank.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        seen: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class _RoomLexicon:
    """BM25 index over the chunks of one room (plus the index-page flags used by the exclusion option)."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.index = BM25Index([tokenize(chunk["text"]) for chunk in chunks])
        self._non_content: list[bool] | None = None

    def non_content(self) -> list[bool]:
        if self._non_content is None:
            self._non_content = [is_non_content_page(chunk["text"]) for chunk in self.chunks]
        return self._non_content

    def rank(self, query: str, depth: int, exclude_non_content: bool) -> list[tuple[dict[str, Any], float]]:
        matches = self.index.top(tokenize(query))
        if exclude_non_content:
            flags = self.non_content()
            matches = [(position, score) for position, score in matches if not flags[position]]
        return [(self.chunks[position], score) for position, score in matches[:depth]]


class HybridRetriever:
    """``retrieve(room_id, query, k=6)`` with dense + BM25 candidates fused by RRF (see module docstring)."""

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        *,
        exclude_non_content: bool = False,
        rrf_k: int = RRF_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> None:
        for name in ("room_chunks", "similarities"):
            if not callable(getattr(vector_store, name, None)):
                raise TypeError(f"HybridRetriever needs a vector store with {name}() (FAISSVectorStore has it)")
        for label, value in (("rrf_k", rrf_k), ("candidate_pool", candidate_pool)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer, got {value!r}")
        self._store = vector_store
        self._embedder = embedder
        self._exclude_non_content = bool(exclude_non_content)
        self._rrf_k = rrf_k
        self._pool = candidate_pool
        self._dense = Retriever(vector_store, embedder, exclude_non_content=self._exclude_non_content)
        self._lexicons: dict[str, tuple[int, _RoomLexicon]] = {}

    @property
    def exclude_non_content(self) -> bool:
        return self._exclude_non_content

    def describe(self) -> dict[str, Any]:
        return {"mode": "hybrid", "fusion": "rrf", "rrf_k": self._rrf_k, "weights": "equal", "candidate_pool": self._pool,
                "lexical": "bm25 (k1=1.5, b=0.75)", "exclude_non_content": self._exclude_non_content}

    def _lexicon(self, room_id: str) -> _RoomLexicon:
        size = len(self._store)
        cached = self._lexicons.get(room_id)
        if cached is None or cached[0] != size:
            cached = (size, _RoomLexicon(self._store.room_chunks(room_id)))
            self._lexicons[room_id] = cached
        return cached[1]

    def retrieve(self, room_id: str, query: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
        """Top ``k`` chunks of ``room_id`` by fusion score, best first.

        Each item has the dense contract's keys (``chunk_id, text, file_name, page, topic_id, score``) where
        ``score`` is the true cosine similarity, plus ``fusion_score`` (the ranking value), ``dense_rank`` and
        ``lexical_rank`` (1-based position in each candidate list, ``None`` if absent) and ``lexical_score``
        (BM25, ``None`` if absent).
        """
        if not isinstance(room_id, str) or not room_id.strip():
            raise ValueError("room_id must be a non-empty string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        depth = max(self._pool, k)
        query_vector = self._embedder.embed_query(query)

        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        dense_keys: list[tuple[str, str]] = []
        for rank, hit in enumerate(self._dense._search(query_vector, room_id, depth), start=1):
            meta = hit.metadata
            if meta.get("room_id") != room_id:  # the store must never do this; fail loudly if it does
                raise RuntimeError(f"room isolation violated: asked for {room_id!r}, store returned {meta.get('room_id')!r}")
            key = (meta["document_id"], meta["chunk_id"])
            dense_keys.append(key)
            candidates[key] = {"meta": meta, "text": hit.text, "cosine": float(hit.score), "dense_rank": rank,
                               "lexical_rank": None, "lexical_score": None}

        lexical_keys: list[tuple[str, str]] = []
        for rank, (chunk, bm25_score) in enumerate(self._lexicon(room_id).rank(query, depth, self._exclude_non_content), start=1):
            if chunk.get("room_id") != room_id:
                raise RuntimeError(f"room isolation violated: asked for {room_id!r}, lexical index returned {chunk.get('room_id')!r}")
            key = (chunk["document_id"], chunk["chunk_id"])
            lexical_keys.append(key)
            entry = candidates.get(key)
            if entry is None:
                entry = candidates[key] = {"meta": chunk, "text": chunk["text"], "cosine": None, "dense_rank": None,
                                           "lexical_rank": None, "lexical_score": None}
            entry["lexical_rank"], entry["lexical_score"] = rank, float(bm25_score)

        lexical_only = [key for key, entry in candidates.items() if entry["cosine"] is None]
        if lexical_only:  # the dense search never returned these: ask the store for their exact cosine
            cosines = self._store.similarities(query_vector, room_id, lexical_only)
            for key in lexical_only:
                if key not in cosines:
                    raise RuntimeError(f"store has no vector for lexical candidate {key!r}")
                candidates[key]["cosine"] = cosines[key]

        fusion = reciprocal_rank_fusion([dense_keys, lexical_keys], self._rrf_k)
        ordered = sorted(candidates, key=lambda key: (-fusion[key], -candidates[key]["cosine"], key[1]))

        results: list[dict[str, Any]] = []
        for key in ordered[:k]:
            entry, meta = candidates[key], candidates[key]["meta"]
            results.append({
                "chunk_id": meta["chunk_id"],
                "text": entry["text"],
                "file_name": meta["file_name"],
                "page": meta["page"],
                "topic_id": meta.get("topic_id"),
                "score": float(entry["cosine"]),          # TRUE cosine, never the fusion value
                "fusion_score": fusion[key],
                "dense_rank": entry["dense_rank"],
                "lexical_rank": entry["lexical_rank"],
                "lexical_score": entry["lexical_score"],
            })
        return results
