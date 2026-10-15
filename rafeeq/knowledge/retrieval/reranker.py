"""Optional Cross-Encoder reranking (BAAI/bge-reranker-v2-m3) on top of an existing retriever.

    inner retriever (dense or hybrid) -> candidate pool (default 50) -> Cross-Encoder -> top k (default 6)

* Off by default. ``build_retriever`` only returns a ``RerankingRetriever`` when ``rerank=True``.
* The model is loaded lazily, once per ``CrossEncoderReranker`` instance; constructing it loads nothing.
* ``score`` of every returned chunk is still the true cosine from the inner retriever, so the Answerer's
  ``min_score`` gate keeps its meaning. The reranker score is added as ``rerank_score`` (the CrossEncoder's
  output under its default activation: only its ORDER is used; it is never compared with ``min_score``), plus ``pre_rerank_rank`` (1-based rank in the candidate list).
* Chunk ids, page, file name, topic id, text and room isolation come from the inner retriever untouched;
  this class only reorders and truncates what the inner retriever returned (it can never invent a chunk).
* Deterministic order: rerank_score desc, then the inner rank (earlier first), then chunk_id.
"""

from __future__ import annotations

from typing import Any, Sequence

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_POOL = 50
DEFAULT_RERANK_BATCH_SIZE = 16
DEFAULT_RERANK_MAX_LENGTH = 512
DEFAULT_RERANK_DEVICE = "auto"


def resolve_device(device: str | None) -> str:
    """``auto`` -> ``cuda`` when torch sees a GPU, otherwise ``cpu``. Anything else is returned unchanged."""
    if device in (None, "auto"):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # torch missing or broken: CPU path will report the real problem on load
            return "cpu"
    return device


def _positive_int(label: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


class CrossEncoderReranker:
    """Scores (query, passage) pairs. ``scorer`` may be injected (tests); otherwise sentence-transformers' CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = DEFAULT_RERANK_DEVICE,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        max_length: int = DEFAULT_RERANK_MAX_LENGTH,
        *,
        scorer: Any = None,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.batch_size = _positive_int("batch_size", batch_size)
        self.max_length = _positive_int("max_length", max_length)
        self.device: str | None = None  # resolved on load
        self.load_count = 0
        self.fallback_reason: str | None = None
        self._model = scorer  # an injected scorer counts as already loaded

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy: torch is only imported when really used

            self.device = resolve_device(self.requested_device)
            try:
                self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            except (RuntimeError, MemoryError) as exc:  # e.g. CUDA out of memory / no CUDA: fall back to CPU once
                if self.device == "cpu":
                    raise
                self.device = "cpu"
                self._model = CrossEncoder(self.model_name, device="cpu", max_length=self.max_length)
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
            self.load_count += 1
        return self._model

    def describe(self) -> dict[str, Any]:
        return {"model": self.model_name, "device": self.device or f"{self.requested_device} (not loaded yet)",
                "batch_size": self.batch_size, "max_length": self.max_length}

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        model = self.load()
        pairs = [[query, p] for p in passages]
        raw = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = [float(x) for x in raw]
        if len(scores) != len(passages):
            raise RuntimeError(f"reranker returned {len(scores)} scores for {len(passages)} passages")
        return scores


class RerankingRetriever:
    """``retrieve(room_id, query, k=6)``: take ``pool`` candidates from ``inner``, rerank, return the best ``k``."""

    def __init__(self, inner: Any, reranker: Any, *, pool: int = DEFAULT_RERANK_POOL) -> None:
        self._inner = inner
        self._reranker = reranker
        self._pool = _positive_int("pool", pool)

    @property
    def exclude_non_content(self) -> bool:
        return bool(getattr(self._inner, "exclude_non_content", False))

    def describe(self) -> dict[str, Any]:
        inner = self._inner.describe() if hasattr(self._inner, "describe") else {"mode": "dense"}
        return {"rerank": True, "pool": self._pool, "reranker": self._reranker.describe() if hasattr(self._reranker, "describe") else {}, "inner": inner}

    def retrieve(self, room_id: str, query: str, k: int = 6) -> list[dict[str, Any]]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        candidates = self._inner.retrieve(room_id, query, max(self._pool, k))  # inner validates room_id / query
        if not candidates:
            return []
        scores = self._reranker.score(query, [c["text"] for c in candidates])
        if len(scores) != len(candidates):
            raise RuntimeError("reranker score count does not match candidate count")
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i, candidates[i]["chunk_id"]))
        results = []
        for i in order[:k]:
            hit = dict(candidates[i])
            hit["rerank_score"] = scores[i]
            hit["pre_rerank_rank"] = i + 1
            results.append(hit)
        return results
