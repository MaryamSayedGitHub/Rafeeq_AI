"""One switch between the two retrieval implementations. Dense is the default."""

from __future__ import annotations

from typing import Any

from .hybrid_retriever import DEFAULT_CANDIDATE_POOL, RRF_K, HybridRetriever
from .reranker import DEFAULT_RERANK_POOL, RerankingRetriever
from .retriever import Retriever

RETRIEVAL_MODES = ("dense", "hybrid")
DEFAULT_RETRIEVAL_MODE = "dense"


def build_retriever(
    vector_store: Any,
    embedder: Any,
    *,
    mode: str = DEFAULT_RETRIEVAL_MODE,
    exclude_non_content: bool = False,
    rrf_k: int = RRF_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    rerank: bool = False,
    reranker: Any = None,
    rerank_pool: int = DEFAULT_RERANK_POOL,
) -> Retriever | HybridRetriever | RerankingRetriever:
    """``mode="dense"`` -> ``Retriever`` (unchanged behaviour); ``mode="hybrid"`` -> ``HybridRetriever``.

    ``rrf_k`` and ``candidate_pool`` only apply to hybrid mode and are rejected otherwise if changed from
    their defaults, so a typo in a configuration cannot silently do nothing.
    """
    if not rerank:
        if reranker is not None or rerank_pool != DEFAULT_RERANK_POOL:
            raise ValueError("reranker and rerank_pool only apply with rerank=True")
        return _build_base(vector_store, embedder, mode, exclude_non_content, rrf_k, candidate_pool)
    if reranker is None:
        from .reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()  # lazy: loads nothing until the first query
    inner = _build_base(vector_store, embedder, mode, exclude_non_content, rrf_k, candidate_pool)
    return RerankingRetriever(inner, reranker, pool=rerank_pool)


def _build_base(vector_store: Any, embedder: Any, mode: str, exclude_non_content: bool, rrf_k: int, candidate_pool: int) -> Retriever | HybridRetriever:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"mode must be one of {', '.join(RETRIEVAL_MODES)}, got {mode!r}")
    if mode == "dense":
        if (rrf_k, candidate_pool) != (RRF_K, DEFAULT_CANDIDATE_POOL):
            raise ValueError("rrf_k and candidate_pool only apply to mode='hybrid'")
        return Retriever(vector_store, embedder, exclude_non_content=exclude_non_content)
    return HybridRetriever(vector_store, embedder, exclude_non_content=exclude_non_content, rrf_k=rrf_k, candidate_pool=candidate_pool)
