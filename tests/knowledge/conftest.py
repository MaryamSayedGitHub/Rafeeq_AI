"""Fixtures and collection rules for the M2 knowledge-base tests."""
import importlib.util
import os

import pytest

from kb_fakes import HashingEmbedder

# faiss-cpu is an OPTIONAL Rafeeq dependency (the "knowledge" extra), so the app and the other members' tests
# work without it. These modules cannot even be imported without it; they are skipped, not failed, until it
# is installed:  uv add --optional knowledge sentence-transformers faiss-cpu   then   uv sync --extra knowledge
NEEDS_FAISS = [
    "test_compare_runs.py", "test_diagnose_answers.py", "test_eval_dataset_ground_truth.py", "test_faiss_store.py",
    "test_heldout_report.py", "test_hybrid_retriever.py", "test_ingest_query_scripts.py", "test_m2_router.py",
    "test_m2_service.py", "test_pipeline.py", "test_replay_hybrid.py", "test_rerank_report.py", "test_reranker.py",
    "test_rescore_run.py", "test_retriever.py", "test_retriever_exclusion.py", "test_run_eval_exclusion.py", "test_run_eval_hybrid.py",
    "test_run_eval_script.py", "test_store_persistence.py",
]
_FAISS_MISSING = importlib.util.find_spec("faiss") is None
collect_ignore = list(NEEDS_FAISS) if _FAISS_MISSING else []
# integration/ loads the real BGE-M3 model (a ~2 GB download, minutes on CPU): only with RAFEEQ_RUN_BGE_M3_TESTS=1
if os.environ.get("RAFEEQ_RUN_BGE_M3_TESTS") != "1":
    collect_ignore.append("integration")


def pytest_report_header(config):
    if _FAISS_MISSING:
        return f"knowledge tests: faiss is not installed, {len(NEEDS_FAISS)} test modules skipped (see tests/knowledge/conftest.py)"


@pytest.fixture()
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture()
def empty_store(hashing_embedder):
    from rafeeq.knowledge.vector_store import FAISSVectorStore

    return FAISSVectorStore(hashing_embedder.dimension)


@pytest.fixture()
def build_store(hashing_embedder):
    """build_store({room_id: {file_name: [page text, ...]}}) -> populated FAISSVectorStore."""
    from rafeeq.knowledge.ingestion.chunker import chunk_pages
    from rafeeq.knowledge.topics import TopicManager
    from rafeeq.knowledge.vector_store import FAISSVectorStore

    def _build(rooms):
        store = FAISSVectorStore(hashing_embedder.dimension)
        manager = TopicManager()
        for room_id, files in rooms.items():
            for file_name, page_texts in files.items():
                pages = [{"page": i + 1, "text": t} for i, t in enumerate(page_texts)]
                chunks = manager.tag_chunks(room_id, chunk_pages(pages, room_id, file_name))
                store.add(hashing_embedder.embed_documents([c["text"] for c in chunks]), chunks)
        hashing_embedder.document_calls.clear()
        return store

    return _build