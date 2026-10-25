"""M2 knowledge service: the contract-facing side of the knowledge base.

Everything the rest of Rafeeq needs from M2 goes through `KnowledgeService`:

    index_file(room_id, file_id, path)    upload hook (M1 calls it after saving a file)
    search(room_id, query, top_k)         POST /search               -> SearchMaterial
    list_topics(room_id)                  GET  /rooms/{id}/topics    -> RoomTopics
    source_page(room_id, file_name, page) GET  /sources/page        -> SourcePage
    tutor_answer(room_id, question)       grounded answer + citations -> TutorAnswer (optional, needs an LLM)

Retrieval uses the provisional configuration decided in the evaluation work: Hybrid (dense + BM25,
RRF k=60, candidate pool 50) with index-like pages excluded. Reranking is available but off.
Nothing heavy is loaded when this module is imported or when the service is created: the embedding
model (BGE-M3), FAISS and PyMuPDF are only touched by the first call that needs them.

Settings (all optional, environment variables):
    RAFEEQ_KNOWLEDGE_DIR        where the vector store is saved   (default: <RAFEEQ_DATA_DIR or storage>/knowledge/vector_store)
    RAFEEQ_KNOWLEDGE_DEVICE     embedding device                  (default: cpu)
    RAFEEQ_KNOWLEDGE_RETRIEVAL  hybrid | dense                    (default: hybrid)
    RAFEEQ_KNOWLEDGE_EXCLUDE    1 / 0, skip index-like pages       (default: 1)
    RAFEEQ_KNOWLEDGE_RERANK     1 to add the cross-encoder reranker (default: 0)
    RAFEEQ_KNOWLEDGE_ROOM_TOPICS 1 = learn each room's topics from its material with an LLM (default: 1);
                                0 = the fixed keyword taxonomy only. Uses LLM_* (else the Gemini settings).
    RAFEEQ_KNOWLEDGE_TOPIC_MIN_SIM  below this cosine a chunk is "General" (default: 0.30)
    GEMINI_API_KEY | GOOGLE_API_KEY, GEMINI_MODEL   only for tutor_answer (Gemini through its OpenAI-compatible endpoint)
"""
from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_TOP_K = 20  # the contract's SearchRequest.top_k maximum
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class KnowledgeError(Exception):
    """Base class; `status` is the HTTP status the router uses."""
    status = 400


class Invalid(KnowledgeError):
    status = 422


class NotFound(KnowledgeError):
    status = 404


class Unavailable(KnowledgeError):
    status = 503


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip() == "1"


def default_store_dir() -> Path:
    explicit = os.environ.get("RAFEEQ_KNOWLEDGE_DIR")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("RAFEEQ_DATA_DIR", "storage")) / "knowledge" / "vector_store"


def llm_from_env() -> Any | None:
    """Gemini through its OpenAI-compatible endpoint, or None when no key / model is configured."""
    key_env = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY"
    model = os.environ.get("GEMINI_MODEL", "").strip()
    if not model or not os.environ.get(key_env):
        return None
    from rafeeq.knowledge.evaluation.judge import OpenAICompatibleLLM

    return OpenAICompatibleLLM(GEMINI_OPENAI_BASE_URL, model, key_env, temperature=0.0)


class _LockedAdd:
    """What `ingest_file` sees as the vector store: only the final `add` holds the lock, so searches
    are not blocked for the minutes that parsing and embedding a big PDF can take."""

    def __init__(self, store: Any, lock: threading.RLock) -> None:
        self._store, self._lock = store, lock

    def add(self, embeddings: Any, chunks: list[dict[str, Any]]) -> int:
        with self._lock:
            return self._store.add(embeddings, chunks)

    def save(self, directory: Any) -> None:  # not used: the service saves with save_store_safely
        raise RuntimeError("the service saves the store itself")

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class KnowledgeService:
    def __init__(
        self,
        store_dir: str | Path | None = None,
        *,
        embedder: Any = None,
        store_cls: Any = None,
        retrieval_mode: str | None = None,
        exclude_non_content: bool | None = None,
        rerank: bool | None = None,
        status_callback: Callable[[str, str, str], None] | None = None,
        replace_callback: Callable[[str, str, Path, str], None] | None = None,
        parse: Callable[[str], list[dict[str, Any]]] | None = None,
        llm: Any = None,
        room_topics: bool = False,
        topic_propose: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.store_dir = Path(store_dir) if store_dir is not None else default_store_dir()
        self._embedder = embedder
        self._store_cls = store_cls
        self._mode = retrieval_mode or os.environ.get("RAFEEQ_KNOWLEDGE_RETRIEVAL", "hybrid")
        self._exclude = _flag("RAFEEQ_KNOWLEDGE_EXCLUDE", True) if exclude_non_content is None else exclude_non_content
        self._rerank = _flag("RAFEEQ_KNOWLEDGE_RERANK", False) if rerank is None else rerank
        self._status_callback = status_callback
        self._replace_callback = replace_callback
        self._parse = parse
        self._llm = llm
        # Learned per-room topics (topics/room_topics.py). Off unless asked for, so tests and scripts
        # never call an LLM by surprise; get_service() turns it on (RAFEEQ_KNOWLEDGE_ROOM_TOPICS, default 1).
        self._room_topics_on = room_topics
        self._topic_propose = topic_propose
        self._room_topics: Any = None
        self._lock = threading.RLock()
        self._store: Any = None
        self._retriever_obj: Any = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-index")  # one writer at a time

    # ------------------------------------------------------------------ lazy parts

    def describe(self) -> dict[str, Any]:
        return {"retrieval": self._mode, "exclude_non_content": self._exclude, "rerank": self._rerank,
                "candidate_pool": 50 if self._mode == "hybrid" else None, "store_dir": str(self.store_dir)}

    def _get_embedder(self) -> Any:
        with self._lock:
            if self._embedder is None:
                from rafeeq.knowledge.embeddings.embedder import Embedder

                self._embedder = Embedder(device=os.environ.get("RAFEEQ_KNOWLEDGE_DEVICE", "cpu"))
            return self._embedder

    def _existing_store(self) -> Any | None:
        """The loaded store, loading it from disk if it exists. None when nothing was ever indexed
        (this never creates a store and never loads the embedding model)."""
        from rafeeq.knowledge.store_persistence import StoreNotFoundError, load_existing_store

        with self._lock:
            if self._store is None:
                try:
                    self._store = load_existing_store(self.store_dir, store_cls=self._store_cls)
                except StoreNotFoundError:
                    return None
            return self._store

    def _retriever(self) -> Any | None:
        with self._lock:
            store = self._existing_store()
            if store is None:
                return None
            if self._retriever_obj is None:
                from rafeeq.knowledge.retrieval.factory import build_retriever

                self._retriever_obj = build_retriever(store, self._get_embedder(), mode=self._mode,
                                                      exclude_non_content=self._exclude, rerank=self._rerank)
            return self._retriever_obj

    def room_topics(self) -> Any:
        if self._room_topics is None:
            from rafeeq.knowledge.topics.room_topics import RoomTopics

            self._room_topics = RoomTopics(self.store_dir)
        return self._room_topics

    def _retag_for(self, room_id: str) -> Callable[[list[dict[str, Any]], Any], list[dict[str, Any]]] | None:
        if not self._room_topics_on:
            return None
        from rafeeq.knowledge.topics.room_topics import default_propose

        propose = self._topic_propose or default_propose()

        def retag(chunks: list[dict[str, Any]], embeddings: Any) -> list[dict[str, Any]]:
            try:
                topics = self.room_topics()
                topics.ensure_topics(room_id, chunks, propose, self._get_embedder())
                ids = topics.assign(room_id, embeddings)
            except Exception:  # topics must never make indexing fail
                log.exception("knowledge: room topics failed for room %s; keeping keyword tags", room_id)
                ids = None
            if ids is None:
                return chunks  # nothing learned (no LLM, or it failed): keep the keyword tags
            return [{**chunk, "topic_id": topic_id} for chunk, topic_id in zip(chunks, ids)]

        return retag

    # ------------------------------------------------------------------ indexing (M1 upload hook)

    def _set_status(self, room_id: str, file_id: str, status: str) -> None:
        if self._status_callback is not None:
            self._status_callback(room_id, file_id, status)
            return
        from rafeeq.platform import service as platform

        platform.get_store().set_file_status(room_id, file_id, status)

    def index_file(self, room_id: str, file_id: str, path: str | Path) -> Future:
        """Register with `rafeeq.platform.service.on_upload`. Returns at once: the file is marked
        "indexing", then "ready" or "failed" by a background worker. The Future resolves to the
        chunk count (0 when it failed) and never raises."""
        path = Path(path)
        prefix = f"{file_id}__"
        display_name = path.name[len(prefix):] if path.name.startswith(prefix) else path.name
        from rafeeq.knowledge.ingestion import office

        if office.is_convertible(path):  # a PowerPoint: converted to PDF here, in the background, then indexed
            self._set_status(room_id, file_id, "indexing")
            return self._pool.submit(self._convert_and_index, room_id, file_id, path, display_name)
        if path.suffix.lower() != ".pdf":
            log.warning("knowledge: %s is not a PDF; only PDFs can be indexed", display_name)
            self._set_status(room_id, file_id, "failed")
            done: Future = Future()
            done.set_result(0)
            return done
        self._set_status(room_id, file_id, "indexing")
        return self._pool.submit(self._index, room_id, file_id, path, display_name)

    def _replace_file(self, room_id: str, file_id: str, new_path: Path, new_name: str) -> None:
        if self._replace_callback is not None:
            self._replace_callback(room_id, file_id, new_path, new_name)
            return
        from rafeeq.platform import service as platform

        platform.get_store().replace_file(room_id, file_id, new_path, new_name)

    def _convert_and_index(self, room_id: str, file_id: str, path: Path, display_name: str) -> int:
        """.pptx -> PDF (one slide = one page), the PDF replaces the upload in the room, then it is indexed."""
        from rafeeq.knowledge.ingestion import office

        try:
            pdf = office.to_pdf(path, path.parent)
            pdf_name = Path(display_name).with_suffix(".pdf").name
            final = path.with_name(f"{file_id}__{pdf_name}")
            if pdf != final:
                pdf.replace(final)
            self._replace_file(room_id, file_id, final, pdf_name)
        except office.ConversionError as e:
            log.warning("knowledge: could not convert %s: %s", display_name, e)
            self._set_status(room_id, file_id, "failed")
            return 0
        except Exception:
            log.exception("knowledge: converting %s failed", display_name)
            self._set_status(room_id, file_id, "failed")
            return 0
        return self._index(room_id, file_id, final, pdf_name)

    def remove_file(self, room_id: str, file_id: str) -> Future:
        """Register with `rafeeq.platform.service.on_delete`. Runs on the same single worker as
        indexing, so a file deleted while it is still being indexed is removed after the indexing
        finishes. The Future resolves to the number of chunks removed and never raises."""
        return self._pool.submit(self._remove, room_id, file_id)

    def _remove(self, room_id: str, file_id: str) -> int:
        from rafeeq.knowledge.store_persistence import save_store_safely

        try:
            with self._lock:
                store = self._existing_store()
                if store is None or not hasattr(store, "remove_document"):
                    return 0
                removed = store.remove_document(room_id, file_id)
                if removed:
                    save_store_safely(store, self.store_dir, store_cls=self._store_cls or type(store))
                    self._retriever_obj = None  # the keyword (BM25) side is rebuilt without the file
            log.info("knowledge: removed file %s from room %s (%d chunks)", file_id, room_id, removed)
            return removed
        except Exception:
            log.exception("knowledge: removing file %s from room %s failed", file_id, room_id)
            return 0

    @staticmethod
    def _unique_name(store: Any, room_id: str, name: str) -> str:
        """A citation is (file_name, page), so two files of one room must not share a name:
        a second "lecture1.pdf" is stored as "lecture1 (2).pdf"."""
        taken = {c["file_name"] for c in store.room_chunks(room_id)}
        if name not in taken:
            return name
        stem, suffix = Path(name).stem, Path(name).suffix
        n = 2
        while f"{stem} ({n}){suffix}" in taken:
            n += 1
        return f"{stem} ({n}){suffix}"

    def _index(self, room_id: str, file_id: str, path: Path, display_name: str) -> int:
        from rafeeq.knowledge.ingestion.pipeline import ingest_file
        from rafeeq.knowledge.store_persistence import document_exists, load_or_create_store, save_store_safely

        try:
            embedder = self._get_embedder()
            with self._lock:
                if self._store is None:
                    if self._store_cls is None:
                        from rafeeq.knowledge.vector_store import FAISSVectorStore

                        self._store_cls = FAISSVectorStore
                    self._store, _ = load_or_create_store(self.store_dir, lambda: embedder.dimension, store_cls=self._store_cls)
                store = self._store
                already = document_exists(store, room_id, file_id)
            if already:  # the hook ran twice for the same upload: the index is already right
                log.info("knowledge: %s is already indexed in room %s", display_name, room_id)
                self._set_status(room_id, file_id, "ready")
                return 0
            with self._lock:
                display_name = self._unique_name(store, room_id, display_name)
            stats = ingest_file(path, room_id, _LockedAdd(store, self._lock), embedder,
                                document_id=file_id, file_name=display_name, parse=self._parse,
                                retag=self._retag_for(room_id))
            with self._lock:
                save_store_safely(store, self.store_dir, store_cls=self._store_cls)
            self._set_status(room_id, file_id, "ready")
            log.info("knowledge: indexed %s into room %s (%d chunks)", display_name, room_id, stats.chunk_count)
            return stats.chunk_count
        except Exception:
            log.exception("knowledge: indexing %s failed", display_name)
            try:
                self._set_status(room_id, file_id, "failed")
            except Exception:
                log.exception("knowledge: could not mark %s as failed", file_id)
            return 0

    # ------------------------------------------------------------------ contract operations

    @staticmethod
    def _text(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise Invalid(f"{label} must be a non-empty string")
        return value

    def search(self, room_id: str, query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """SearchMaterial: `{"chunks": [{file_name, page, text, topic_id}]}`, best first."""
        self._text(room_id, "room_id")
        self._text(query, "query")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise Invalid(f"top_k must be an integer between 1 and {MAX_TOP_K}")
        with self._lock:
            retriever = self._retriever()
            hits = retriever.retrieve(room_id, query, top_k) if retriever is not None else []
        return {"chunks": [{"file_name": h["file_name"], "page": int(h["page"]), "text": h["text"], "topic_id": h["topic_id"]} for h in hits]}

    def list_topics(self, room_id: str) -> dict[str, Any]:
        """RoomTopics: only the topics that really occur in the room's material, most frequent first."""
        self._text(room_id, "room_id")
        from rafeeq.knowledge.topics import get_default_manager

        with self._lock:
            store = self._existing_store()
            chunks = store.room_chunks(room_id) if store is not None else []
        counts = Counter(c["topic_id"] for c in chunks)
        names = {t["topic_id"]: t["name"] for t in get_default_manager().list_topics(room_id)}
        names.update(self.room_topics().names(room_id))  # learned topics of this room win
        ordered = sorted(counts, key=lambda topic_id: (-counts[topic_id], topic_id))
        return {"topics": [{"topic_id": t, "name": names.get(t, t)} for t in ordered]}

    def room_chunks(self, room_id: str) -> list[dict[str, Any]]:
        """Every chunk of the room's material with its topic name: what M4 builds quizzes from.
        Empty when nothing is indexed yet (never loads the embedding model)."""
        self._text(room_id, "room_id")
        from rafeeq.knowledge.topics import get_default_manager

        with self._lock:
            store = self._existing_store()
            chunks = store.room_chunks(room_id) if store is not None else []
        names = {t["topic_id"]: t["name"] for t in get_default_manager().list_topics(room_id)}
        names.update(self.room_topics().names(room_id))
        return [{"chunk_id": c["chunk_id"], "text": c["text"], "file_name": c["file_name"], "page": c["page"],
                 "block_type": c.get("block_type", "text"), "topic_id": c["topic_id"],
                 "topic_name": names.get(c["topic_id"], c["topic_id"])} for c in chunks]

    def topic_for_citations(self, room_id: str, citations: list[dict[str, Any]]) -> str | None:
        """The topic most of the cited pages belong to, so a question can count towards the
        student's profile (TutorAnswer itself carries no topic). None when nothing matches."""
        wanted = {(c.get("file_name"), c.get("page")) for c in citations or []}
        if not wanted:
            return None
        with self._lock:
            store = self._existing_store()
            chunks = store.room_chunks(room_id) if store is not None else []
        counts = Counter(c["topic_id"] for c in chunks if (c["file_name"], c["page"]) in wanted)
        return counts.most_common(1)[0][0] if counts else None

    OVERVIEW_SYSTEM = (
        "You are Rafeeq, a patient tutor. The student asked about a lesson as a whole. Using ONLY the "
        "excerpts below from that lesson file, explain its main idea simply: what it is about, the 3 to 5 "
        "key points, and how they connect. Short sentences, no jargon without a one-line explanation. "
        "Write in the language of the student's request (Arabic, including Egyptian dialect, or English); "
        "keep technical terms in their original language when that is clearer. "
        "Mention the page of each point like (p. 12). Never add facts that are not in the excerpts. "
        'Reply with JSON {"answer": "<the explanation>", "pages": [<page numbers you used>]}.'
    )

    def overview(self, room_id: str, question: str, propose: Callable[[str, str], Any] | None = None
                 ) -> dict[str, Any] | None:
        """TutorAnswer that explains the room's most recently uploaded file as a whole ("explain the last
        lesson simply"). Grounded in excerpts of that file only. None when nothing is indexed or no LLM."""
        self._text(room_id, "room_id")
        with self._lock:
            store = self._existing_store()
            chunks = store.room_chunks(room_id) if store is not None else []
        if not chunks:
            return None
        latest = chunks[-1]["document_id"]  # the store keeps insertion order: last indexed = newest upload
        doc = [c for c in chunks if c["document_id"] == latest]
        step = max(1, len(doc) // 12)
        picked = doc[::step][:12]
        excerpts = "\n\n".join(f"(p. {c['page']}) {c['text'][:700]}" for c in picked)
        if propose is None:
            from rafeeq.knowledge.topics.room_topics import default_propose

            propose = default_propose()
        if propose is None:
            return None
        try:
            out = propose(self.OVERVIEW_SYSTEM, f"Student's request: {question}\n\nFile: {doc[0]['file_name']}\n\n{excerpts}")
        except Exception:
            log.exception("knowledge: lesson overview failed")
            return None
        answer = str((out or {}).get("answer", "")).strip()
        if not answer:
            return None
        pages_used = {int(p) for p in (out or {}).get("pages", []) if str(p).isdigit()}
        available = sorted({c["page"] for c in picked})
        pages = [p for p in available if p in pages_used] or available[:3]
        return {"answer": answer, "grounded": True, "steps": [],
                "citations": [{"file_name": doc[0]["file_name"], "page": p} for p in pages]}

    def source_page(self, room_id: str, file_name: str, page: int) -> dict[str, Any]:
        """SourcePage: the text of one page of one file, so a citation can be opened."""
        self._text(room_id, "room_id")
        self._text(file_name, "file_name")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise Invalid("page must be an integer >= 1")
        with self._lock:
            store = self._existing_store()
            chunks = store.room_chunks(room_id) if store is not None else []
        texts = [c["text"] for c in chunks if c["file_name"] == file_name and c["page"] == page]
        if not texts:
            raise NotFound(f"{file_name} page {page} is not in this room")
        return {"file_name": file_name, "page": page, "text": "\n".join(texts)}

    def tutor_answer(self, room_id: str, question: str) -> dict[str, Any]:
        """TutorAnswer from the grounded Answerer: answers only from the room's material, or says it
        is not covered (grounded=False). Needs an LLM (see `llm_from_env`); `steps` is always empty."""
        self._text(room_id, "room_id")
        self._text(question, "question")
        llm = self._llm or llm_from_env()
        if llm is None:
            raise Unavailable("no answer LLM configured: set GEMINI_API_KEY (or GOOGLE_API_KEY) and GEMINI_MODEL")
        with self._lock:
            retriever = self._retriever()
        from rafeeq.knowledge.answering.answerer import NOT_COVERED_MESSAGE, Answerer
        from rafeeq.knowledge.evaluation.metrics import answer_body

        if retriever is None:
            return {"answer": NOT_COVERED_MESSAGE, "citations": [], "grounded": False, "steps": []}
        result = Answerer(retriever, llm).answer(room_id, question)
        if not result["grounded"]:
            return {"answer": result["answer_md"], "citations": [], "grounded": False, "steps": []}
        return {"answer": answer_body(result["answer_md"]),
                "citations": [{"file_name": c["file_name"], "page": int(c["page"])} for c in result["citations"]],
                "grounded": True, "steps": []}


# ------------------------------------------------------------------ default instance

_service: KnowledgeService | None = None
_service_lock = threading.Lock()


def get_service() -> KnowledgeService:
    global _service
    with _service_lock:
        if _service is None:
            _service = KnowledgeService(room_topics=_flag("RAFEEQ_KNOWLEDGE_ROOM_TOPICS", True))
        return _service


def set_service(service: KnowledgeService | None) -> None:
    """Tests use this to point M2 at a temporary store and a fake embedder."""
    global _service
    with _service_lock:
        _service = service
