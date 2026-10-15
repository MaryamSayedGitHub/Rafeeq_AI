# M2 knowledge base (Rafeeq)

Page-aware, multilingual (Arabic/English) course knowledge base: PDFs are split into one chunk per page, embedded with BGE-M3,
stored in FAISS, and searched per "room". Retrieval has two modes: **dense** (the default, unchanged) and **hybrid**
(dense + in-project BM25 fused with Reciprocal Rank Fusion).

## Integration with Rafeeq (read this first)

This folder is Member 2's part of Rafeeq. It used to be the standalone `knowledge_base` project; its code now lives here as
`rafeeq.knowledge` (imports `from rafeeq.knowledge...`), its tests in `tests/knowledge/`, its documents in `docs/knowledge/`,
its benchmarks and audits in `rafeeq/knowledge/eval/`, and its command-line tools in `rafeeq/knowledge/scripts/`.

**What the rest of Rafeeq sees** (`rafeeq/knowledge/service.py`, `router.py`, `integration.py`):

| contract | provided by |
|---|---|
| upload hook (M1 `on_upload`) | `KnowledgeService.index_file(room_id, file_id, path)`: marks the file `indexing`, indexes the PDF in the background, then `ready` or `failed` |
| `POST /search` -> `SearchMaterial` | `KnowledgeService.search(room_id, query, top_k)` |
| `GET /rooms/{room_id}/topics` -> `RoomTopics` | `KnowledgeService.list_topics(room_id)` (only topics present in the room) |
| `GET /sources/page` -> `SourcePage` | `KnowledgeService.source_page(room_id, file_name, page)` |
| Copilot `tutor_answer` (optional, M3 may replace it) | `KnowledgeService.tutor_answer(room_id, question)`: grounded answer with page citations, or `grounded=false` |

All three routes need `Authorization: Bearer <token>` and room membership, like M1's room routes. Retrieval uses the provisional
configuration from the evaluation work: **Hybrid (dense + BM25, RRF k=60, candidate pool 50) with index-like pages excluded**; the
cross-encoder reranker exists but is off. Everything is **off until you turn it on**, so the app, Docker image and the other
members' tests behave exactly as before:

| setting (environment) | meaning |
|---|---|
| `RAFEEQ_KNOWLEDGE=1` | plug M2 into the app (real routes + indexing of uploaded PDFs) |
| `RAFEEQ_KNOWLEDGE_TUTOR=1` | also connect `tutor_answer` to the Copilot (needs `GEMINI_API_KEY` or `GOOGLE_API_KEY`, and `GEMINI_MODEL`) |
| `RAFEEQ_KNOWLEDGE_DIR` | where the vector store is saved (default `storage/knowledge/vector_store`, git-ignored) |
| `RAFEEQ_KNOWLEDGE_DEVICE` | `cpu` (default) or `cuda` for BGE-M3 |
| `RAFEEQ_KNOWLEDGE_RETRIEVAL`, `_EXCLUDE`, `_RERANK` | `hybrid`/`dense`, `1`/`0`, `1`/`0` (defaults `hybrid`, `1`, `0`) |

**Install the heavy dependencies once** (torch via sentence-transformers is large, so they are an optional extra, not a default dependency):

```powershell
uv add --optional knowledge sentence-transformers faiss-cpu   # edits pyproject.toml and uv.lock
uv sync --extra knowledge
```

`numpy` and `pymupdf` are already Rafeeq dependencies. Only PDFs are indexed (other uploads end as `failed`). The first query or
upload downloads BGE-M3 (about 2 GB) from the Hugging Face hub.

**Tests**

```powershell
uv run pytest                          # whole repo; tests/knowledge skips the modules that need faiss until it is installed
uv run pytest tests/knowledge -v       # M2 only
$env:RAFEEQ_RUN_BGE_M3_TESTS = "1"; uv run pytest tests/knowledge/integration   # real BGE-M3 (slow, downloads the model)
```

**Known limits of this integration**
* Topics: the keyword taxonomy (`topics/topic_manager.py`) was written for the LLM book; material from other subjects is mostly tagged `topic_general` until a per-room taxonomy is registered.
* Indexing runs in a background thread and a restart during indexing leaves the file at `indexing` (re-upload it).
* A second upload with the same file name in one room is stored as `name (2).pdf` (citations must be unambiguous), so its citations show that name.
* The vector store is a single FAISS index on local disk; there is no delete/re-index of a removed file yet.

The sections below are the standalone project's documentation. Paths were updated to the new layout; run all commands from the repository root.

## Switch between dense and hybrid retrieval

Dense is the default everywhere. Nothing changes unless you ask for hybrid.

Command line:

```powershell
uv run python rafeeq/knowledge/scripts/query_book.py "What is self-attention?"                                   # dense (default)
uv run python rafeeq/knowledge/scripts/query_book.py "What is self-attention?" --retrieval-mode hybrid           # dense + BM25 + RRF
uv run python rafeeq/knowledge/scripts/query_book.py "What is self-attention?" --retrieval-mode hybrid --exclude-non-content
```

Python:

```python
from rafeeq.knowledge.retrieval import build_retriever, set_default_retriever, HybridRetriever

dense  = build_retriever(store, embedder)                                  # same as Retriever(store, embedder)
hybrid = build_retriever(store, embedder, mode="hybrid", exclude_non_content=True)
# or directly: HybridRetriever(store, embedder, rrf_k=60, candidate_pool=50)
hits = hybrid.retrieve("llm_course", "What is self-attention?", k=6)
set_default_retriever(hybrid)                                              # optional: make module-level retrieve() use it
```

Every hit keeps the dense contract (`chunk_id, text, file_name, page, topic_id, score`), where `score` is always the **true cosine
similarity** (so the Answerer's `min_score` gate keeps its meaning). Hybrid adds `fusion_score` (the value used for ranking),
`dense_rank`, `lexical_rank` and `lexical_score`.

### Hybrid design (fixed, not tuned)

* BM25 Okapi (k1 = 1.5, b = 0.75) over the chunks of the requested room, built in memory on first use.
* RRF: `fusion = 1/(60 + rank_dense) + 1/(60 + rank_bm25)`, equal weights, candidate pool 50 per list.
* Ties: higher cosine, then `chunk_id`. A question with no lexical overlap gets exactly the dense order.
* Room isolation: both candidate lists come only from the requested room, and a hit from another room raises `RuntimeError`.
* `--exclude-non-content` removes index-like pages from both lists before ranks are assigned.

## Reproduce the evaluation

Each run writes to a **new** directory; `run_eval.py` refuses to overwrite an existing `results.json`, and the stored results in
`rafeeq/knowledge/eval/results/` were not modified.

```powershell
# the four modes (needs the BGE-M3 model and the saved store in data\vector_store)
uv run python rafeeq/knowledge/scripts/run_eval.py --out-dir rafeeq\knowledge\eval\results\dense_v2
uv run python rafeeq/knowledge/scripts/run_eval.py --exclude-non-content --out-dir rafeeq\knowledge\eval\results\dense_exclude_v2
uv run python rafeeq/knowledge/scripts/run_eval.py --retrieval-mode hybrid --out-dir rafeeq\knowledge\eval\results\hybrid
uv run python rafeeq/knowledge/scripts/run_eval.py --retrieval-mode hybrid --exclude-non-content --out-dir rafeeq\knowledge\eval\results\hybrid_exclude

# per-question improvements / regressions
uv run python rafeeq/knowledge/scripts/compare_runs.py rafeeq\knowledge\eval\results\dense_v2\results.json rafeeq\knowledge\eval\results\hybrid\results.json --out rafeeq\knowledge\eval\analysis\dense_vs_hybrid.md
uv run python rafeeq/knowledge/scripts/compare_runs.py rafeeq\knowledge\eval\results\dense_exclude_v2\results.json rafeeq\knowledge\eval\results\hybrid_exclude\results.json --out rafeeq\knowledge\eval\analysis\dense_exclude_vs_hybrid_exclude.md
```

Without a model (what was run for the report in this repository): replay hybrid ranking offline from a stored dense run
(candidate pool = the stored depth, 5; see `HYBRID_IMPLEMENTATION_REPORT.md` for what this does and does not show):

```powershell
uv run python rafeeq/knowledge/scripts/replay_hybrid.py rafeeq\knowledge\eval\results\baseline_again\results.json --out-dir rafeeq\knowledge\eval\results\replay_hybrid_baseline
uv run python rafeeq/knowledge/scripts/replay_hybrid.py rafeeq\knowledge\eval\results\exclude_index\results.json --exclude-non-content --out-dir rafeeq\knowledge\eval\results\replay_hybrid_exclude
```

## Held-out evaluation (use this for any Hybrid-vs-Dense claim)

`rafeeq/knowledge/eval/benchmark/hands_on_llm_v1.jsonl` / `v1_1` are the **development** benchmark. For a final comparison use the frozen, separate
`rafeeq/knowledge/eval/benchmark/hands_on_llm_heldout_v1.jsonl` (54 questions: 23 single-passage, 19 multi-passage, 12 out-of-scope; English and Arabic;
labels written from the book text only). Its protocol, label-audit method, known issues, pre-registered decision rule and the four
commands to run are in `rafeeq/knowledge/eval/HELDOUT_PROTOCOL.md`; per-question evidence is in `rafeeq/knowledge/eval/audit/heldout_label_audit.md`.
Do not tune anything on it. After running the four configurations (Dense, Dense + exclusion, Hybrid, Hybrid + exclusion, pool 50):

```powershell
uv run python rafeeq/knowledge/scripts/heldout_report.py --dense rafeeq\knowledge\eval\results\heldout_dense\results.json --dense-exclude rafeeq\knowledge\eval\results\heldout_dense_exclude\results.json `
  --hybrid rafeeq\knowledge\eval\results\heldout_hybrid\results.json --hybrid-exclude rafeeq\knowledge\eval\results\heldout_hybrid_exclude\results.json --out rafeeq\knowledge\eval\analysis\heldout_final_comparison.md
```

The report refuses replay output, other benchmarks, other settings, and a benchmark file whose sha256 differs from `hands_on_llm_heldout_v1.sha256`.

More detail on the benchmark and metrics: `rafeeq/knowledge/eval/EVALUATION.md`. Implementation report: `HYBRID_IMPLEMENTATION_REPORT.md`.
Audit of q01/q21: `rafeeq/knowledge/eval/audit/q01_q21_audit.md`. Label versions and re-scoring: `rafeeq/knowledge/eval/BENCHMARK_LABELS.md`, `rafeeq/knowledge/scripts/rescore_run.py`.

## RAG answer-layer evaluation (Gemini, provisional configuration)

The grounded `Answerer` (excerpts only, `[n]` citations, `NOT_COVERED` / `min_score` 0.35 abstention) is evaluated by `run_eval.py` when an LLM is given. Gemini is used through Google's OpenAI-compatible endpoint (no new dependency; the key comes from the environment and is never stored):

```powershell
$env:GEMINI_API_KEY = "..."
uv run python rafeeq/knowledge/scripts/run_eval.py --benchmark rafeeq\knowledge\eval\benchmark\hands_on_llm_v1_1.jsonl `
  --llm-provider gemini --llm-model <model> --provisional-rag-config --llm-min-interval 6 `
  --out-dir rafeeq\knowledge\eval\results\rag_dev_hybrid_exclude
```

`--provisional-rag-config` = Hybrid + index-like pages excluded, candidate pool 50. Add `--judge --judge-model <a different model>` for LLM-judged correctness / faithfulness against the reference answers. The report shows single-passage, multi-passage and out-of-scope groups separately. Develop the prompt on the dev benchmark; run the held-out file once, with nothing changed afterwards.


## Optional: Cross-Encoder reranking

`--rerank` (off by default) reranks the top-50 candidates of the chosen retrieval mode with `BAAI/bge-reranker-v2-m3` and returns the top 6 to the unchanged Answerer. Dense and hybrid behaviour without the flag is unchanged.

```powershell
uv run python rafeeq/knowledge/scripts/query_book.py "What is self-attention?" --retrieval-mode hybrid --rerank
```
Evaluation, Colab notebook (`colab/rerank_retrieval_eval.ipynb`), commands and limitations: [docs/RERANKING.md](docs/RERANKING.md).
