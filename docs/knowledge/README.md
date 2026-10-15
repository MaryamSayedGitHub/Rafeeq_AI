# M2 knowledge base: evaluation documents

`EVALUATION.md`, `HYBRID_IMPLEMENTATION_REPORT.md`, `RERANKING.md` and `pages_inspection.txt` were written in the standalone
`knowledge_base` project and are kept **verbatim** as the record of how the retrieval was built and measured. Their commands
use the old layout. In the Rafeeq repository:

| in these documents | in Rafeeq |
|---|---|
| `src/knowledge_base/...` , `knowledge_base.` | `rafeeq/knowledge/...` , `rafeeq.knowledge.` |
| `scripts/<name>.py` | `rafeeq/knowledge/scripts/<name>.py` |
| `eval/...` | `rafeeq/knowledge/eval/...` |
| `tests/...` | `tests/knowledge/...` |
| `data/vector_store`, `data/raw` | unchanged (git-ignored: build them with `rafeeq/knowledge/scripts/ingest_book.py`) |

Run everything from the repository root. How M2 plugs into the app: `rafeeq/knowledge/README.md`.
