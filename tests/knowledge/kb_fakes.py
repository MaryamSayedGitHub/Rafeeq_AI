"""Test doubles shared by the retriever / pipeline tests. NOT production code.

``HashingEmbedder`` is a tiny deterministic stand-in for BGE-M3: it hashes
words into a 256-dim bag-of-words vector (L2-normalised, like the real
embedder). A few Arabic words are mapped to the English concept they mean so
Arabic queries can be exercised end to end. That proves the plumbing
(Arabic text in -> right page out), NOT BGE-M3's real cross-lingual quality;
that is checked by tests/integration/test_retrieval_bge_m3.py with the real model.
"""

from __future__ import annotations

import re
import zlib

DIM = 256

# Arabic word -> English concept ("" = question filler, dropped)
ARABIC_CONCEPTS = {
    "انتباه": "attention", "الانتباه": "attention", "الذاتي": "self", "ذاتي": "self",
    "الشبكات": "network", "العصبية": "neural", "الالتفافية": "convolutional",
    "يعني": "", "إيه": "", "ما": "", "هو": "", "هي": "", "ماذا": "",
}
STOP_WORDS = {"the", "a", "an", "is", "are", "to", "of", "for", "on", "in", "what", "does", "do",
              "how", "each", "other", "allows", "used", "use", "by", "and", "can", "it"}


def tokens(text: str) -> list[str]:
    out = []
    for word in re.findall(r"\w+", text.lower()):
        word = ARABIC_CONCEPTS.get(word, word)
        if not word or word in STOP_WORDS:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]  # crude plural stripping: tokens -> token
        out.append(word)
    return out


def sparse_vector(text: str) -> dict[int, float]:
    counts: dict[int, float] = {}
    for tok in tokens(text):
        index = zlib.crc32(tok.encode("utf-8")) % DIM
        counts[index] = counts.get(index, 0.0) + 1.0
    return counts or {0: 1.0}


class HashingEmbedder:
    """Deterministic fake with the same interface as the real Embedder."""

    dimension = DIM

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def _dense(self, text: str):
        import numpy as np

        vector = np.zeros(DIM, dtype=np.float32)
        for index, value in sparse_vector(text).items():
            vector[index] = value
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts):
        import numpy as np

        self.document_calls.append(list(texts))
        return np.stack([self._dense(t) for t in texts])

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._dense(text)