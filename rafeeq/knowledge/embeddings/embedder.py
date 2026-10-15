"""Multilingual text embeddings with BAAI/bge-m3 (SentenceTransformers).

- The model is loaded lazily: importing this module or creating an Embedder
  never downloads or loads anything. The first call that needs the model
  (``load()``, ``embed_documents()``, ``embed_query()``, ``dimension``) loads it.
- All vectors are L2-normalised, so cosine similarity == dot product.
- Device is configurable ("cpu" by default; use "cuda" on a cloud GPU).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 8  # small on purpose: safe for CPU / low-memory machines


class Embedder:
    """Thin, reusable wrapper around a SentenceTransformer model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None  # loaded on first use

    # ---------- model ----------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Load the model if it is not loaded yet, and return it."""
        if self._model is None:
            # Imported here so that importing this module stays cheap
            # (torch is only imported when a model is really needed).
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension (1024 for BAAI/bge-m3). Loads the model."""
        return int(self.load().get_embedding_dimension())

    # ---------- embedding ----------

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many texts -> float32 array of shape (len(texts), dimension)."""
        texts = self._validate_texts(texts)
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query -> float32 array of shape (dimension,)."""
        if not isinstance(text, str):
            raise TypeError(f"query must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return self._encode([text])[0]

    # ---------- internals ----------

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            raise TypeError(
                "texts must be a list of strings, not a single string "
                "(use embed_query() for one query)"
            )
        try:
            texts = list(texts)
        except TypeError:
            raise TypeError(f"texts must be a list of strings, got {type(texts).__name__}") from None
        if not texts:
            raise ValueError("texts must contain at least one text; got an empty list")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"texts[{i}] must be a string, got {type(t).__name__}")
            if not t.strip():
                raise ValueError(f"texts[{i}] is empty or whitespace-only")
        return texts

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self.load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # unit length -> dot product == cosine
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)