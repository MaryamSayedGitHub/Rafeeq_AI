"""Deterministic topic tagging for course chunks.

Design
------
- A room has a FIXED taxonomy of topics (about 8-15). Topic ids are stable
  slugs such as ``topic_attention``. The taxonomy is not learned from the
  uploaded files, so adding a file can never rename, add, or remove a topic.
- Tagging is a pure function of ``(taxonomy, text)``: same text -> same
  topic, always. No randomness, no LLM, no network, no model.
- The default classifier scores keyword / phrase matches. It sits behind the
  small ``TopicClassifier`` protocol, so a stronger classifier (embedding
  based, LLM based, ...) can replace it without touching callers.
- Anything that matches no topic well enough gets ``topic_general``, so
  every chunk always receives a valid topic id.
- Tagging happens AFTER chunking; the chunker is not involved.

Public contract (module level, as requested by the team)::

    tag_topic(room_id, text) -> topic_id
    list_topics(room_id)     -> [{"topic_id": "...", "name": "..."}]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

GENERAL_TOPIC_ID = "topic_general"
GENERAL_TOPIC_NAME = "General"

MAX_TOPICS_PER_ROOM = 15  # including the general fallback
MIN_SCORE = 2  # below this the text is "general" (one stray word is not a topic)
MAX_COUNT_PER_KEYWORD = 3  # repeated mentions of one keyword stop adding score

_TOPIC_ID_RE = re.compile(r"^topic_[a-z0-9]+(?:_[a-z0-9]+)*$")
_KEYWORD_SPLIT_RE = re.compile(r"[\s\-]+")


@dataclass(frozen=True)
class Topic:
    """One topic of a room's taxonomy."""

    topic_id: str
    name: str
    keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.topic_id, str) or not _TOPIC_ID_RE.match(self.topic_id):
            raise ValueError(
                f"topic_id must look like 'topic_some_name' (lowercase, digits, underscores), "
                f"got {self.topic_id!r}"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"topic name must be a non-empty string for {self.topic_id}")
        if not isinstance(self.keywords, tuple) or any(
            not isinstance(k, str) or not k.strip() for k in self.keywords
        ):
            raise ValueError(f"keywords of {self.topic_id} must be a tuple of non-empty strings")


class TopicClassifier(Protocol):
    """Replaceable classifier interface: return the id of one of ``topics``."""

    def classify(self, text: str, topics: Sequence[Topic]) -> str: ...


# ------------------------------------------------------------ default taxonomy
# Matches the "Hands-On Large Language Models" course material. Other rooms
# can register their own taxonomy with ``TopicManager.register_room_topics``.
# Arabic keywords are intentionally few: extend them with the team's course
# terminology. Multi-word keywords are worth double a single word.

DEFAULT_TOPICS: tuple[Topic, ...] = (
    Topic("topic_tokenization", "Tokenization", (
        "tokenizer", "tokenization", "tokenizing", "bpe", "byte pair encoding", "wordpiece",
        "sentencepiece", "subword", "vocabulary", "special tokens",
    )),
    Topic("topic_embeddings", "Embeddings & Vector Representations", (
        "embedding", "word2vec", "sentence embedding", "vector representation", "vector space",
        "cosine similarity", "sentence-transformers", "التضمين", "التضمينات",
    )),
    Topic("topic_attention", "Attention Mechanism", (
        "attention", "self-attention", "multi-head", "multi-head attention", "cross-attention",
        "query key value", "attention head", "attention mask", "الانتباه الذاتي", "الانتباه",
    )),
    Topic("topic_transformers", "Transformer Architecture", (
        "transformer", "encoder", "decoder", "positional encoding", "feed-forward",
        "layer normalization", "residual connection", "bert", "gpt", "ترانسفورمر",
    )),
    Topic("topic_language_models", "Language Models & Text Generation", (
        "language model", "llm", "next-token prediction", "text generation", "decoding",
        "temperature", "top-p", "top-k", "sampling", "autoregressive", "النموذج اللغوي",
    )),
    Topic("topic_prompting", "Prompt Engineering", (
        "prompt", "prompting", "prompt engineering", "few-shot", "zero-shot", "one-shot",
        "chain-of-thought", "in-context learning", "system prompt",
    )),
    Topic("topic_text_classification", "Text Classification", (
        "classification", "classifier", "sentiment", "sentiment analysis", "text classification",
        "labeled data", "f1 score", "التصنيف",
    )),
    Topic("topic_semantic_search_rag", "Semantic Search & RAG", (
        "semantic search", "retrieval", "rag", "retrieval-augmented generation", "reranking",
        "rerank", "vector database", "faiss", "chunking", "knowledge base", "dense retrieval",
    )),
    Topic("topic_text_clustering", "Text Clustering & Topic Modeling", (
        "clustering", "cluster", "topic modeling", "bertopic", "umap", "hdbscan", "k-means",
        "dimensionality reduction",
    )),
    Topic("topic_fine_tuning", "Training & Fine-Tuning", (
        "fine-tuning", "fine tuning", "finetuning", "lora", "peft", "pretraining", "pre-training",
        "instruction tuning", "rlhf", "preference tuning", "supervised fine-tuning",
        "training data", "ضبط دقيق",
    )),
    Topic("topic_evaluation", "Evaluation & Benchmarks", (
        "evaluation", "benchmark", "perplexity", "bleu", "rouge", "mmlu", "metric",
        "human evaluation", "llm-as-a-judge",
    )),
    Topic("topic_multimodal_agents", "Multimodal Models & Agents", (
        "multimodal", "vision-language", "image captioning", "clip", "agent", "tool use",
        "function calling", "langchain", "multimodal model",
    )),
    Topic("topic_deep_learning_basics", "Neural Network Basics", (
        "neural network", "gradient descent", "backpropagation", "convolutional",
        "cnn", "rnn", "lstm", "activation function", "loss function", "الشبكات العصبية",
    )),
)


# ------------------------------------------------------- default classifier

def _compile_keyword(keyword: str) -> tuple[re.Pattern[str], int]:
    """Whole-word, case-insensitive pattern. Spaces and hyphens are interchangeable
    and a plural 's' is accepted. Returns (pattern, weight)."""
    parts = [p for p in _KEYWORD_SPLIT_RE.split(keyword.strip()) if p]
    body = r"[\s\-]+".join(re.escape(p) for p in parts)
    pattern = re.compile(rf"(?<!\w){body}(?:s|es)?(?!\w)", re.IGNORECASE)
    return pattern, (2 if len(parts) > 1 else 1)


class KeywordTopicClassifier:
    """Deterministic keyword scorer.

    score(topic) = sum over its keywords of weight * min(occurrences, 3).
    The highest score wins; ties go to the topic listed first in the
    taxonomy; a best score below ``MIN_SCORE`` means ``topic_general``.
    """

    def __init__(self) -> None:
        self._compiled: dict[Topic, tuple[tuple[re.Pattern[str], int], ...]] = {}

    def _patterns(self, topic: Topic) -> tuple[tuple[re.Pattern[str], int], ...]:
        if topic not in self._compiled:
            self._compiled[topic] = tuple(_compile_keyword(k) for k in topic.keywords)
        return self._compiled[topic]

    def score(self, text: str, topic: Topic) -> int:
        return sum(
            weight * min(len(pattern.findall(text)), MAX_COUNT_PER_KEYWORD)
            for pattern, weight in self._patterns(topic)
        )

    def classify(self, text: str, topics: Sequence[Topic]) -> str:
        best_id, best_score = GENERAL_TOPIC_ID, 0
        for topic in topics:
            if not topic.keywords:
                continue
            score = self.score(text, topic)
            if score > best_score:  # strict '>' keeps taxonomy order as the tie-break
                best_id, best_score = topic.topic_id, score
        return best_id if best_score >= MIN_SCORE else GENERAL_TOPIC_ID


# ------------------------------------------------------------------ manager

def _require_non_empty_str(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")


def _finalise(topics: Sequence[Topic]) -> tuple[Topic, ...]:
    """Validate a taxonomy and append the general fallback topic."""
    topics = tuple(topics)
    if not topics:
        raise ValueError("a taxonomy needs at least one topic")
    for topic in topics:
        if not isinstance(topic, Topic):
            raise TypeError(f"topics must be Topic objects, got {type(topic).__name__}")
    ids = [t.topic_id for t in topics]
    if GENERAL_TOPIC_ID in ids:
        raise ValueError(f"{GENERAL_TOPIC_ID} is reserved for the automatic fallback topic")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate topic_id in taxonomy")
    if len(topics) + 1 > MAX_TOPICS_PER_ROOM:
        raise ValueError(
            f"too many topics ({len(topics)}); at most {MAX_TOPICS_PER_ROOM - 1} plus the general fallback"
        )
    return topics + (Topic(GENERAL_TOPIC_ID, GENERAL_TOPIC_NAME),)


class TopicManager:
    """Per-room fixed taxonomies + a replaceable classifier."""

    def __init__(
        self,
        default_topics: Sequence[Topic] = DEFAULT_TOPICS,
        classifier: TopicClassifier | None = None,
    ) -> None:
        self._default = _finalise(default_topics)
        self._rooms: dict[str, tuple[Topic, ...]] = {}
        self._classifier: TopicClassifier = classifier or KeywordTopicClassifier()

    def register_room_topics(self, room_id: str, topics: Sequence[Topic]) -> None:
        """Give ``room_id`` its own taxonomy. Re-registering a different one is
        refused, because topic ids must stay stable once chunks carry them."""
        _require_non_empty_str(room_id, "room_id")
        final = _finalise(topics)
        existing = self._rooms.get(room_id)
        if existing is not None and existing != final:
            raise ValueError(
                f"room {room_id!r} already has a different taxonomy; topics must stay stable"
            )
        self._rooms[room_id] = final

    def topics_for(self, room_id: str) -> tuple[Topic, ...]:
        _require_non_empty_str(room_id, "room_id")
        return self._rooms.get(room_id, self._default)

    def list_topics(self, room_id: str) -> list[dict[str, str]]:
        return [{"topic_id": t.topic_id, "name": t.name} for t in self.topics_for(room_id)]

    def tag_topic(self, room_id: str, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        topics = self.topics_for(room_id)
        if not text.strip():
            return GENERAL_TOPIC_ID
        topic_id = self._classifier.classify(text, topics)
        # A bad classifier result is a bug to surface, never something to paper over
        # with topic_general.
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError(
                f"classifier returned an invalid topic id {topic_id!r}; expected a non-empty string"
            )
        if topic_id not in {t.topic_id for t in topics}:
            raise ValueError(
                f"classifier returned {topic_id!r}, which is not a topic of room {room_id!r}"
            )
        return topic_id

    def tag_chunks(
        self, room_id: str, chunks: Sequence[Mapping[str, Any]], *, overwrite: bool = False
    ) -> list[dict[str, Any]]:
        """Return copies of ``chunks`` with ``topic_id`` added.

        Inputs are not mutated. Every chunk must belong to ``room_id``.

        ``topic_id`` handling:
        - missing, ``None`` or ``""``: classified.
        - already set and ``overwrite=False``: kept, but only if it is a topic of this
          room's taxonomy; otherwise ``ValueError`` (a stale or foreign id is never
          silently preserved).
        - ``overwrite=True``: always reclassified, whatever the old value was.

        Either every chunk is returned with a valid ``topic_id`` or an exception is
        raised; no partial result is ever returned.
        """
        _require_non_empty_str(room_id, "room_id")
        valid_ids = {t.topic_id for t in self.topics_for(room_id)}
        tagged: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping) or "text" not in chunk:
                raise ValueError(f"chunks[{i}] must be a dict with a 'text' field")
            if chunk.get("room_id") != room_id:
                raise ValueError(
                    f"chunks[{i}] has room_id {chunk.get('room_id')!r}, expected {room_id!r}"
                )
            new_chunk = dict(chunk)
            existing = new_chunk.get("topic_id")
            if overwrite or existing is None or existing == "":
                new_chunk["topic_id"] = self.tag_topic(room_id, chunk["text"])
            elif not isinstance(existing, str) or existing not in valid_ids:
                raise ValueError(
                    f"chunks[{i}] has topic_id {existing!r}, which is not a topic of room "
                    f"{room_id!r}; fix it or pass overwrite=True to reclassify"
                )
            tagged.append(new_chunk)
        return tagged


# ------------------------------------------------- module-level team contract

_default_manager = TopicManager()


def get_default_manager() -> TopicManager:
    return _default_manager


def tag_topic(room_id: str, text: str) -> str:
    """Topic id for ``text`` in ``room_id`` (deterministic)."""
    return _default_manager.tag_topic(room_id, text)


def list_topics(room_id: str) -> list[dict[str, str]]:
    """``[{"topic_id": ..., "name": ...}]`` for ``room_id``."""
    return _default_manager.list_topics(room_id)