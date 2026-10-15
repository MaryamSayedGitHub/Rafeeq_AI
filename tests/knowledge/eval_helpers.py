"""Shared helpers for the evaluation tests. NOT production code."""

import importlib.util
import sys
from pathlib import Path

from kb_fakes import HashingEmbedder
from rafeeq.knowledge.ingestion.chunker import chunk_pages
from rafeeq.knowledge.topics import TopicManager
from rafeeq.knowledge.vector_store import FAISSVectorStore

SCRIPTS = Path(__file__).resolve().parents[2] / "rafeeq" / "knowledge" / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_store(directory, rooms, embedder=None):
    """rooms = {room_id: {file_name: [page text, ...]}}; saves a real store and returns it."""
    embedder = embedder or HashingEmbedder()
    store = FAISSVectorStore(embedder.dimension)
    manager = TopicManager()
    for room_id, files in rooms.items():
        for file_name, texts in files.items():
            pages = [{"page": i + 1, "text": t} for i, t in enumerate(texts)]
            chunks = manager.tag_chunks(room_id, chunk_pages(pages, room_id, file_name))
            store.add(embedder.embed_documents([c["text"] for c in chunks]), chunks)
    store.save(directory)
    return store


BOOK = [
    "Self-attention lets every token attend to the other tokens in the sequence.",
    "Convolutional neural networks process images with learned filters.",
    "Gradient descent optimizes the weights of neural networks step by step.",
    "Retrieval-augmented generation fetches documents before answering a question.",
    "LoRA adapts large models by training small low-rank matrices only.",
    "Tokenizers split raw text into tokens before embedding.",
]
OTHER = ["Self-attention is also mentioned in this second document about a different course."]