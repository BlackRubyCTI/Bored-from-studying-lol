"""Persistent vector memory with hybrid retrieval (Milestone 4).

``memory.vector_memory`` holds the implementation; this package only
re-exports the public names.
"""

from __future__ import annotations

from memory.vector_memory import (
    DEFAULT_DIMENSIONS,
    FACT_IMPORTANCE,
    SOURCE_CHAT,
    SOURCE_DOCUMENT,
    SOURCE_FACT,
    DocumentRecord,
    HashingEmbedder,
    MemoryChunk,
    MemoryHit,
    VectorMemory,
    cosine,
    tokenize,
)

__all__ = [
    "DEFAULT_DIMENSIONS",
    "FACT_IMPORTANCE",
    "SOURCE_CHAT",
    "SOURCE_DOCUMENT",
    "SOURCE_FACT",
    "DocumentRecord",
    "HashingEmbedder",
    "MemoryChunk",
    "MemoryHit",
    "VectorMemory",
    "cosine",
    "tokenize",
]
