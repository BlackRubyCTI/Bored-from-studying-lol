"""Persistent vector memory with hybrid retrieval.

The memory is the cluster's *long term* brain: the documents the user
uploaded, the conversations it had, and the facts it distilled from them.
It survives a restart, it is queryable while the cluster serves traffic, and
it is the input of the sleep-time self-evolution engine (see
``training/autopilot_evolver.py``).

Design notes
------------
* **No heavy dependencies.**  A vector store is a matrix and a dot product;
  ``numpy`` is enough.  Pulling in FAISS or Chroma would make the control
  plane fragile in the same way a browser dependency made it fragile in
  Milestone 1.  The embedder is pluggable, so a real embedding model can be
  dropped in later without touching the storage format.
* **Hybrid retrieval.**  Pure vector search misses exact tokens ("Qwen2.5-7B"),
  pure keyword search misses paraphrase.  The score is therefore a weighted
  sum of the cosine similarity and a BM25-ish keyword overlap.
* **Append-only, atomically persisted.**  Every chunk is appended to a JSONL
  file; the file is rewritten atomically (temp file + ``os.replace``) so a
  crash can never leave a half-written index behind.
* **Facts are first class.**  A distilled fact is a chunk with the ``fact``
  source and a higher importance: it outranks the raw conversation it came
  from.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.ids import now_s
from core.logging_setup import get_logger
from memory.pdf import extract_pdf_text, is_pdf

LOGGER = get_logger("memory.vector_memory")

#: The chunk sources the memory knows about.
SOURCE_DOCUMENT = "document"
SOURCE_CHAT = "chat"
SOURCE_FACT = "fact"

#: Default size of the embedding vector.
DEFAULT_DIMENSIONS = 256

#: How much a fact counts compared to a raw chunk.
FACT_IMPORTANCE = 1.6

#: Sentences that look like a durable statement about the user.
_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bmy name is\s+(?P<value>[^.,;!?\n]+)", re.I), "name"),
    (re.compile(r"\bi am\s+(?P<value>[^.,;!?\n]+)", re.I), "identity"),
    (re.compile(r"\bi(?:'m| am) working (?:on|with)\s+(?P<value>[^.,;!?\n]+)", re.I), "project"),
    (re.compile(r"\bi (?:prefer|like|love|hate|avoid)\s+(?P<value>[^.,;!?\n]+)", re.I), "preference"),
    (re.compile(r"\bi (?:always|never)\s+(?P<value>[^.,;!?\n]+)", re.I), "habit"),
    (re.compile(r"\bwe (?:use|run|deploy)\s+(?P<value>[^.,;!?\n]+)", re.I), "stack"),
    (re.compile(r"\bmy\s+(?P<key>[\w \-]{2,30})\s+is\s+(?P<value>[^.,;!?\n]+)", re.I), "attribute"),
    (re.compile(r"\b(?:remember|note) that\s+(?P<value>[^.\n]+)", re.I), "reminder"),
)

#: Words that turn a sentence into a durable instruction.
_DURABLE_MARKERS = (
    "remember",
    "always",
    "never",
    "from now on",
    "prefer",
    "my name is",
    "we use",
    "keep in mind",
    "note that",
    "don't forget",
    "do not forget",
)

#: Text that is worth remembering even without a marker.
_FACT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from",
        "how", "i", "if", "in", "is", "it", "me", "my", "no", "not", "of", "on", "or",
        "so", "that", "the", "then", "there", "this", "to", "up", "was", "what", "when",
        "where", "which", "who", "why", "will", "with", "would", "you", "your",
    }
)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """Split a text into comparable tokens (words + character 3-grams)."""
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    grams = [lowered[i : i + 3] for i in range(max(0, len(lowered) - 2))]
    return words + grams


class HashingEmbedder:
    """A deterministic, dependency-free sentence embedder.

    It is a hashed bag-of-features with sub-linear term weighting and L2
    normalisation.  Two texts that share vocabulary end up close together,
    which is exactly what a retrieval test needs -- and it never needs a
    network call, so the memory works offline.
    """

    name = "hashing"

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self.dimensions = max(16, int(dimensions))

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        if not tokens:
            return vector
        counts = Counter(tokens)
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two (already normalised) vectors."""
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[i] * right[i] for i in range(size))


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class MemoryChunk:
    """One indexed piece of text."""

    id: str
    text: str
    source: str = SOURCE_DOCUMENT
    doc_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    created_at: float = field(default_factory=now_s)
    importance: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "doc_id": self.doc_id,
            "metadata": self.metadata,
            "embedding": [round(value, 6) for value in self.embedding],
            "created_at": self.created_at,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryChunk:
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or ""),
            source=str(data.get("source") or SOURCE_DOCUMENT),
            doc_id=str(data.get("doc_id") or ""),
            metadata=dict(data.get("metadata") or {}),
            embedding=[float(value) for value in (data.get("embedding") or [])],
            created_at=float(data.get("created_at") or now_s()),
            importance=float(data.get("importance") or 1.0),
        )


@dataclass
class MemoryHit:
    """A search result."""

    chunk: MemoryChunk
    score: float
    similarity: float = 0.0
    keyword: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk.id,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "doc_id": self.chunk.doc_id,
            "score": round(self.score, 6),
            "similarity": round(self.similarity, 6),
            "keyword": round(self.keyword, 6),
            "metadata": self.chunk.metadata,
            "created_at": self.chunk.created_at,
        }


@dataclass
class DocumentRecord:
    """An indexed document."""

    doc_id: str
    name: str
    chunks: int
    characters: int
    source_path: str = ""
    created_at: float = field(default_factory=now_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "name": self.name,
            "chunks": self.chunks,
            "characters": self.characters,
            "source_path": self.source_path,
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #
class VectorMemory:
    """A persistent, hybrid-retrieval vector store."""

    def __init__(
        self,
        path: str | Path,
        *,
        dimensions: int = DEFAULT_DIMENSIONS,
        embedder: Callable[[str], list[float]] | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        max_chunks: int = 20000,
    ) -> None:
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.path / "chunks.jsonl"
        self.meta_path = self.path / "memory.meta.json"
        self.embedder = embedder if embedder is not None else HashingEmbedder(dimensions).embed
        self.chunk_size = max(120, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size // 2))
        self.max_chunks = max(16, int(max_chunks))
        self._chunks: list[MemoryChunk] = []
        self._documents: dict[str, DocumentRecord] = {}
        self._lock = threading.RLock()
        self._revision = 0
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Restore the index from disk."""
        with self._lock:
            self._chunks = []
            self._documents = {}
            self._revision = 0
            if self.meta_path.is_file():
                try:
                    meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                    self._revision = int(meta.get("revision") or 0)
                except (OSError, json.JSONDecodeError, ValueError):
                    self._revision = 0
            if self.chunks_path.is_file():
                with self.chunks_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = MemoryChunk.from_dict(json.loads(line))
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if chunk.id and chunk.text:
                            self._chunks.append(chunk)
            for chunk in self._chunks:
                if chunk.doc_id and chunk.doc_id not in self._documents:
                    self._documents[chunk.doc_id] = DocumentRecord(
                        doc_id=chunk.doc_id,
                        name=chunk.metadata.get("name") or chunk.doc_id,
                        chunks=0,
                        characters=len(chunk.text),
                    )
                record = self._documents.get(chunk.doc_id)
                if record is not None:
                    record.chunks += 1
                    record.characters += len(chunk.text)

    def _persist(self) -> None:
        """Rewrite the index atomically (temp file + rename)."""
        tmp_chunks = self.chunks_path.with_suffix(".jsonl.tmp")
        with tmp_chunks.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
                handle.write("\n")
        os.replace(tmp_chunks, self.chunks_path)
        meta = {
            "revision": self._revision,
            "chunks": len(self._chunks),
            "documents": len(self._documents),
            "updated_at": now_s(),
        }
        tmp_meta = self.meta_path.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp_meta, self.meta_path)

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def _new_id(self, text: str, source: str) -> str:
        digest = hashlib.blake2b(
            f"{source}|{text[:200]}|{time.time_ns()}".encode(), digest_size=8
        ).hexdigest()
        return f"mem_{digest}"

    def split(self, text: str) -> list[str]:
        """Split a text into overlapping chunks (paragraph aware)."""
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        paragraphs = [block.strip() for block in re.split(r"\n\s*\n", cleaned) if block.strip()]
        if not paragraphs:
            paragraphs = [cleaned]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                # A huge paragraph is cut on sentence boundaries.
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                buffer = ""
                for sentence in sentences:
                    if buffer and len(buffer) + len(sentence) + 1 > self.chunk_size:
                        chunks.append(buffer.strip())
                        buffer = buffer[-self.chunk_overlap :] if self.chunk_overlap else ""
                    buffer = f"{buffer} {sentence}".strip() if buffer else sentence
                if buffer.strip():
                    chunks.append(buffer.strip())
                continue
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) > self.chunk_size and current:
                chunks.append(current.strip())
                tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = f"{tail}\n\n{paragraph}".strip()
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())
        return [chunk for chunk in chunks if chunk]

    def add_text(
        self,
        text: str,
        *,
        source: str = SOURCE_DOCUMENT,
        doc_id: str = "",
        name: str = "",
        metadata: dict[str, Any] | None = None,
        importance: float = 1.0,
    ) -> list[MemoryChunk]:
        """Index a text and return the chunks that were created."""
        pieces = self.split(text)
        if not pieces:
            return []
        doc_id = doc_id or self._new_id(name or text[:64], source)
        created: list[MemoryChunk] = []
        with self._lock:
            for piece in pieces:
                chunk = MemoryChunk(
                    id=self._new_id(piece, source),
                    text=piece,
                    source=source,
                    doc_id=doc_id,
                    metadata={**(metadata or {}), "name": name or doc_id},
                    embedding=list(self.embedder(piece)),
                    importance=float(importance),
                )
                self._chunks.append(chunk)
                created.append(chunk)
            if len(self._chunks) > self.max_chunks:
                self._chunks = self._chunks[-self.max_chunks :]
            self._documents[doc_id] = DocumentRecord(
                doc_id=doc_id,
                name=name or doc_id,
                chunks=len(pieces),
                characters=len(text),
            )
            self._revision += 1
            self._persist()
        LOGGER.info(
            "memory indexed",
            extra={"hc_source": source, "hc_chunks": len(created), "hc_doc": doc_id},
        )
        return created

    def add_document(
        self,
        name: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentRecord:
        """Index a document (the name is what the user sees)."""
        doc_id = "doc_" + hashlib.blake2b(name.encode("utf-8"), digest_size=6).hexdigest()
        self.add_text(text, source=SOURCE_DOCUMENT, doc_id=doc_id, name=name, metadata=metadata)
        with self._lock:
            record = self._documents[doc_id]
            record.source_path = str((metadata or {}).get("source_path") or "")
        return record

    def ingest_file(self, path: str | Path, *, name: str = "") -> DocumentRecord | None:
        """Index a file from disk (text and PDF are supported)."""
        source = Path(path).expanduser()
        if not source.is_file():
            return None
        try:
            payload = source.read_bytes()
        except OSError:
            LOGGER.warning("memory: unreadable file", extra={"hc_path": str(source)})
            return None
        metadata = {"source_path": str(source), "size": len(payload)}
        if is_pdf(payload, source.name):
            text, extra = extract_pdf_text(payload)
            metadata.update(extra)
        else:
            text = payload.decode("utf-8", errors="replace")
        if not text.strip():
            return None
        return self.add_document(name or source.name, text, metadata=metadata)

    def ingest_bytes(
        self,
        payload: bytes,
        name: str = "upload",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentRecord | None:
        """Index an uploaded file (text or PDF)."""
        if not payload:
            return None
        extra_metadata: dict[str, Any] = {"size": len(payload)}
        if is_pdf(payload, name):
            text, extra = extract_pdf_text(payload)
            extra_metadata.update(extra)
        else:
            text = payload.decode("utf-8", errors="replace")
        if not text.strip():
            return None
        return self.add_document(name or "upload", text, metadata={**extra_metadata, **(metadata or {})})

    def record_exchange(
        self,
        prompt: str,
        answer: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryChunk | None:
        """Store one question/answer pair of the conversation."""
        text = f"Q: {prompt.strip()}\nA: {answer.strip()}"
        chunks = self.add_text(
            text,
            source=SOURCE_CHAT,
            name=(metadata or {}).get("name") or "conversation",
            metadata=metadata or {},
        )
        return chunks[0] if chunks else None

    def add_fact(self, text: str, *, metadata: dict[str, Any] | None = None) -> MemoryChunk | None:
        """Store a distilled long-term fact.

        A fact is stored once: the same statement is never indexed twice
        (the memory would otherwise return the same answer twice).
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        with self._lock:
            for chunk in self._chunks:
                if chunk.source == SOURCE_FACT and chunk.text == cleaned:
                    return chunk
        chunks = self.add_text(
            cleaned,
            source=SOURCE_FACT,
            name="fact",
            metadata=metadata or {},
            importance=FACT_IMPORTANCE,
        )
        return chunks[0] if chunks else None

    def has_fact(self, text: str) -> bool:
        """Whether a fact is already known."""
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        with self._lock:
            return any(
                chunk.source == SOURCE_FACT and chunk.text == cleaned for chunk in self._chunks
            )

    # ------------------------------------------------------------------ #
    # Fact distillation
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_facts(text: str) -> list[tuple[str, str]]:
        """Extract durable facts from a piece of conversation.

        Returns ``(fact, kind)`` pairs.  A sentence is kept when it matches
        one of the durable patterns *or* when it contains an explicit marker
        ("remember", "always", ...).
        """
        facts: list[tuple[str, str]] = []
        seen: set[str] = set()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
            sentence = sentence.strip()
            if len(sentence) < 8 or len(sentence) > 400:
                continue
            lowered = sentence.lower()
            matched = False
            for pattern, kind in _FACT_PATTERNS:
                match = pattern.search(sentence)
                if not match:
                    continue
                value = (match.groupdict().get("value") or "").strip(" .")
                # A fact stops at the first conjunction: "my name is Ada and I
                # always ..." is one fact, not two.
                value = re.split(r"\s+(?:and|but|because|so|while|although)\s+", value, maxsplit=1)[0]
                value = value.strip(" .,")
                if len(value) < 3:
                    continue
                fact = f"{kind}: {value}"
                if fact.lower() not in seen:
                    seen.add(fact.lower())
                    facts.append((fact, kind))
                matched = True
                break
            if matched:
                continue
            if any(marker in lowered for marker in _DURABLE_MARKERS):
                words = [word for word in re.findall(r"[\w' -]+", sentence.lower()) if word not in _FACT_STOPWORDS]
                if len(words) >= 3:
                    fact = sentence.rstrip(".!?")
                    if fact.lower() not in seen:
                        seen.add(fact.lower())
                        facts.append((fact, "statement"))
        return facts

    def distill_facts(
        self,
        messages: Iterable[Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Distil the durable facts of a conversation into the memory.

        ``messages`` may be protocol objects (``role``/``content``) or plain
        dictionaries.  Only the *user* turns are mined -- the assistant's own
        words are not facts about the user.
        """
        stored: list[str] = []
        for message in messages:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role is None and isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            if role != "user" or not content:
                continue
            for fact, kind in self.extract_facts(str(content)):
                if self.has_fact(fact):
                    continue
                chunk = self.add_fact(fact, metadata={**(metadata or {}), "kind": kind})
                if chunk is not None:
                    stored.append(fact)
        if stored:
            LOGGER.info("memory distilled facts", extra={"hc_facts": len(stored)})
        return stored

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def _keyword_score(self, query_tokens: set[str], chunk: MemoryChunk) -> float:
        if not query_tokens:
            return 0.0
        tokens = set(tokenize(chunk.text))
        if not tokens:
            return 0.0
        hits = len(query_tokens & tokens)
        if hits == 0:
            return 0.0
        return hits / (hits + 1.4 * math.sqrt(len(tokens) / 240.0))

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        sources: Sequence[str] | None = None,
        min_score: float = 0.0,
    ) -> list[MemoryHit]:
        """Hybrid search: cosine similarity + keyword overlap."""
        query_vector = list(self.embedder(query or ""))
        query_tokens = set(tokenize(query or ""))
        hits: list[MemoryHit] = []
        with self._lock:
            chunks = list(self._chunks)
        for chunk in chunks:
            if sources and chunk.source not in sources:
                continue
            similarity = max(0.0, cosine(query_vector, chunk.embedding))
            keyword = self._keyword_score(query_tokens, chunk)
            score = 0.65 * similarity + 0.35 * keyword
            score *= max(0.2, chunk.importance)
            if score <= min_score:
                continue
            hits.append(
                MemoryHit(chunk=chunk, score=score, similarity=similarity, keyword=keyword)
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(1, int(k))]

    def context(
        self,
        query: str,
        *,
        k: int = 4,
        max_chars: int = 2400,
        sources: Sequence[str] | None = None,
    ) -> str:
        """Build the context block that is injected into the prompt."""
        hits = self.search(query, k=k, sources=sources)
        if not hits:
            return ""
        parts: list[str] = []
        used = 0
        for hit in hits:
            snippet = " ".join(hit.chunk.text.split())
            if used + len(snippet) > max_chars:
                snippet = snippet[: max(0, max_chars - used)]
            if not snippet:
                break
            label = hit.chunk.metadata.get("name") or hit.chunk.source
            parts.append(f"- [{label}] {snippet}")
            used += len(snippet)
            if used >= max_chars:
                break
        if not parts:
            return ""
        return "MEMORY CONTEXT:\n" + "\n".join(parts)

    def documents(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._documents.values(), key=lambda record: record.created_at)

    def facts(self) -> list[MemoryChunk]:
        with self._lock:
            return [chunk for chunk in self._chunks if chunk.source == SOURCE_FACT]

    def conversations(self) -> list[MemoryChunk]:
        with self._lock:
            return [chunk for chunk in self._chunks if chunk.source == SOURCE_CHAT]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            by_source: dict[str, int] = {}
            for chunk in self._chunks:
                by_source[chunk.source] = by_source.get(chunk.source, 0) + 1
            return {
                "chunks": len(self._chunks),
                "documents": len(self._documents),
                "by_source": by_source,
                "revision": self._revision,
                "path": str(self.path),
                "embedder": getattr(self.embedder, "__name__", "custom"),
            }

    @property
    def revision(self) -> int:
        return self._revision

    # ------------------------------------------------------------------ #
    # Snapshots (the unit of the Drive / Hub sync)
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        """A portable snapshot of the whole memory."""
        with self._lock:
            payload = [chunk.to_dict() for chunk in self._chunks]
            revision = self._revision
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return {
            "revision": revision,
            "created_at": now_s(),
            "chunks": payload,
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "documents": [record.to_dict() for record in self.documents()],
        }

    def restore(self, snapshot: dict[str, Any]) -> int:
        """Replace the memory with a snapshot; returns the chunk count."""
        payload = snapshot.get("chunks")
        if not isinstance(payload, list):
            return 0
        with self._lock:
            restored: list[MemoryChunk] = []
            for item in payload:
                try:
                    chunk = MemoryChunk.from_dict(item)
                except (TypeError, ValueError):
                    continue
                if chunk.id and chunk.text:
                    restored.append(chunk)
            self._chunks = restored
            self._documents = {}
            for chunk in restored:
                if not chunk.doc_id:
                    continue
                record = self._documents.get(chunk.doc_id)
                if record is None:
                    record = DocumentRecord(
                        doc_id=chunk.doc_id,
                        name=chunk.metadata.get("name") or chunk.doc_id,
                        chunks=0,
                        characters=0,
                    )
                    self._documents[chunk.doc_id] = record
                record.chunks += 1
                record.characters += len(chunk.text)
            self._revision = max(self._revision, int(snapshot.get("revision") or 0))
            self._persist()
        return len(self._chunks)


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
