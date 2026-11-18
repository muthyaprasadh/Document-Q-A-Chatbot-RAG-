from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from endee import Endee
except ImportError:
    Endee = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class DocumentStore:
    """Small reusable store that works with Endee or an in-memory fallback."""

    def __init__(self, index_name: str = "documents") -> None:
        self.chunks: list[str] = []
        self.vectors: list[list[float]] = []
        self.endee = None
        self.model = None
        self._model_error: str | None = None

        if Endee is not None:
            try:
                self.endee = Endee(index_name=index_name)
            except Exception as exc:
                self._model_error = f"Endee unavailable: {exc}"

    def _get_model(self):
        if self.model is not None:
            return self.model
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:
                self._model_error = f"Embedding model unavailable: {exc}"
        return self.model

    def _embed(self, text: str) -> list[float] | None:
        model = self._get_model()
        if model is None:
            return None
        vector = model.encode(text)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def add_text(self, text: str) -> int:
        added = 0
        for chunk in chunk_text(text):
            if chunk not in self.chunks:
                self.chunks.append(chunk)
                vector = self._embed(chunk)
                if vector is not None:
                    self.vectors.append(vector)
                else:
                    self.vectors.append([])
                if self.endee is not None and vector is not None:
                    self.endee.add(vector=vector, metadata={"text": chunk})
                added += 1
        return added

    def search(self, query: str, top_k: int = 5) -> list[str]:
        query_vector = self._embed(query)

        if self.endee is not None and query_vector is not None:
            try:
                results = self.endee.search(vector=query_vector, k=top_k)
                texts: list[str] = []
                for result in results:
                    if isinstance(result, dict):
                        metadata = result.get("metadata", {})
                    else:
                        metadata = getattr(result, "metadata", {})
                    if isinstance(metadata, dict) and metadata.get("text"):
                        texts.append(str(metadata["text"]))
                    elif getattr(result, "text", None):
                        texts.append(str(result.text))
                if texts:
                    return texts
            except Exception:
                pass

        if query_vector is not None:
            from .similarity import cosine_similarity

            scored = []
            for chunk, vector in zip(self.chunks, self.vectors):
                if vector:
                    scored.append((cosine_similarity(query_vector, vector), chunk))
            if scored:
                return [chunk for _, chunk in sorted(scored, reverse=True)[:top_k]]

        return keyword_search(self.chunks, query, top_k)


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text into sentence-preserving chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def keyword_search(chunks: Iterable[str], query: str, top_k: int = 5) -> list[str]:
    query_terms = set(re.findall(r"\w+", query.lower()))
    if not query_terms:
        return list(chunks)[:top_k]

    scored = []
    for chunk in chunks:
        terms = set(re.findall(r"\w+", chunk.lower()))
        score = len(query_terms & terms)
        if score:
            scored.append((score, chunk))
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
